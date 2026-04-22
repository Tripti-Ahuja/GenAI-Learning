import anthropic
import sqlite3
import json
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic()

# ============================================================
# TOOLS
# ============================================================

tools = [
    {
        "name": "list_tables",
        "description": "Lists all tables in the database with their column names.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "run_sql_query",
        "description": "Executes a READ-ONLY SQL SELECT query on the sales database.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "SQL SELECT query"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "calculator",
        "description": "Evaluates a math expression.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression like '920000 * 1.15'"}
            },
            "required": ["expression"]
        }
    },
    {
        "name": "task_complete",
        "description": "Call this when you have fully answered the user's question and no more tool calls are needed.",
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "Final summary of what was found"}
            },
            "required": ["summary"]
        }
    }
]

# ============================================================
# TOOL IMPLEMENTATIONS
# ============================================================

def list_tables():
    conn = sqlite3.connect("sales.db")
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    result = {}
    for table in tables:
        cursor.execute(f"PRAGMA table_info({table})")
        columns = [row[1] for row in cursor.fetchall()]
        result[table] = columns
    conn.close()
    return json.dumps(result)

def run_sql_query(query):
    if not query.strip().upper().startswith("SELECT"):
        return json.dumps({"error": "Only SELECT queries allowed"})
    try:
        conn = sqlite3.connect("sales.db")
        cursor = conn.cursor()
        cursor.execute(query)
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        conn.close()
        results = [dict(zip(columns, row)) for row in rows]
        return json.dumps({"columns": columns, "row_count": len(rows), "data": results})
    except Exception as e:
        return json.dumps({"error": str(e)})

def calculator(expression):
    try:
        return json.dumps({"expression": expression, "result": eval(expression)})
    except Exception as e:
        return json.dumps({"error": str(e)})

def task_complete(summary):
    return json.dumps({"status": "complete", "summary": summary})

tool_functions = {
    "list_tables": list_tables,
    "run_sql_query": run_sql_query,
    "calculator": calculator,
    "task_complete": task_complete,
}

# ============================================================
# REACT AGENT WITH THINKING
# ============================================================

SYSTEM_PROMPT = """You are a data analyst agent. You solve problems step by step.

For each step:
1. THINK: Explain what you need to find out and why
2. ACT: Call the appropriate tool
3. OBSERVE: Review the result
4. REPEAT or call task_complete when done

Always start by listing tables if you don't know the schema.
Always call task_complete when you have the final answer."""

MAX_STEPS = 8

def run_agent(question):
    print(f"\n{'='*60}")
    print(f"  QUESTION: {question}")
    print(f"{'='*60}")

    messages = [{"role": "user", "content": question}]
    step = 1

    while step <= MAX_STEPS:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=500,
            temperature=0,
            tools=tools,
            system=SYSTEM_PROMPT,
            messages=messages
        )

        # Print any thinking text Claude produces
        for block in response.content:
            if hasattr(block, "text") and block.text:
                print(f"\n  THINK: {block.text}")

        if response.stop_reason == "end_turn":
            print(f"\n  [Agent finished]")
            break

        # Process tool calls
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                tool_name = block.name
                tool_input = block.input

                # Execute the tool
                func = tool_functions[tool_name]
                result = func(**tool_input)

                if tool_name == "task_complete":
                    parsed = json.loads(result)
                    print(f"\n  FINAL ANSWER: {parsed['summary']}")
                    messages.append({"role": "assistant", "content": response.content})
                    messages.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": block.id, "content": result}]})
                    return

                # Display the step
                if tool_name == "run_sql_query":
                    print(f"\n  STEP {step} [ACT]: run_sql_query")
                    print(f"    SQL: {tool_input['query']}")
                elif tool_name == "calculator":
                    print(f"\n  STEP {step} [ACT]: calculator({tool_input['expression']})")
                else:
                    print(f"\n  STEP {step} [ACT]: {tool_name}()")

                parsed = json.loads(result)
                if "error" in parsed:
                    print(f"    OBSERVE: Error - {parsed['error']}")
                elif "row_count" in parsed:
                    print(f"    OBSERVE: {parsed['row_count']} rows returned")
                elif "result" in parsed:
                    print(f"    OBSERVE: Result = {parsed['result']}")
                else:
                    print(f"    OBSERVE: {result[:100]}")

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result
                })
                step += 1

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

    if step > MAX_STEPS:
        print(f"\n  [STOPPED: Hit max {MAX_STEPS} steps]")

# ============================================================
# TEST THE AGENT
# ============================================================

run_agent("Which product generates the most revenue and how much more does it make than the least popular product?")

run_agent("What is the average order value per region? Which region is performing best?")

run_agent("Show me the top 3 customers. If each of them increased spending by 30%, what would the new totals be?")