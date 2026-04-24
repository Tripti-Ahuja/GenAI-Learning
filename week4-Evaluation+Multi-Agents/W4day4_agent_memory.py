import anthropic
import sqlite3
import json
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic()

# ============================================================
# TOOLS (same as Day 3 + save_memory + recall_memory)
# ============================================================

tools = [
    {
        "name": "run_sql_query",
        "description": "Executes a READ-ONLY SQL SELECT query. Tables: 'customers' (id, name, region, signup_date) and 'orders' (id, customer_id, amount, product, order_date).",
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
                "expression": {"type": "string", "description": "Math expression"}
            },
            "required": ["expression"]
        }
    },
    {
        "name": "save_memory",
        "description": "Saves an important fact or result for later reference. Use this whenever you discover a key number, finding, or insight that might be useful for follow-up questions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Short label like 'north_revenue' or 'top_customer'"},
                "value": {"type": "string", "description": "The fact to remember"}
            },
            "required": ["key", "value"]
        }
    },
    {
        "name": "recall_memory",
        "description": "Retrieves all previously saved facts. Call this at the start of a new question to check what you already know before querying the database.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "task_complete",
        "description": "Call when you have fully answered the question.",
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "Final answer summary"}
            },
            "required": ["summary"]
        }
    }
]

# ============================================================
# MEMORY STORE + TOOL IMPLEMENTATIONS
# ============================================================

memory = {}

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

def save_memory(key, value):
    memory[key] = value
    return json.dumps({"saved": key, "value": value, "total_memories": len(memory)})

def recall_memory():
    if not memory:
        return json.dumps({"memories": "No facts saved yet."})
    return json.dumps({"memories": memory})

def task_complete(summary):
    return json.dumps({"status": "complete", "summary": summary})

tool_functions = {
    "run_sql_query": run_sql_query,
    "calculator": calculator,
    "save_memory": save_memory,
    "recall_memory": recall_memory,
    "task_complete": task_complete,
}

# ============================================================
# AGENT WITH MEMORY
# ============================================================

SYSTEM_PROMPT = """You are a data analyst agent with memory.

Rules:
1. At the START of each new question, call recall_memory to check what you already know
2. If memory has the data you need, use it instead of re-querying the database
3. After finding important results (revenue numbers, top customers, etc.), save them with save_memory
4. Think step by step before each action
5. Call task_complete when done"""

MAX_STEPS = 10
conversation_history = []

def run_agent(question):
    print(f"\n{'='*60}")
    print(f"  QUESTION: {question}")
    print(f"{'='*60}")

    conversation_history.append({"role": "user", "content": question})
    step = 1
    messages_for_turn = list(conversation_history)

    while step <= MAX_STEPS:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=500,
            temperature=0,
            tools=tools,
            system=SYSTEM_PROMPT,
            messages=messages_for_turn
        )

        for block in response.content:
            if hasattr(block, "text") and block.text:
                print(f"\n  THINK: {block.text[:150]}")

        if response.stop_reason == "end_turn":
            final_text = next((b.text for b in response.content if hasattr(b, "text")), "")
            print(f"\n  FINAL: {final_text[:200]}")
            conversation_history.append({"role": "assistant", "content": final_text})
            break

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                func = tool_functions[block.name]
                result = func(**block.input)

                if block.name == "task_complete":
                    parsed = json.loads(result)
                    print(f"\n  FINAL ANSWER: {parsed['summary']}")
                    conversation_history.append({"role": "assistant", "content": parsed['summary']})
                    messages_for_turn.append({"role": "assistant", "content": response.content})
                    messages_for_turn.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": block.id, "content": result}]})
                    return
                elif block.name == "save_memory":
                    print(f"\n  STEP {step} [SAVE]: '{block.input['key']}' = '{block.input['value']}'")
                elif block.name == "recall_memory":
                    parsed = json.loads(result)
                    mem_count = len(parsed.get("memories", {})) if isinstance(parsed.get("memories"), dict) else 0
                    print(f"\n  STEP {step} [RECALL]: {mem_count} facts in memory")
                elif block.name == "run_sql_query":
                    print(f"\n  STEP {step} [SQL]: {block.input['query'][:80]}")
                    parsed = json.loads(result)
                    if "row_count" in parsed:
                        print(f"    → {parsed['row_count']} rows")
                elif block.name == "calculator":
                    parsed = json.loads(result)
                    print(f"\n  STEP {step} [CALC]: {block.input['expression']} = {parsed.get('result', 'error')}")

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result
                })
                step += 1

        messages_for_turn.append({"role": "assistant", "content": response.content})
        messages_for_turn.append({"role": "user", "content": tool_results})

    if step > MAX_STEPS:
        print(f"\n  [STOPPED: Max steps reached]")

# ============================================================
# TEST — Conversation that benefits from memory
# ============================================================

print("=" * 60)
print("  AGENT WITH MEMORY")
print(f"  Memory store: {memory}")
print("=" * 60)

# Question 1 — Agent queries DB and saves key facts
run_agent("What is the total revenue by region?")

print(f"\n  [Memory after Q1: {memory}]")

# Question 2 — Agent should recall memory instead of re-querying
run_agent("Which region from the previous results had the lowest revenue?")

print(f"\n  [Memory after Q2: {memory}]")

# Question 3 — Agent should use saved data + calculator
run_agent("If the weakest region grew by 50%, would it beat the second weakest?")

print(f"\n  [Memory after Q3: {memory}]")