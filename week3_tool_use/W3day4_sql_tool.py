import anthropic
import sqlite3
import json
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic()

# ============================================================
# TOOL DEFINITION — Claude can run SQL queries
# ============================================================

tools = [
    {
        "name": "run_sql_query",
        "description": "Executes a SQL query on the sales database. The database has two tables: 'customers' (id, name, region, signup_date) and 'orders' (id, customer_id, amount, product, order_date). Use SELECT queries only.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "SQL SELECT query to execute"}
            },
            "required": ["query"]
        }
    }
]

# ============================================================
# TOOL IMPLEMENTATION — actually runs the SQL
# ============================================================

def run_sql_query(query):
    # Safety check — only allow SELECT queries
    if not query.strip().upper().startswith("SELECT"):
        return json.dumps({"error": "Only SELECT queries are allowed"})

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

tool_functions = {"run_sql_query": run_sql_query}

# ============================================================
# CHAINING LOOP — same pattern from Day 3
# ============================================================

def ask_database(question):
    print(f"\nUser: {question}")
    print("-" * 50)

    messages = [{"role": "user", "content": question}]
    step = 1

    while True:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=400,
            temperature=0,
            tools=tools,
            messages=messages
        )

        if response.stop_reason == "end_turn":
            final_text = next((b.text for b in response.content if hasattr(b, "text")), "")
            print(f"\nAnswer: {final_text}")
            break

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                func = tool_functions[block.name]
                result = func(**block.input)
                print(f"  Step {step}: SQL → {block.input['query']}")

                # Show the data returned
                parsed = json.loads(result)
                if "data" in parsed:
                    print(f"  Result: {parsed['row_count']} rows returned")
                else:
                    print(f"  Error: {parsed['error']}")

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result
                })
                step += 1

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

# ============================================================
# TEST — Ask questions in plain English
# ============================================================

print("=" * 60)
print("  SQL TOOL: Ask your database in plain English")
print("=" * 60)

ask_database("How many customers are in each region?")
ask_database("What is the total revenue by product?")
ask_database("Who are the top 3 customers by total spending?")
ask_database("What is the average order value for the North region?")
ask_database("Show me monthly revenue for 2025")