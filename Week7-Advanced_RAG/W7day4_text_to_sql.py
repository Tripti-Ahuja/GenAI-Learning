import anthropic
import sqlite3
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dotenv import load_dotenv

load_dotenv()
claude_client = anthropic.Anthropic()

print("Loading model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Ready!\n")

# ============================================================
# STEP 1: Extract schema info as searchable descriptions
# ============================================================

def get_schema_descriptions():
    conn = sqlite3.connect("sales.db")
    cursor = conn.cursor()

    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]

    descriptions = []
    for table in tables:
        cursor.execute(f"PRAGMA table_info({table})")
        columns = cursor.fetchall()
        col_info = ", ".join([f"{col[1]} ({col[2]})" for col in columns])

        # Get sample data
        cursor.execute(f"SELECT * FROM {table} LIMIT 3")
        sample_rows = cursor.fetchall()
        sample_text = str(sample_rows[:2])

        description = {
            "table": table,
            "columns": col_info,
            "description": f"Table '{table}' has columns: {col_info}. Sample data: {sample_text}",
            "detail": f"Table: {table}\nColumns: {col_info}\nSample rows: {sample_text}"
        }
        descriptions.append(description)

    conn.close()
    return descriptions

schema_descriptions = get_schema_descriptions()

print("Database schema:")
for desc in schema_descriptions:
    print(f"  {desc['table']}: {desc['columns']}")

# Embed schema descriptions for retrieval
schema_texts = [desc["description"] for desc in schema_descriptions]
schema_embeddings = embed_model.encode(schema_texts)

# ============================================================
# STEP 2: Retrieve relevant tables for a question
# ============================================================

def retrieve_relevant_tables(query):
    query_embedding = embed_model.encode([query])
    scores = cosine_similarity(query_embedding, schema_embeddings)[0]

    relevant = []
    for i, score in enumerate(scores):
        if score > 0.1:
            relevant.append({
                "table": schema_descriptions[i]["table"],
                "detail": schema_descriptions[i]["detail"],
                "score": round(float(score), 3)
            })

    relevant.sort(key=lambda x: x["score"], reverse=True)
    return relevant

# ============================================================
# STEP 3: Generate SQL from natural language
# ============================================================

def generate_sql(query, relevant_tables):
    schema_context = "\n\n".join([t["detail"] for t in relevant_tables])

    prompt = f"""You are a SQL expert. Generate a SQLite SELECT query to answer the user's question.

DATABASE SCHEMA:
{schema_context}

RULES:
1. Return ONLY the SQL query, no explanation, no markdown
2. Use SQLite syntax
3. Only SELECT queries
4. Use proper JOINs when combining tables

QUESTION: {query}

SQL:"""

    response = claude_client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text.strip().replace("```sql", "").replace("```", "").strip()

# ============================================================
# STEP 4: Execute SQL safely
# ============================================================

def execute_sql(query):
    if not query.strip().upper().startswith("SELECT"):
        return {"error": "Only SELECT allowed"}
    try:
        conn = sqlite3.connect("sales.db")
        cursor = conn.cursor()
        cursor.execute(query)
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        conn.close()
        return {"columns": columns, "rows": rows, "count": len(rows)}
    except Exception as e:
        return {"error": str(e)}

# ============================================================
# STEP 5: Summarize results in natural language
# ============================================================

def summarize_results(question, sql, results):
    if "error" in results:
        return f"SQL Error: {results['error']}"

    data_text = json.dumps({"columns": results["columns"], "data": [dict(zip(results["columns"], row)) for row in results["rows"][:10]]}, indent=2)

    prompt = f"""Summarize this database query result in plain English. Be concise — 2-3 sentences.

QUESTION: {question}
SQL USED: {sql}
RESULTS:
{data_text}

SUMMARY:"""

    response = claude_client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text

# ============================================================
# STEP 6: Complete Text-to-SQL RAG pipeline
# ============================================================

def ask(question):
    print(f"\nQ: {question}")
    print("-" * 55)

    # Retrieve relevant tables
    tables = retrieve_relevant_tables(question)
    print(f"  Tables found: {[t['table'] + ' (' + str(t['score']) + ')' for t in tables]}")

    # Generate SQL
    sql = generate_sql(question, tables)
    print(f"  SQL: {sql}")

    # Execute
    results = execute_sql(sql)
    if "error" in results:
        print(f"  Error: {results['error']}")
        return

    print(f"  Rows returned: {results['count']}")

    # Summarize
    summary = summarize_results(question, sql, results)
    print(f"\n  Answer: {summary}")

# ============================================================
# TEST
# ============================================================

print("\n" + "=" * 60)
print("  TEXT-TO-SQL RAG")
print("=" * 60)

ask("How many customers do we have in each region?")
ask("What is the total revenue by product?")
ask("Who are our top 3 customers by spending?")
ask("What is the average order value per region?")
ask("Which month had the highest revenue in 2025?")
ask("Show me customers who have ordered more than one product type")