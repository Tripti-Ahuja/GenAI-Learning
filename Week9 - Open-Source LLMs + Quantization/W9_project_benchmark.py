import requests
import anthropic
import json
import time
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import numpy as np
from dotenv import load_dotenv

load_dotenv()
claude = anthropic.Anthropic()

print("Loading embedding model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Ready!\n")

# ============================================================
# MODEL BACKENDS
# ============================================================

def ask_local(prompt, model="llama3.2:3b"):
    start = time.time()
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=120
        )
        elapsed = round(time.time() - start, 1)
        return response.json()["response"].strip(), elapsed, 0.0
    except Exception as e:
        return f"Error: {str(e)}", 0, 0

def ask_claude(prompt):
    start = time.time()
    response = claude.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    elapsed = round(time.time() - start, 1)
    cost = round((response.usage.input_tokens * 1.0 + response.usage.output_tokens * 5.0) / 1_000_000, 5)
    return response.content[0].text.strip(), elapsed, cost

# ============================================================
# TEST CATEGORIES
# ============================================================

tests = [
    {
        "category": "Factual Q&A",
        "tasks": [
            {"prompt": "What is a LEFT JOIN in SQL? Answer in one sentence.", "check": "left"},
            {"prompt": "What is the capital of Japan? Answer in one word.", "check": "tokyo"},
            {"prompt": "What does API stand for? Answer in one sentence.", "check": "application"},
        ]
    },
    {
        "category": "SQL Generation",
        "tasks": [
            {"prompt": "Write ONLY a SQL query (no explanation) to find total revenue by region from tables: customers(id, name, region) and orders(customer_id, amount).", "check": "select"},
            {"prompt": "Write ONLY a SQL query (no explanation) to find the top 3 customers by spending from tables: customers(id, name) and orders(customer_id, amount).", "check": "limit"},
            {"prompt": "Write ONLY a SQL query (no explanation) to count orders per product from table: orders(id, product, amount).", "check": "group by"},
        ]
    },
    {
        "category": "Classification",
        "tasks": [
            {"prompt": "Classify as positive, negative, or neutral. Reply with ONLY one word.\n\nReview: 'Absolutely love this product!'", "check": "positive"},
            {"prompt": "Classify as positive, negative, or neutral. Reply with ONLY one word.\n\nReview: 'Crashed 3 times today. Terrible.'", "check": "negative"},
            {"prompt": "Classify as positive, negative, or neutral. Reply with ONLY one word.\n\nReview: 'It works fine. Nothing special.'", "check": "neutral"},
        ]
    },
    {
        "category": "Summarization",
        "tasks": [
            {"prompt": "Summarize in one sentence: The north region led all regions with $920K in Q4 revenue, a 15% increase from Q3, driven by the new sales playbook and three Enterprise Plan deals.", "check": "north"},
            {"prompt": "Summarize in one sentence: Customer churn dropped to 3.2% in Q4 from 5.1% in Q3 thanks to the new onboarding process with dedicated account managers for 90 days.", "check": "churn"},
        ]
    },
    {
        "category": "RAG Grounding",
        "tasks": [
            {"prompt": "Answer based ONLY on this context. If not found, say 'Not found.'\n\nContext: Q4 revenue was $2.3 million.\n\nQuestion: What was Q4 revenue?", "check": "2.3"},
            {"prompt": "Answer based ONLY on this context. If not found, say 'Not found.'\n\nContext: Q4 revenue was $2.3 million.\n\nQuestion: What is our marketing budget?", "check": "not found"},
        ]
    },
]

# ============================================================
# RUN BENCHMARK
# ============================================================

models = [
    {"name": "Llama 3.2 3B", "func": lambda p: ask_local(p, "llama3.2:3b"), "type": "local"},
    {"name": "Mistral 7B", "func": lambda p: ask_local(p, "mistral"), "type": "local"},
    {"name": "Claude Haiku", "func": ask_claude, "type": "cloud"},
]

print("=" * 70)
print("  LOCAL MODEL BENCHMARK")
print("  Testing: Llama 3.2 (3B) vs Mistral (7B) vs Claude Haiku (cloud)")
print("=" * 70)

all_results = {m["name"]: {"pass": 0, "fail": 0, "total_time": 0, "total_cost": 0, "details": []} for m in models}

for test_group in tests:
    category = test_group["category"]
    print(f"\n{'='*70}")
    print(f"  CATEGORY: {category}")
    print(f"{'='*70}")

    for task in test_group["tasks"]:
        print(f"\n  Prompt: \"{task['prompt'][:60]}...\"")
        print(f"  Expected keyword: '{task['check']}'")

        for model in models:
            answer, elapsed, cost = model["func"](task["prompt"])
            passed = task["check"].lower() in answer.lower()
            status = "PASS" if passed else "FAIL"

            all_results[model["name"]]["pass" if passed else "fail"] += 1
            all_results[model["name"]]["total_time"] += elapsed
            all_results[model["name"]]["total_cost"] += cost
            all_results[model["name"]]["details"].append({
                "category": category,
                "status": status,
                "time": elapsed
            })

            icon = "✅" if passed else "❌"
            print(f"    {icon} {model['name']:<15} [{elapsed}s | ${cost}]: {answer[:80]}...")

# ============================================================
# FINAL SCORECARD
# ============================================================

total_tasks = sum(len(tg["tasks"]) for tg in tests)

print(f"\n{'='*70}")
print("  FINAL SCORECARD")
print(f"{'='*70}")
print(f"\n  {'Model':<18} {'Pass Rate':<12} {'Avg Time':<12} {'Total Cost':<12} {'Type'}")
print(f"  {'-'*60}")

for model in models:
    name = model["name"]
    r = all_results[name]
    total = r["pass"] + r["fail"]
    rate = f"{r['pass']}/{total} ({round(r['pass']/total*100)}%)"
    avg_time = f"{round(r['total_time']/total, 1)}s"
    cost = f"${r['total_cost']:.4f}"
    print(f"  {name:<18} {rate:<12} {avg_time:<12} {cost:<12} {model['type']}")

# Category breakdown
print(f"\n  {'Category':<20}", end="")
for model in models:
    print(f" {model['name']:<15}", end="")
print()
print(f"  {'-'*65}")

for test_group in tests:
    cat = test_group["category"]
    print(f"  {cat:<20}", end="")
    for model in models:
        name = model["name"]
        cat_results = [d for d in all_results[name]["details"] if d["category"] == cat]
        passed = sum(1 for d in cat_results if d["status"] == "PASS")
        total = len(cat_results)
        print(f" {passed}/{total:<14}", end="")
    print()

print(f"\n  RECOMMENDATION:")
print(f"  - Use LOCAL models for: classification, simple Q&A, internal tools, sensitive data")
print(f"  - Use CLOUD (Claude) for: SQL generation, complex reasoning, customer-facing apps")
print(f"  - Cost savings: {total_tasks} queries cost ${sum(r['total_cost'] for r in all_results.values()):.4f} on Claude vs $0.00 locally")