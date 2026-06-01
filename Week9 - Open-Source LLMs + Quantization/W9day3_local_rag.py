import requests
import time
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import numpy as np

# Optional: Claude for comparison
import anthropic
from dotenv import load_dotenv
load_dotenv()
claude = anthropic.Anthropic()

print("Loading embedding model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Ready!\n")

# ============================================================
# DOCUMENTS
# ============================================================

documents = [
    {"text": "Q4 2025 revenue was $2.3 million, up 15% year over year. Strongest quarter ever.", "source": "Q4 Report"},
    {"text": "North region led with $920K in Q4. 15% growth from Q3 driven by new sales playbook.", "source": "Q4 Report"},
    {"text": "South region had weakest performance at $310K, down 8%. Two accounts churned citing Salesforce issues.", "source": "Q4 Report"},
    {"text": "West region contributed $740K driven by Analytics Suite adoption among mid-market customers.", "source": "Q4 Report"},
    {"text": "Enterprise Plan is top product at $208K revenue. Average deal size $40K.", "source": "Q4 Report"},
    {"text": "Dashboard Pro leads with 9 orders but lowest revenue at $12K per unit. 40% upgrade to Analytics Suite.", "source": "Q4 Report"},
    {"text": "Customer churn dropped to 3.2% in Q4 from 5.1% in Q3. New onboarding process helped.", "source": "Q4 Report"},
    {"text": "Amit Patel is highest spending customer at $90,000 across 3 orders.", "source": "Q4 Report"},
    {"text": "Management targets 25 new customer acquisitions in 2026.", "source": "Q4 Report"},
    {"text": "East region generated $61,500. Newest territory launched July 2024 with 12% monthly growth.", "source": "Q4 Report"},
]

# Hybrid search setup
doc_texts = [d["text"] for d in documents]
doc_embeddings = embed_model.encode(doc_texts)
tokenized = [t.lower().split() for t in doc_texts]
bm25 = BM25Okapi(tokenized)

# ============================================================
# HYBRID SEARCH (same as Week 7)
# ============================================================

def hybrid_search(query, top_k=3):
    query_vec = embed_model.encode([query])
    neural_scores = cosine_similarity(query_vec, doc_embeddings)[0]
    bm25_scores = bm25.get_scores(query.lower().split())

    n_norm = neural_scores / neural_scores.max() if neural_scores.max() > 0 else neural_scores
    b_norm = bm25_scores / bm25_scores.max() if bm25_scores.max() > 0 else bm25_scores

    combined = 0.5 * n_norm + 0.5 * b_norm
    top_idx = combined.argsort()[-top_k:][::-1]

    results = []
    for idx in top_idx:
        results.append({
            "text": documents[idx]["text"],
            "source": documents[idx]["source"],
            "score": round(float(combined[idx]), 3)
        })
    return results

# ============================================================
# TWO GENERATION BACKENDS
# ============================================================

def generate_local(query, context, model="llama3.2:3b"):
    prompt = f"""Answer based ONLY on the provided context. Be concise — 2-3 sentences. Cite sources.
If the context doesn't have the answer, say "Not found in the documents."

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""

    start = time.time()
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=120
    )
    elapsed = round(time.time() - start, 1)
    return response.json()["response"], elapsed, 0.0

def generate_claude(query, context):
    prompt = f"""Answer based ONLY on the provided context. Be concise — 2-3 sentences. Cite sources.
If the context doesn't have the answer, say "Not found in the documents."

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""

    start = time.time()
    response = claude.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    elapsed = round(time.time() - start, 1)
    tokens = response.usage.input_tokens + response.usage.output_tokens
    cost = round((response.usage.input_tokens * 1.0 + response.usage.output_tokens * 5.0) / 1_000_000, 5)
    return response.content[0].text, elapsed, cost

# ============================================================
# RAG PIPELINE — choose your backend
# ============================================================

def ask(query, backend="local", model="llama3.2:3b"):
    results = hybrid_search(query, top_k=3)
    context = "\n\n".join([f"[{r['source']}] {r['text']}" for r in results])

    if backend == "local":
        answer, elapsed, cost = generate_local(query, context, model)
    else:
        answer, elapsed, cost = generate_claude(query, context)

    return {
        "answer": answer.strip(),
        "time": elapsed,
        "cost": cost,
        "backend": backend,
        "model": model if backend == "local" else "claude-haiku",
        "sources": results
    }

# ============================================================
# COMPARISON TEST
# ============================================================

queries = [
    "What was our Q4 revenue?",
    "Which region is struggling and why?",
    "Who is our best customer?",
    "What are our 2026 plans?",
    "What is our marketing budget?",
]

print("=" * 65)
print("  LOCAL RAG vs CLOUD RAG — Same retrieval, different generation")
print("=" * 65)

total_local_time = 0
total_claude_time = 0
total_cost = 0

for query in queries:
    print(f"\nQ: {query}")
    print("-" * 60)

    # Local RAG
    local = ask(query, backend="local", model="llama3.2:3b")
    total_local_time += local["time"]

    # Claude RAG
    cloud = ask(query, backend="claude")
    total_claude_time += cloud["time"]
    total_cost += cloud["cost"]

    print(f"\n  🖥️ Local [{local['time']}s | $0.00]:")
    print(f"     {local['answer'][:150]}")
    print(f"\n  ☁️ Claude [{cloud['time']}s | ${cloud['cost']}]:")
    print(f"     {cloud['answer'][:150]}")

# ============================================================
# SUMMARY
# ============================================================

print(f"\n{'='*65}")
print("  SUMMARY")
print(f"{'='*65}")
print(f"  Local (Llama 3.2 3B):")
print(f"    Total time: {total_local_time}s | Cost: $0.00")
print(f"    Avg per query: {round(total_local_time/len(queries), 1)}s")
print(f"\n  Cloud (Claude Haiku):")
print(f"    Total time: {total_claude_time}s | Cost: ${total_cost:.4f}")
print(f"    Avg per query: {round(total_claude_time/len(queries), 1)}s")
print(f"\n  When to use which:")
print(f"    Local → Sensitive data, offline, zero budget, internal tools")
print(f"    Cloud → Customer-facing, quality-critical, need speed")