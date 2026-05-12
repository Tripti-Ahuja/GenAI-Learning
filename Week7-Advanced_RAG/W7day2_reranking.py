import anthropic
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import numpy as np
from dotenv import load_dotenv

load_dotenv()
claude = anthropic.Anthropic()

print("Loading models...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
# Cross-encoder re-ranker — scores query-document pairs more carefully
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
print("Ready!\n")

# ============================================================
# STEP 1: Documents
# ============================================================

documents = [
    "Q4 2025 revenue was $2.3 million, up 15% year over year",
    "North region generated $920K in Q4 2025, leading all regions",
    "South region had the weakest performance at $310K, down 8% from Q3. Two accounts churned citing Salesforce integration issues.",
    "West region contributed $740K driven by Analytics Suite adoption among mid-market customers",
    "East region generated $61,500. Newest territory launched July 2024 with 12% month-over-month growth.",
    "Enterprise Plan is the top product at $208K total revenue with $40K average deal size and 6-month sales cycles.",
    "Analytics Suite generated $163K across 5 deals. Popular with Tableau and PowerBI users. Salesforce connector planned Q1 2026.",
    "Dashboard Pro leads in order volume with 9 orders but lowest revenue per unit at $12K. 40% of customers upgrade to Analytics Suite within 12 months.",
    "Customer churn rate dropped to 3.2% in Q4, down from 5.1% in Q3. New onboarding process helped.",
    "Amit Patel is the highest spending customer at $90,000 across 3 orders in the north region.",
    "Average order value across all products is $24,125. Enterprise Plan pulls this average up.",
    "Monthly revenue peaked in April 2025 at $64,000 driven by 2 Enterprise Plan renewals.",
    "Management targets 25 new customer acquisitions in 2026 with east region expansion.",
    "Rajesh Kumar placed 3 orders totaling $64,000 since January 2024. Uses Dashboard Pro and Analytics Suite.",
    "New onboarding with dedicated account managers for 90 days reduced churn significantly.",
]

# Pre-compute embeddings and BM25
doc_embeddings = embed_model.encode(documents)
tokenized_docs = [doc.lower().split() for doc in documents]
bm25 = BM25Okapi(tokenized_docs)

# ============================================================
# STEP 2: Hybrid search (from Day 1)
# ============================================================

def hybrid_search(query, top_k=5):
    query_vec = embed_model.encode([query])
    neural_scores = cosine_similarity(query_vec, doc_embeddings)[0]
    bm25_scores = bm25.get_scores(query.lower().split())

    neural_norm = neural_scores / neural_scores.max() if neural_scores.max() > 0 else neural_scores
    bm25_norm = bm25_scores / bm25_scores.max() if bm25_scores.max() > 0 else bm25_scores

    combined = 0.5 * neural_norm + 0.5 * bm25_norm
    top_idx = combined.argsort()[-top_k:][::-1]

    results = []
    for idx in top_idx:
        results.append({
            "text": documents[idx],
            "hybrid_score": round(float(combined[idx]), 3),
            "idx": idx
        })
    return results

# ============================================================
# STEP 3: Re-rank the top results
# ============================================================

def rerank(query, results):
    # Cross-encoder scores each (query, document) pair carefully
    pairs = [[query, r["text"]] for r in results]
    rerank_scores = reranker.predict(pairs)

    # Attach scores and sort
    for i, r in enumerate(results):
        r["rerank_score"] = round(float(rerank_scores[i]), 3)

    reranked = sorted(results, key=lambda x: x["rerank_score"], reverse=True)
    return reranked

# ============================================================
# STEP 4: Compare before and after re-ranking
# ============================================================

queries = [
    "What product should new customers start with?",
    "Why is the south region losing customers?",
    "What is our most expensive product and its deal size?",
    "Which customer uses multiple products?",
    "How did we reduce customer churn?",
]

print("=" * 70)
print("  HYBRID SEARCH vs HYBRID + RE-RANKING")
print("=" * 70)

for query in queries:
    print(f"\nQuery: \"{query}\"")
    print("-" * 65)

    # Get top 5 from hybrid search
    hybrid_results = hybrid_search(query, top_k=5)

    # Re-rank those 5
    reranked_results = rerank(query, hybrid_results)

    print(f"\n  BEFORE re-ranking (hybrid only):")
    for i, r in enumerate(hybrid_results[:3], 1):
        print(f"    {i}. [{r['hybrid_score']}] {r['text'][:65]}")

    print(f"\n  AFTER re-ranking:")
    for i, r in enumerate(reranked_results[:3], 1):
        print(f"    {i}. [{r['rerank_score']}] {r['text'][:65]}")

    # Did re-ranking change the top result?
    if hybrid_results[0]["idx"] != reranked_results[0]["idx"]:
        print(f"\n    ** Re-ranking changed the top result! **")
    else:
        print(f"\n    Top result stayed the same.")