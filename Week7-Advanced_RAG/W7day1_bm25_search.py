from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import numpy as np

print("Loading model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Ready!\n")

# ============================================================
# STEP 1: Documents
# ============================================================

documents = [
    "Q4 2025 revenue was $2.3 million, up 15% year over year",
    "North region generated $920K in Q4 2025, leading all regions",
    "South region had the weakest performance at $310K, down 8% from Q3. Two accounts churned citing Salesforce integration issues.",
    "West region contributed $740K driven by Analytics Suite adoption among mid-market customers",
    "East region generated $61,500 in total revenue. Newest territory launched July 2024.",
    "Enterprise Plan is the top product at $208K total revenue with $40K average deal size",
    "Analytics Suite generated $163K across 5 deals. Popular with Tableau and PowerBI users.",
    "Dashboard Pro leads in order volume with 9 orders but lowest revenue per unit at $12K. 40% of customers upgrade to Analytics Suite within 12 months.",
    "Customer churn rate dropped to 3.2% in Q4, down from 5.1% in Q3",
    "Amit Patel is the highest spending customer at $90,000 across 3 orders",
    "Average order value across all products is $24,125",
    "Monthly revenue peaked in April 2025 at $64,000",
    "Management targets 25 new customer acquisitions in 2026",
    "Rajesh Kumar placed 3 orders totaling $64,000 since January 2024",
    "New onboarding process with dedicated account managers for 90 days reduced churn",
]

# ============================================================
# STEP 2: Set up both search methods
# ============================================================

# Neural embeddings
doc_embeddings = embed_model.encode(documents)

# BM25 — needs tokenized documents (split into words)
tokenized_docs = [doc.lower().split() for doc in documents]
bm25 = BM25Okapi(tokenized_docs)

# ============================================================
# STEP 3: Three search methods
# ============================================================

def neural_search(query, top_k=3):
    query_vec = embed_model.encode([query])
    scores = cosine_similarity(query_vec, doc_embeddings)[0]
    top_idx = scores.argsort()[-top_k:][::-1]
    return [(idx, round(float(scores[idx]), 3)) for idx in top_idx]

def bm25_search(query, top_k=3):
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_idx = scores.argsort()[-top_k:][::-1]
    return [(idx, round(float(scores[idx]), 3)) for idx in top_idx]

def hybrid_search(query, top_k=3, neural_weight=0.5):
    # Get scores from both methods
    query_vec = embed_model.encode([query])
    neural_scores = cosine_similarity(query_vec, doc_embeddings)[0]

    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)

    # Normalize both to 0-1 range
    if neural_scores.max() > 0:
        neural_norm = neural_scores / neural_scores.max()
    else:
        neural_norm = neural_scores

    if bm25_scores.max() > 0:
        bm25_norm = bm25_scores / bm25_scores.max()
    else:
        bm25_norm = bm25_scores

    # Combine: weighted average
    combined = (neural_weight * neural_norm) + ((1 - neural_weight) * bm25_norm)
    top_idx = combined.argsort()[-top_k:][::-1]
    return [(idx, round(float(combined[idx]), 3)) for idx in top_idx]

# ============================================================
# STEP 4: Compare all 3 methods
# ============================================================

queries = [
    "How much money did we make last quarter?",
    "Tell me about Dashboard Pro",
    "Why did south region underperform?",
    "What is the average deal size for Enterprise Plan?",
    "Salesforce integration problems",
]

print("=" * 70)
print("  BM25 vs NEURAL vs HYBRID SEARCH")
print("=" * 70)

for query in queries:
    print(f"\nQuery: \"{query}\"")
    print("-" * 65)

    neural_results = neural_search(query)
    bm25_results = bm25_search(query)
    hybrid_results = hybrid_search(query)

    print(f"\n  {'BM25 (keyword)':<25} {'Neural (meaning)':<25} {'Hybrid (combined)'}")
    print(f"  {'-'*23}   {'-'*23}   {'-'*23}")

    for i in range(3):
        bm25_idx, bm25_score = bm25_results[i]
        neural_idx, neural_score = neural_results[i]
        hybrid_idx, hybrid_score = hybrid_results[i]

        b_text = documents[bm25_idx][:22]
        n_text = documents[neural_idx][:22]
        h_text = documents[hybrid_idx][:22]

        print(f"  [{bm25_score}] {b_text:<22} [{neural_score}] {n_text:<22} [{hybrid_score}] {h_text}")