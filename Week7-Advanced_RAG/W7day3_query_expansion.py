import anthropic
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import numpy as np
import json
from dotenv import load_dotenv

load_dotenv()
claude_client = anthropic.Anthropic()

print("Loading models...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
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
    "Enterprise Plan is the top product at $208K total revenue with $40K average deal size",
    "Analytics Suite generated $163K across 5 deals. Popular with Tableau and PowerBI users.",
    "Dashboard Pro leads in order volume with 9 orders but lowest revenue per unit at $12K. 40% of customers upgrade to Analytics Suite within 12 months.",
    "Customer churn rate dropped to 3.2% in Q4, down from 5.1% in Q3. New onboarding process helped.",
    "Amit Patel is the highest spending customer at $90,000 across 3 orders",
    "Average order value across all products is $24,125",
    "Management targets 25 new customer acquisitions in 2026",
    "Rajesh Kumar uses Dashboard Pro and Analytics Suite. 3 orders totaling $64,000.",
    "New onboarding with dedicated account managers for 90 days reduced churn significantly.",
]

doc_embeddings = embed_model.encode(documents)
tokenized_docs = [doc.lower().split() for doc in documents]
bm25 = BM25Okapi(tokenized_docs)

# ============================================================
# STEP 2: Query expansion — Claude rephrases the question
# ============================================================

def expand_query(original_query):
    prompt = f"""Rephrase this question 3 different ways to help search a business report. 
Each version should use different words but same meaning.
Return ONLY a JSON array of 3 strings, no markdown.

Original: "{original_query}"

Example output: ["rephrased version 1", "rephrased version 2", "rephrased version 3"]"""

    response = claude_client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        temperature=0.3,
        messages=[{"role": "user", "content": prompt}]
    )

    text = response.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        rephrased = json.loads(text)
        return [original_query] + rephrased
    except:
        return [original_query]

# ============================================================
# STEP 3: Search with all expanded queries
# ============================================================

def hybrid_search(query, top_k=3):
    query_vec = embed_model.encode([query])
    neural_scores = cosine_similarity(query_vec, doc_embeddings)[0]
    bm25_scores = bm25.get_scores(query.lower().split())

    neural_norm = neural_scores / neural_scores.max() if neural_scores.max() > 0 else neural_scores
    bm25_norm = bm25_scores / bm25_scores.max() if bm25_scores.max() > 0 else bm25_scores

    combined = 0.5 * neural_norm + 0.5 * bm25_norm
    top_idx = combined.argsort()[-top_k:][::-1]
    return [(idx, round(float(combined[idx]), 3)) for idx in top_idx]

def expanded_search(original_query, top_k=3):
    # Get all query versions
    all_queries = expand_query(original_query)

    # Search with each version and collect all results
    doc_scores = {}
    for query in all_queries:
        results = hybrid_search(query, top_k=5)
        for idx, score in results:
            if idx not in doc_scores:
                doc_scores[idx] = []
            doc_scores[idx].append(score)

    # For each document, take the MAX score across all query versions
    final_scores = {}
    for idx, scores in doc_scores.items():
        final_scores[idx] = max(scores)

    # Sort and return top_k
    sorted_results = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return sorted_results

# ============================================================
# STEP 4: Compare original vs expanded search
# ============================================================

queries = [
    "How's our customer retention?",
    "What product do beginners buy?",
    "Which territory is growing fastest?",
    "Are we losing any big clients?",
]

print("=" * 70)
print("  ORIGINAL QUERY vs EXPANDED QUERY SEARCH")
print("=" * 70)

for query in queries:
    print(f"\nOriginal query: \"{query}\"")
    print("-" * 65)

    # Expand the query
    expanded = expand_query(query)
    print(f"\n  Expanded versions:")
    for i, q in enumerate(expanded):
        label = "Original" if i == 0 else f"Version {i}"
        print(f"    {label}: \"{q}\"")

    # Search with original only
    original_results = hybrid_search(query, top_k=3)

    # Search with all expanded versions
    expanded_results = expanded_search(query, top_k=3)

    print(f"\n  Original search top 3:")
    for idx, score in original_results:
        print(f"    [{score}] {documents[idx][:65]}")

    print(f"\n  Expanded search top 3:")
    for idx, score in expanded_results:
        print(f"    [{score}] {documents[idx][:65]}")

    # Did expansion find different documents?
    orig_set = set(idx for idx, _ in original_results)
    exp_set = set(idx for idx, _ in expanded_results)
    new_docs = exp_set - orig_set
    if new_docs:
        print(f"\n    ** Expansion found {len(new_docs)} new document(s) that original missed! **")
    else:
        print(f"\n    Same documents, potentially reordered.")