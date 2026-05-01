from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

print("Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Ready!\n")

# ============================================================
# DOCUMENT STORE — 20 documents with metadata
# ============================================================

documents = [
    {"text": "Q4 2025 revenue was $2.3 million, up 15% year over year", "category": "finance", "region": "all", "year": 2025},
    {"text": "North region generated $920K in Q4 2025, leading all regions", "category": "finance", "region": "north", "year": 2025},
    {"text": "South region had the weakest performance at $310K revenue", "category": "finance", "region": "south", "year": 2025},
    {"text": "West region contributed $740K in Q4 revenue", "category": "finance", "region": "west", "year": 2025},
    {"text": "East region generated $61,500 in total revenue", "category": "finance", "region": "east", "year": 2025},
    {"text": "2024 total revenue was $1.8 million across all regions", "category": "finance", "region": "all", "year": 2024},
    {"text": "Q3 to Q4 revenue growth for north region was 15%", "category": "finance", "region": "north", "year": 2025},
    {"text": "Customer churn rate dropped to 3.2% in Q4 2025", "category": "customers", "region": "all", "year": 2025},
    {"text": "Amit Patel is the highest spending customer at $90,000", "category": "customers", "region": "north", "year": 2025},
    {"text": "Rajesh Kumar placed 3 orders totaling $64,000", "category": "customers", "region": "north", "year": 2025},
    {"text": "Priya Sharma from south region spent $40,500", "category": "customers", "region": "south", "year": 2025},
    {"text": "10 new customers signed up across all regions in 2024", "category": "customers", "region": "all", "year": 2024},
    {"text": "Enterprise Plan is the top selling product at $208K revenue", "category": "products", "region": "all", "year": 2025},
    {"text": "Dashboard Pro has the most orders but lowest revenue per unit", "category": "products", "region": "all", "year": 2025},
    {"text": "Analytics Suite generated $163K in total revenue", "category": "products", "region": "all", "year": 2025},
    {"text": "Average order value across all products is $24,125", "category": "products", "region": "all", "year": 2025},
    {"text": "Monthly revenue peaked in April 2025 at $64,000", "category": "finance", "region": "all", "year": 2025},
    {"text": "Customer satisfaction score improved to 4.2 out of 5", "category": "customers", "region": "all", "year": 2025},
    {"text": "Product team launched 3 new features in Q4 2025", "category": "products", "region": "all", "year": 2025},
    {"text": "Sales team expanded to 4 regions in 2024", "category": "finance", "region": "all", "year": 2024},
]

# Pre-compute all embeddings
all_texts = [doc["text"] for doc in documents]
neural_embeddings = model.encode(all_texts)

tfidf_vectorizer = TfidfVectorizer(stop_words="english")
tfidf_vectors = tfidf_vectorizer.fit_transform(all_texts)

print(f"Loaded {len(documents)} documents")
print(f"Neural embedding size: {neural_embeddings.shape[1]} dimensions")
print(f"TF-IDF vocabulary size: {tfidf_vectors.shape[1]} words")

# ============================================================
# SEARCH FUNCTION — with filtering and method choice
# ============================================================

def search(query, n_results=3, method="neural", category=None, region=None, year=None):
    # Step 1: Filter
    filtered = []
    for i, doc in enumerate(documents):
        if category and doc["category"] != category:
            continue
        if region and doc["region"] != region:
            continue
        if year and doc["year"] != year:
            continue
        filtered.append(i)

    if not filtered:
        print(f"\n  No documents match filters.")
        return []

    # Step 2: Search
    if method == "neural":
        query_vec = model.encode([query])
        filtered_vecs = np.array([neural_embeddings[i] for i in filtered])
        scores = cosine_similarity(query_vec, filtered_vecs)[0]
    else:
        query_vec = tfidf_vectorizer.transform([query])
        filtered_vecs = tfidf_vectors[filtered]
        scores = cosine_similarity(query_vec, filtered_vecs)[0]

    # Step 3: Rank
    top = scores.argsort()[-n_results:][::-1]

    results = []
    for rank, idx in enumerate(top, 1):
        real_idx = filtered[idx]
        doc = documents[real_idx]
        score = round(float(scores[idx]), 3)
        results.append({"rank": rank, "score": score, "text": doc["text"], "category": doc["category"], "region": doc["region"]})

    return results

# ============================================================
# INTERACTIVE SEARCH
# ============================================================

print("\n" + "=" * 60)
print("  SEMANTIC SEARCH ENGINE")
print("  Commands:")
print("    Just type a question to search")
print("    'filter category region year' e.g. 'filter finance north 2025'")
print("    'compare' to see TF-IDF vs Neural side by side")
print("    'quit' to exit")
print("=" * 60)

active_filters = {"category": None, "region": None, "year": None}

while True:
    user_input = input("\nSearch: ").strip()

    if not user_input:
        continue

    elif user_input.lower() == "quit":
        print("Goodbye!")
        break

    elif user_input.lower().startswith("filter"):
        parts = user_input.split()
        active_filters["category"] = parts[1] if len(parts) > 1 and parts[1] != "none" else None
        active_filters["region"] = parts[2] if len(parts) > 2 and parts[2] != "none" else None
        active_filters["year"] = int(parts[3]) if len(parts) > 3 and parts[3] != "none" else None
        active = {k: v for k, v in active_filters.items() if v}
        print(f"  Filters set: {active if active else 'None (searching all docs)'}")

    elif user_input.lower() == "compare":
        query = input("  Enter query to compare: ").strip()
        if not query:
            continue

        print(f"\n  Query: \"{query}\"")
        filters = {k: v for k, v in active_filters.items() if v}
        print(f"  Filters: {filters if filters else 'None'}")

        tfidf_results = search(query, n_results=3, method="tfidf", **active_filters)
        neural_results = search(query, n_results=3, method="neural", **active_filters)

        print(f"\n  {'TF-IDF Results:':<40} {'Neural Results:'}")
        print(f"  {'-'*38}  {'-'*38}")

        for i in range(max(len(tfidf_results), len(neural_results))):
            left = f"[{tfidf_results[i]['score']}] {tfidf_results[i]['text'][:30]}" if i < len(tfidf_results) else ""
            right = f"[{neural_results[i]['score']}] {neural_results[i]['text'][:30]}" if i < len(neural_results) else ""
            print(f"  {left:<40} {right}")

    else:
        results = search(user_input, n_results=3, method="neural", **active_filters)
        filters = {k: v for k, v in active_filters.items() if v}
        print(f"\n  Filters: {filters if filters else 'None'}")
        print(f"  Results ({len(results)}):")
        for r in results:
            print(f"    {r['rank']}. [{r['score']}] [{r['category']}/{r['region']}] {r['text'][:65]}")