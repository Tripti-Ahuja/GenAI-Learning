from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

print("Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Ready!\n")

# ============================================================
# STEP 1: Documents with metadata (like columns in a database)
# ============================================================

documents = [
    {"text": "Q4 2025 revenue was $2.3 million, up 15% YoY", "category": "finance", "region": "all", "quarter": "Q4", "year": 2025},
    {"text": "North region generated $920K in Q4", "category": "finance", "region": "north", "quarter": "Q4", "year": 2025},
    {"text": "South region had weakest performance at $310K", "category": "finance", "region": "south", "quarter": "Q4", "year": 2025},
    {"text": "Customer churn dropped to 3.2% in Q4 2025", "category": "customers", "region": "all", "quarter": "Q4", "year": 2025},
    {"text": "Amit Patel is the top customer at $90K spending", "category": "customers", "region": "north", "quarter": "all", "year": 2025},
    {"text": "10 new customers signed up in 2024", "category": "customers", "region": "all", "quarter": "all", "year": 2024},
    {"text": "Enterprise Plan is top product at $208K revenue", "category": "products", "region": "all", "quarter": "all", "year": 2025},
    {"text": "Dashboard Pro has most orders but lowest revenue", "category": "products", "region": "all", "quarter": "all", "year": 2025},
    {"text": "Q3 2025 north revenue was $800K", "category": "finance", "region": "north", "quarter": "Q3", "year": 2025},
    {"text": "Q3 2025 south revenue was $350K", "category": "finance", "region": "south", "quarter": "Q3", "year": 2025},
    {"text": "2024 total revenue was $1.8 million", "category": "finance", "region": "all", "quarter": "all", "year": 2024},
    {"text": "West region grew 40% from Q3 to Q4", "category": "finance", "region": "west", "quarter": "Q4", "year": 2025},
]

# Embed all documents once
all_embeddings = model.encode([doc["text"] for doc in documents])

# ============================================================
# STEP 2: Search with metadata filters
# ============================================================

def search(query, n_results=3, category=None, region=None, year=None, quarter=None):
    print(f"\nQuery: \"{query}\"")
    filters = []
    if category: filters.append(f"category={category}")
    if region: filters.append(f"region={region}")
    if year: filters.append(f"year={year}")
    if quarter: filters.append(f"quarter={quarter}")
    print(f"Filters: {', '.join(filters) if filters else 'None'}")

    # Step 1: Filter documents by metadata
    filtered_indices = []
    for i, doc in enumerate(documents):
        if category and doc["category"] != category:
            continue
        if region and doc["region"] != region:
            continue
        if year and doc["year"] != year:
            continue
        if quarter and doc["quarter"] != quarter:
            continue
        filtered_indices.append(i)

    print(f"Documents after filter: {len(filtered_indices)} of {len(documents)}")

    if not filtered_indices:
        print("  No documents match these filters.")
        return

    # Step 2: Compare query against ONLY filtered documents
    query_embedding = model.encode([query])
    filtered_embeddings = np.array([all_embeddings[i] for i in filtered_indices])
    similarities = cosine_similarity(query_embedding, filtered_embeddings)[0]

    # Step 3: Rank by similarity
    top_indices = similarities.argsort()[-n_results:][::-1]

    print(f"Top {min(n_results, len(top_indices))} results:")
    for rank, idx in enumerate(top_indices, 1):
        real_idx = filtered_indices[idx]
        doc = documents[real_idx]
        score = round(similarities[idx], 3)
        print(f"  {rank}. [{score}] [{doc['category']}/{doc['region']}/{doc['quarter']}] {doc['text'][:60]}")

# ============================================================
# STEP 3: Test different filter combinations
# ============================================================

print("=" * 60)
print("  METADATA FILTERING + SEMANTIC SEARCH")
print("=" * 60)

# No filter — searches everything
search("What was the revenue?")

# Filter by category — only finance docs
search("What was the revenue?", category="finance")

# Filter by region — only north
search("How are we doing?", region="north")

# Filter by year — only 2024 data
search("What happened with customers?", year=2024)

# Filter by quarter — only Q3
search("Revenue numbers", quarter="Q3")

# Combined filters — finance + north + Q4
search("Performance update", category="finance", region="north", quarter="Q4")

# Filter that returns no results
search("Revenue data", region="east", quarter="Q3")