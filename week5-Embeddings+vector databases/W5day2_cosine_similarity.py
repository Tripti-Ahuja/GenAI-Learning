import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# STEP 1: Documents grouped by topic (you know which should match)
# ============================================================

documents = [
    # Revenue topic (index 0-2)
    "Q4 revenue was $2.3 million, up 15% year over year",
    "Total sales reached $2.3M in the fourth quarter",
    "The company generated $482K in total order value",

    # Region topic (index 3-5)
    "North region leads all regions with $920K revenue",
    "South region had the weakest performance at $310K",
    "The west region contributed $740K in Q4",

    # Customer topic (index 6-8)
    "Amit Patel is the top customer spending $90,000",
    "Our highest value client has spent ninety thousand dollars",
    "Rajesh Kumar placed 3 orders totaling $64,000",

    # Product topic (index 9-11)
    "Enterprise Plan is the best selling product",
    "Dashboard Pro has the most orders but lowest revenue",
    "Analytics Suite generated $163K in total revenue",
]

labels = [
    "Revenue-Q4", "Revenue-Sales", "Revenue-Total",
    "Region-North", "Region-South", "Region-West",
    "Customer-Amit", "Customer-TopClient", "Customer-Rajesh",
    "Product-Enterprise", "Product-Dashboard", "Product-Analytics"
]

# ============================================================
# STEP 2: Create embeddings and compute ALL pairwise similarities
# ============================================================

vectorizer = TfidfVectorizer(stop_words="english")
vectors = vectorizer.fit_transform(documents)

# This creates a 12x12 matrix — every document compared to every other
similarity_matrix = cosine_similarity(vectors)

print("=" * 60)
print("  COSINE SIMILARITY MATRIX")
print("=" * 60)

# Print header
print(f"\n{'':>20}", end="")
for label in labels:
    print(f"{label[:8]:>10}", end="")
print()

# Print matrix
for i, label in enumerate(labels):
    print(f"{label:>20}", end="")
    for j in range(len(labels)):
        score = similarity_matrix[i][j]
        # Highlight high similarities (same topic should be high)
        if i == j:
            marker = "  ----  "
        elif score > 0.3:
            marker = f"  {score:.2f}* "
        elif score > 0.1:
            marker = f"  {score:.2f}  "
        else:
            marker = f"  {score:.2f}  "
        print(f"{marker:>10}", end="")
    print()

# ============================================================
# STEP 3: Find the most similar pair and most different pair
# ============================================================

print(f"\n{'='*60}")
print("  ANALYSIS")
print(f"{'='*60}")

best_score = 0
best_pair = ("", "")
worst_score = 1
worst_pair = ("", "")

for i in range(len(documents)):
    for j in range(i + 1, len(documents)):
        score = similarity_matrix[i][j]
        if score > best_score:
            best_score = score
            best_pair = (labels[i], labels[j])
        if score < worst_score:
            worst_score = score
            worst_pair = (labels[i], labels[j])

print(f"\n  Most similar pair:")
print(f"    {best_pair[0]} <-> {best_pair[1]} = {best_score:.3f}")

print(f"\n  Most different pair:")
print(f"    {worst_pair[0]} <-> {worst_pair[1]} = {worst_score:.3f}")

# ============================================================
# STEP 4: Check if same-topic docs score higher than cross-topic
# ============================================================

print(f"\n{'='*60}")
print("  WITHIN-TOPIC vs CROSS-TOPIC SIMILARITY")
print(f"{'='*60}")

topics = {
    "Revenue": [0, 1, 2],
    "Region": [3, 4, 5],
    "Customer": [6, 7, 8],
    "Product": [9, 10, 11],
}

for topic_name, indices in topics.items():
    # Average similarity within this topic
    within_scores = []
    for i in indices:
        for j in indices:
            if i != j:
                within_scores.append(similarity_matrix[i][j])
    avg_within = np.mean(within_scores) if within_scores else 0

    # Average similarity with OTHER topics
    cross_scores = []
    other_indices = [k for k in range(12) if k not in indices]
    for i in indices:
        for j in other_indices:
            cross_scores.append(similarity_matrix[i][j])
    avg_cross = np.mean(cross_scores) if cross_scores else 0

    gap = avg_within - avg_cross
    print(f"\n  {topic_name}:")
    print(f"    Within-topic avg: {avg_within:.3f}")
    print(f"    Cross-topic avg:  {avg_cross:.3f}")
    print(f"    Gap: {gap:+.3f} {'(good separation)' if gap > 0.05 else '(weak separation)'}")