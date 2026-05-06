from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

print("Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Ready!\n")

# ============================================================
# STEP 1: A long document (simulating a real report)
# ============================================================

long_document = """
QUARTERLY BUSINESS REPORT - Q4 2025

EXECUTIVE SUMMARY
The company delivered record-breaking results in Q4 2025. Total revenue reached $2.3 million, representing a 15% increase year over year. This growth was primarily driven by strong Enterprise Plan adoption in the north region and improved customer retention across all segments. The board has approved an expansion plan for 2026 targeting 25 new customer acquisitions.

REVENUE BY REGION
The north region led all regions with $920K in Q4 revenue. This represents a 15% growth from Q3, largely attributed to the new sales playbook introduced in September. Three large Enterprise Plan deals closed in October and November, contributing $140K combined. The sales team in the north expanded from 3 to 5 representatives during the quarter.

The west region contributed $740K, driven by strong Analytics Suite adoption among mid-market customers. The region saw particular success with companies transitioning from legacy BI tools like Tableau and PowerBI. Average deal size in the west was $19,473.

The south region had the weakest performance at $310K, down 8% from Q3. Two key accounts churned during the quarter, citing integration issues with their existing Salesforce workflows. The sales team is implementing a new onboarding process to address these concerns in Q1 2026.

The east region, our newest territory launched in July 2024, generated $61,500. While this is the smallest contribution, the region showed promising month-over-month growth of 12%. Three new pilot programs are scheduled for Q1 2026.

PRODUCT PERFORMANCE
Enterprise Plan remains our flagship offering at $208K total revenue. The product commands an average deal size of $40K with typical sales cycles of 6 months. Customer satisfaction scores for Enterprise Plan users averaged 4.5 out of 5.

Analytics Suite generated $163K across 5 deals during the quarter. The product is particularly popular with data teams who already use Tableau or PowerBI, as it offers seamless integration with existing workflows. We plan to add Salesforce connector in Q1 2026.

Dashboard Pro continues to lead in order volume with 9 total orders, but generates the lowest revenue per unit at approximately $12K average. The product serves as an effective entry point, with 40% of Dashboard Pro customers upgrading to Analytics Suite within 12 months.

CUSTOMER HIGHLIGHTS
Amit Patel remains our highest-value customer with $90,000 in total spending across 3 orders. Based in the north region, Amit has been a customer since Q1 2024 and has consistently expanded usage across his organization.

Rajesh Kumar is our most loyal customer, having placed 3 orders totaling $64,000 since January 2024. His team uses both Dashboard Pro and Analytics Suite for their daily reporting needs.

Customer churn rate dropped to 3.2% in Q4, down from 5.1% in Q3. The improvement is attributed to the new onboarding process launched in August, which includes dedicated account managers for the first 90 days.

OUTLOOK FOR 2026
Management targets 25 new customer acquisitions in 2026, up from 10 in 2024. Key initiatives include expanding the east region sales team, launching the Salesforce connector for Analytics Suite, and introducing a new mid-tier pricing plan to bridge the gap between Dashboard Pro and Enterprise Plan.
"""

# ============================================================
# STEP 2: Three different chunking strategies
# ============================================================

def chunk_by_characters(text, chunk_size=300, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
    return chunks

def chunk_by_sentences(text, sentences_per_chunk=3, overlap=1):
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s for s in sentences if len(s) > 10]
    chunks = []
    start = 0
    while start < len(sentences):
        end = min(start + sentences_per_chunk, len(sentences))
        chunk = " ".join(sentences[start:end])
        if chunk.strip():
            chunks.append(chunk.strip())
        step = sentences_per_chunk - overlap
        if step < 1:
            step = 1
        start += step
    return chunks

def chunk_by_sections(text):
    sections = text.strip().split("\n\n")
    chunks = [s.strip() for s in sections if len(s.strip()) > 30]
    return chunks

# ============================================================
# STEP 3: Compare chunking strategies
# ============================================================

char_chunks = chunk_by_characters(long_document, chunk_size=300, overlap=50)
sent_chunks = chunk_by_sentences(long_document, sentences_per_chunk=3, overlap=1)
section_chunks = chunk_by_sections(long_document)

print("=" * 60)
print("  CHUNKING STRATEGIES COMPARISON")
print("=" * 60)

strategies = [
    ("Character-based (300 chars, 50 overlap)", char_chunks),
    ("Sentence-based (3 sentences, 1 overlap)", sent_chunks),
    ("Section-based (split by paragraphs)", section_chunks),
]

for name, chunks in strategies:
    sizes = [len(c) for c in chunks]
    print(f"\n  {name}:")
    print(f"    Total chunks: {len(chunks)}")
    print(f"    Avg chunk size: {int(np.mean(sizes))} chars")
    print(f"    Smallest: {min(sizes)} chars | Largest: {max(sizes)} chars")
    print(f"    First chunk: \"{chunks[0][:80]}...\"")

# ============================================================
# STEP 4: Search quality comparison across strategies
# ============================================================

queries = [
    "What was Q4 revenue?",
    "Why did south region perform poorly?",
    "What product do customers upgrade to?",
]

print(f"\n{'='*60}")
print("  SEARCH QUALITY: Which chunking finds the best answer?")
print(f"{'='*60}")

for query in queries:
    print(f"\n  Query: \"{query}\"")
    print(f"  {'-'*50}")

    for name, chunks in strategies:
        embeddings = model.encode(chunks)
        query_vec = model.encode([query])
        scores = cosine_similarity(query_vec, embeddings)[0]
        best_idx = scores.argmax()
        best_score = scores[best_idx]

        short_name = name.split("(")[0].strip()
        print(f"\n    {short_name}:")
        print(f"    Score: {best_score:.3f}")
        print(f"    Match: \"{chunks[best_idx][:100]}...\"")