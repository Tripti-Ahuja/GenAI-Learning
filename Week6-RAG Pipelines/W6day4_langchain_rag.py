import anthropic
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dotenv import load_dotenv

load_dotenv()
claude = anthropic.Anthropic()

print("Loading model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Ready!\n")

# ============================================================
# STEP 1: The document (same report from Day 3)
# ============================================================

report = """
QUARTERLY BUSINESS REPORT - Q4 2025

EXECUTIVE SUMMARY
The company delivered record-breaking results in Q4 2025. Total revenue reached $2.3 million, representing a 15% increase year over year. This growth was primarily driven by strong Enterprise Plan adoption in the north region and improved customer retention.

REVENUE BY REGION
The north region led all regions with $920K in Q4 revenue. This represents a 15% growth from Q3, attributed to the new sales playbook introduced in September.

The west region contributed $740K, driven by strong Analytics Suite adoption among mid-market customers transitioning from Tableau and PowerBI.

The south region had the weakest performance at $310K, down 8% from Q3. Two key accounts churned citing integration issues with Salesforce workflows.

The east region generated $61,500. This is the newest territory launched in July 2024, showing 12% month-over-month growth.

PRODUCT PERFORMANCE
Enterprise Plan remains our flagship at $208K total revenue with average deal size of $40K and 6-month sales cycles.

Analytics Suite generated $163K across 5 deals. Popular with data teams using Tableau or PowerBI. Salesforce connector planned for Q1 2026.

Dashboard Pro leads in order volume with 9 orders but lowest revenue per unit at $12K. 40% of customers upgrade to Analytics Suite within 12 months.

CUSTOMER METRICS
Customer churn rate dropped to 3.2% in Q4, down from 5.1% in Q3. New onboarding process with dedicated account managers for 90 days has been effective.

Amit Patel is our highest-value customer at $90,000 across 3 orders. Rajesh Kumar is most loyal with 3 orders totaling $64,000 since January 2024.

OUTLOOK FOR 2026
Management targets 25 new customer acquisitions. Key initiatives: expanding east region sales team, launching Salesforce connector, introducing mid-tier pricing plan.
"""

# ============================================================
# STEP 2: LangChain-style text splitting
# ============================================================

from langchain_text_splitters import RecursiveCharacterTextSplitter

# This is LangChain's smart splitter — it tries to split by paragraphs first,
# then sentences, then words, then characters as a last resort
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""]
)

chunks = splitter.create_documents([report])

print(f"LangChain split the report into {len(chunks)} chunks:")
for i, chunk in enumerate(chunks):
    print(f"  Chunk {i+1} ({len(chunk.page_content)} chars): \"{chunk.page_content[:60]}...\"")

# ============================================================
# STEP 3: Build RAG — your code vs LangChain comparison
# ============================================================

# Embed all chunks
chunk_texts = [chunk.page_content for chunk in chunks]
chunk_embeddings = embed_model.encode(chunk_texts)

def retrieve(query, top_k=3):
    query_embedding = embed_model.encode([query])
    scores = cosine_similarity(query_embedding, chunk_embeddings)[0]
    top_indices = scores.argsort()[-top_k:][::-1]
    results = []
    for idx in top_indices:
        results.append({
            "text": chunk_texts[idx],
            "score": round(float(scores[idx]), 3),
            "chunk_id": idx + 1
        })
    return results

def ask(query):
    print(f"\nQ: {query}")
    print("-" * 50)

    retrieved = retrieve(query, top_k=3)
    print("Retrieved:")
    for doc in retrieved:
        print(f"  [{doc['score']}] Chunk {doc['chunk_id']}: \"{doc['text'][:55]}...\"")

    context = "\n\n".join([f"[Chunk {d['chunk_id']}] {d['text']}" for d in retrieved])

    prompt = f"""Answer based ONLY on the provided context. Cite sources as [Chunk N].
If the answer isn't in the context, say "Not found in the report."
Be concise — 2-3 sentences max.

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""

    response = claude.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    print(f"\nA: {response.content[0].text}")

# ============================================================
# STEP 4: Compare chunk quality — your Day 3 sections vs LangChain
# ============================================================

print("=" * 60)
print("  LANGCHAIN RecursiveCharacterTextSplitter")
print("=" * 60)

# Show what LangChain's separators do
print(f"\n  How LangChain splits (priority order):")
print(f"  1st try: Split by '\\n\\n' (paragraph breaks)")
print(f"  2nd try: Split by '\\n' (line breaks)")
print(f"  3rd try: Split by '. ' (sentences)")
print(f"  4th try: Split by ' ' (words)")
print(f"  Last resort: Split by '' (characters)")
print(f"\n  Chunk size: 300 chars | Overlap: 50 chars")
print(f"  Total chunks: {len(chunks)}")

# ============================================================
# STEP 5: Test queries
# ============================================================

print(f"\n{'='*60}")
print("  RAG WITH LANGCHAIN CHUNKING")
print(f"{'='*60}")

ask("What was Q4 revenue?")
ask("Why did the south region struggle?")
ask("What product do most customers start with?")
ask("What are the 2026 plans?")
ask("What is our employee count?")