import anthropic
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
from dotenv import load_dotenv

load_dotenv()
claude = anthropic.Anthropic()

print("Loading model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Ready!\n")

# ============================================================
# STEP 1: A real document — quarterly business report
# ============================================================

report = """
QUARTERLY BUSINESS REPORT - Q4 2025

EXECUTIVE SUMMARY
The company delivered record-breaking results in Q4 2025. Total revenue reached $2.3 million, representing a 15% increase year over year. This growth was primarily driven by strong Enterprise Plan adoption in the north region and improved customer retention across all segments.

REVENUE BY REGION
The north region led all regions with $920K in Q4 revenue. This represents a 15% growth from Q3, largely attributed to the new sales playbook introduced in September. Three large Enterprise Plan deals closed in October and November.

The west region contributed $740K, driven by strong Analytics Suite adoption among mid-market customers. Companies transitioning from Tableau and PowerBI found our Analytics Suite particularly appealing.

The south region had the weakest performance at $310K, down 8% from Q3. Two key accounts churned during the quarter, citing integration issues with their existing Salesforce workflows.

The east region generated $61,500 in total revenue. This is our newest territory launched in July 2024. The region showed promising month-over-month growth of 12%.

PRODUCT PERFORMANCE
Enterprise Plan remains our flagship at $208K total revenue with an average deal size of $40K. Sales cycles typically run 6 months. Customer satisfaction scores averaged 4.5 out of 5.

Analytics Suite generated $163K across 5 deals. The product is popular with data teams already using Tableau or PowerBI. A Salesforce connector is planned for Q1 2026.

Dashboard Pro leads in order volume with 9 orders but has the lowest revenue per unit at $12K average. However, 40% of Dashboard Pro customers upgrade to Analytics Suite within 12 months.

CUSTOMER METRICS
Customer churn rate dropped to 3.2% in Q4, down from 5.1% in Q3. The new onboarding process with dedicated account managers for 90 days has been effective.

Amit Patel remains our highest-value customer at $90,000 across 3 orders. Rajesh Kumar is our most loyal customer with 3 orders totaling $64,000 since January 2024.

OUTLOOK FOR 2026
Management targets 25 new customer acquisitions in 2026. Key initiatives include expanding the east region sales team, launching the Salesforce connector, and introducing a mid-tier pricing plan.
"""

# ============================================================
# STEP 2: Chunk by sections with metadata
# ============================================================

def chunk_document(text):
    sections = text.strip().split("\n\n")
    chunks = []
    current_heading = "General"

    for section in sections:
        section = section.strip()
        if not section or len(section) < 20:
            continue

        # Check if this is a heading (ALL CAPS or short line)
        lines = section.split("\n")
        if lines[0].isupper() and len(lines[0]) < 60:
            current_heading = lines[0].strip()
            # If there's content after the heading, use it
            content = "\n".join(lines[1:]).strip()
            if content and len(content) > 20:
                chunks.append({"text": content, "section": current_heading, "chunk_id": len(chunks) + 1})
        else:
            chunks.append({"text": section, "section": current_heading, "chunk_id": len(chunks) + 1})

    return chunks

chunks = chunk_document(report)

print(f"Document chunked into {len(chunks)} pieces:")
for c in chunks:
    print(f"  Chunk {c['chunk_id']} [{c['section']}]: \"{c['text'][:60]}...\"")

# Embed all chunks
chunk_embeddings = embed_model.encode([c["text"] for c in chunks])

# ============================================================
# STEP 3: Retrieve with source tracking
# ============================================================

def retrieve(query, top_k=3):
    query_embedding = embed_model.encode([query])
    scores = cosine_similarity(query_embedding, chunk_embeddings)[0]
    top_indices = scores.argsort()[-top_k:][::-1]

    results = []
    for idx in top_indices:
        chunk = chunks[idx]
        results.append({
            "text": chunk["text"],
            "section": chunk["section"],
            "chunk_id": chunk["chunk_id"],
            "score": round(float(scores[idx]), 3)
        })
    return results

# ============================================================
# STEP 4: Generate answer with citations
# ============================================================

def generate_with_citations(query, retrieved_docs):
    context = ""
    for doc in retrieved_docs:
        context += f"[Source: {doc['section']}, Chunk {doc['chunk_id']}]\n{doc['text']}\n\n"

    prompt = f"""Answer the question using ONLY the provided sources. 
Follow these rules:
1. After each claim, cite the source in brackets like [Source: SECTION NAME]
2. If the sources don't contain the answer, say "This information is not in the report."
3. Be concise — 2-4 sentences max.

SOURCES:
{context}

QUESTION: {query}

ANSWER:"""

    response = claude.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text

# ============================================================
# STEP 5: Complete RAG pipeline with citations
# ============================================================

def ask(query):
    print(f"\nQ: {query}")
    print("-" * 50)

    retrieved = retrieve(query, top_k=3)
    print("Sources found:")
    for doc in retrieved:
        print(f"  [{doc['score']}] {doc['section']} (Chunk {doc['chunk_id']}): \"{doc['text'][:55]}...\"")

    answer = generate_with_citations(query, retrieved)
    print(f"\nA: {answer}")

# ============================================================
# TEST
# ============================================================

print("\n" + "=" * 60)
print("  RAG WITH SOURCE CITATIONS")
print("=" * 60)

ask("What was the total Q4 revenue and how did it compare to last year?")

ask("Why did the south region underperform?")

ask("Which product has the best upgrade path?")

ask("What are the plans for 2026?")

ask("What is our marketing budget?")