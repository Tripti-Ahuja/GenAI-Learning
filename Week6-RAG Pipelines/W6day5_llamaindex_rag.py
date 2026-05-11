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
# STEP 1: LlamaIndex-style document loading and node creation
# ============================================================

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

# LlamaIndex calls documents "Documents" and chunks "Nodes"
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

# Create a LlamaIndex Document
document = Document(text=report, metadata={"source": "Q4_2025_Report", "type": "quarterly_report"})

# LlamaIndex uses "SentenceSplitter" to create "Nodes" (their word for chunks)
splitter = SentenceSplitter(chunk_size=300, chunk_overlap=50)
nodes = splitter.get_nodes_from_documents([document])

print(f"LlamaIndex created {len(nodes)} nodes:")
for i, node in enumerate(nodes):
    print(f"  Node {i+1} ({len(node.text)} chars): \"{node.text[:60]}...\"")
    print(f"    Metadata: {node.metadata}")

# ============================================================
# STEP 2: Embed nodes and build search
# ============================================================

node_texts = [node.text for node in nodes]
node_embeddings = embed_model.encode(node_texts)

def retrieve(query, top_k=3):
    query_embedding = embed_model.encode([query])
    scores = cosine_similarity(query_embedding, node_embeddings)[0]
    top_indices = scores.argsort()[-top_k:][::-1]

    results = []
    for idx in top_indices:
        node = nodes[idx]
        results.append({
            "text": node.text,
            "score": round(float(scores[idx]), 3),
            "node_id": idx + 1,
            "metadata": node.metadata
        })
    return results

# ============================================================
# STEP 3: Generate with citations
# ============================================================

def ask(query):
    print(f"\nQ: {query}")
    print("-" * 50)

    retrieved = retrieve(query, top_k=3)
    print("Retrieved nodes:")
    for doc in retrieved:
        print(f"  [{doc['score']}] Node {doc['node_id']}: \"{doc['text'][:55]}...\"")

    context = "\n\n".join([f"[Node {d['node_id']}, Source: {d['metadata']['source']}]\n{d['text']}" for d in retrieved])

    prompt = f"""Answer based ONLY on the provided context. Cite sources as [Node N].
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
# STEP 4: Compare all 3 approaches
# ============================================================

print("\n" + "=" * 60)
print("  FRAMEWORK COMPARISON")
print("=" * 60)

print(f"""
  From Scratch (Day 1-3):
    - You write everything: chunking, embedding, search, prompting
    - Full control, more code, best for learning
    - Chunks: section-based with manual metadata

  LangChain (Day 4):
    - RecursiveCharacterTextSplitter handles chunking
    - Less code, industry standard, general purpose
    - Chunks: character-based with smart separators

  LlamaIndex (Today):
    - Document → Nodes pipeline
    - Built specifically for RAG, less code
    - Chunks: SentenceSplitter keeps sentences intact
    - Metadata automatically carried from Document to Nodes
""")

# ============================================================
# STEP 5: Test queries
# ============================================================

print("=" * 60)
print("  RAG WITH LLAMAINDEX")
print("=" * 60)

ask("What was Q4 revenue?")
ask("Why did the south region struggle?")
ask("Which product has the best upgrade path?")
ask("What are the 2026 plans?")
ask("What is our employee count?")