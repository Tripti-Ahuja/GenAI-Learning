import anthropic
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dotenv import load_dotenv

load_dotenv()
claude = anthropic.Anthropic()

print("Loading embedding model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Ready!\n")

# ============================================================
# STEP 1: Document store (your knowledge base)
# ============================================================

documents = [
    "Q4 2025 revenue was $2.3 million, up 15% year over year. This was the strongest quarter in company history.",
    "North region generated $920K in Q4 2025, leading all regions. Growth was driven by 3 large Enterprise Plan deals.",
    "South region had the weakest Q4 at $310K revenue, down 8% from Q3. Two key accounts churned during the quarter.",
    "West region contributed $740K in Q4, with strong Analytics Suite adoption among mid-market customers.",
    "East region generated $61,500 in total revenue. This is the newest region, launched in July 2024.",
    "Enterprise Plan is the top product at $208K total revenue. Average deal size is $40K with 6-month sales cycles.",
    "Dashboard Pro has the most orders (9 total) but lowest revenue per unit at ~$12K average.",
    "Analytics Suite generated $163K across 5 deals. Popular with data teams who already use Tableau or PowerBI.",
    "Amit Patel is the highest spending customer at $90,000 across 3 orders. Based in north region.",
    "Rajesh Kumar placed 3 orders totaling $64,000. Loyal customer since January 2024.",
    "Customer churn rate dropped to 3.2% in Q4 2025, down from 5.1% in Q3. New onboarding process helped.",
    "10 new customers signed up in 2024. Target for 2025 is 25 new customers across all regions.",
    "Average order value across all products is $24,125. Enterprise Plan pulls this average up significantly.",
    "Monthly revenue peaked in April 2025 at $64,000, driven by 2 Enterprise Plan renewals.",
    "Q3 to Q4 north region growth was 15%. Management credits the new sales playbook introduced in September.",
]

# Embed all documents once
doc_embeddings = embed_model.encode([doc for doc in documents])

# ============================================================
# STEP 2: Retrieval — find top relevant documents
# ============================================================

def retrieve(query, top_k=3):
    query_embedding = embed_model.encode([query])
    scores = cosine_similarity(query_embedding, doc_embeddings)[0]
    top_indices = scores.argsort()[-top_k:][::-1]

    results = []
    for idx in top_indices:
        results.append({
            "text": documents[idx],
            "score": round(float(scores[idx]), 3)
        })
    return results

# ============================================================
# STEP 3: Generation — Claude answers using retrieved context
# ============================================================

def generate_answer(query, retrieved_docs):
    # Build the context from retrieved documents
    context = "\n\n".join([f"[Source {i+1}] {doc['text']}" for i, doc in enumerate(retrieved_docs)])

    prompt = f"""Answer the user's question based ONLY on the provided context. 
If the context doesn't contain enough information, say "I don't have enough information to answer that."
Always mention which source(s) you used.

CONTEXT:
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
# STEP 4: The complete RAG pipeline
# ============================================================

def ask(query):
    print(f"\nQuestion: {query}")
    print("-" * 50)

    # Retrieve
    retrieved = retrieve(query, top_k=3)
    print("Retrieved documents:")
    for i, doc in enumerate(retrieved):
        print(f"  [{doc['score']}] {doc['text'][:70]}...")

    # Generate
    answer = generate_answer(query, retrieved)
    print(f"\nAnswer: {answer}")
    return answer

# ============================================================
# TEST THE RAG PIPELINE
# ============================================================

print("=" * 60)
print("  RAG PIPELINE: Retrieve → Augment → Generate")
print("=" * 60)

ask("What was our total revenue in Q4 2025?")

ask("Which region is struggling the most and why?")

ask("Who is our best customer?")

ask("What product should we focus on for growth?")

ask("How is customer retention trending?")

# This question has NO answer in the documents
ask("What is our marketing budget for 2026?")