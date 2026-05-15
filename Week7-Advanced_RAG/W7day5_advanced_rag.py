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
# DOCUMENT STORE
# ============================================================

documents = [
    {"text": "Q4 2025 revenue was $2.3 million, up 15% year over year. Strongest quarter in company history.", "section": "Executive Summary"},
    {"text": "North region led with $920K in Q4 revenue. 15% growth from Q3 driven by new sales playbook introduced in September.", "section": "Revenue by Region"},
    {"text": "South region had weakest performance at $310K, down 8% from Q3. Two key accounts churned citing Salesforce integration issues.", "section": "Revenue by Region"},
    {"text": "West region contributed $740K driven by Analytics Suite adoption among mid-market customers transitioning from Tableau.", "section": "Revenue by Region"},
    {"text": "East region generated $61,500. Newest territory launched July 2024 showing 12% month-over-month growth.", "section": "Revenue by Region"},
    {"text": "Enterprise Plan is top product at $208K revenue. Average deal size $40K with 6-month sales cycles. Satisfaction score 4.5/5.", "section": "Products"},
    {"text": "Analytics Suite generated $163K across 5 deals. Popular with Tableau and PowerBI users. Salesforce connector planned Q1 2026.", "section": "Products"},
    {"text": "Dashboard Pro leads in order volume with 9 orders but lowest revenue per unit at $12K. 40% of customers upgrade to Analytics Suite within 12 months.", "section": "Products"},
    {"text": "Customer churn rate dropped to 3.2% in Q4, down from 5.1% in Q3. New onboarding process with dedicated account managers for 90 days helped.", "section": "Customer Metrics"},
    {"text": "Amit Patel is highest-value customer at $90,000 across 3 orders. Based in north region since Q1 2024.", "section": "Customer Metrics"},
    {"text": "Rajesh Kumar placed 3 orders totaling $64,000 since January 2024. Uses both Dashboard Pro and Analytics Suite.", "section": "Customer Metrics"},
    {"text": "Average order value across all products is $24,125. Enterprise Plan pulls this average up significantly.", "section": "Products"},
    {"text": "Monthly revenue peaked in April 2025 at $64,000 driven by 2 Enterprise Plan renewals.", "section": "Revenue Trends"},
    {"text": "Management targets 25 new customer acquisitions in 2026. Plans include east region expansion and Salesforce connector launch.", "section": "Outlook"},
    {"text": "New mid-tier pricing plan to bridge gap between Dashboard Pro ($12K) and Enterprise Plan ($40K) launching in 2026.", "section": "Outlook"},
]

# Pre-compute all search indexes
doc_texts = [d["text"] for d in documents]
doc_embeddings = embed_model.encode(doc_texts)
tokenized_docs = [doc.lower().split() for doc in doc_texts]
bm25 = BM25Okapi(tokenized_docs)

print(f"Loaded {len(documents)} documents\n")

# ============================================================
# STEP 1: Query expansion
# ============================================================

def expand_query(query):
    prompt = f"""Rephrase this question 2 different ways using different words. Return ONLY a JSON array, no markdown.

Original: "{query}"
Example: ["version 1", "version 2"]"""

    response = claude_client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=150,
        temperature=0.3,
        messages=[{"role": "user", "content": prompt}]
    )
    text = response.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("```")[1].replace("json", "").strip()
    try:
        versions = json.loads(text)
        return [query] + versions
    except:
        return [query]

# ============================================================
# STEP 2: Hybrid search
# ============================================================

def hybrid_search(query, top_k=5, neural_weight=0.5):
    query_vec = embed_model.encode([query])
    neural_scores = cosine_similarity(query_vec, doc_embeddings)[0]
    bm25_scores = bm25.get_scores(query.lower().split())

    neural_norm = neural_scores / neural_scores.max() if neural_scores.max() > 0 else neural_scores
    bm25_norm = bm25_scores / bm25_scores.max() if bm25_scores.max() > 0 else bm25_scores

    combined = neural_weight * neural_norm + (1 - neural_weight) * bm25_norm
    top_idx = combined.argsort()[-top_k:][::-1]
    return [(idx, round(float(combined[idx]), 3)) for idx in top_idx]

# ============================================================
# STEP 3: Expanded search (multiple query versions)
# ============================================================

def expanded_search(query, top_k=5):
    all_queries = expand_query(query)
    doc_scores = {}
    for q in all_queries:
        results = hybrid_search(q, top_k=5)
        for idx, score in results:
            if idx not in doc_scores:
                doc_scores[idx] = []
            doc_scores[idx].append(score)

    final = {idx: max(scores) for idx, scores in doc_scores.items()}
    sorted_results = sorted(final.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return sorted_results

# ============================================================
# STEP 4: Re-rank
# ============================================================

def rerank_results(query, results, top_k=3):
    if not results:
        return []
    pairs = [[query, doc_texts[idx]] for idx, _ in results]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)
    return [(idx, round(float(rerank_score), 3)) for (idx, _), rerank_score in ranked[:top_k]]

# ============================================================
# STEP 5: Generate answer with citations
# ============================================================

def generate_answer(query, final_results):
    context = ""
    for idx, score in final_results:
        doc = documents[idx]
        context += f"[Source: {doc['section']}] {doc['text']}\n\n"

    prompt = f"""Answer the question using ONLY the provided sources.
Rules:
1. Cite sources as [Source: SECTION NAME] after each claim
2. If sources don't have the answer, say "Not found in the report."
3. Be concise — 2-3 sentences max.

SOURCES:
{context}

QUESTION: {query}

ANSWER:"""

    response = claude_client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=250,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text

# ============================================================
# FULL PIPELINE
# ============================================================

def ask(query, show_steps=True):
    if show_steps:
        print(f"\nQ: {query}")
        print("-" * 55)

    # Step 1: Expand query
    expanded = expand_query(query)
    if show_steps:
        print(f"  Expanded: {expanded}")

    # Step 2: Hybrid search with all versions
    search_results = expanded_search(query, top_k=5)
    if show_steps:
        print(f"  Hybrid search found {len(search_results)} docs")

    # Step 3: Re-rank
    reranked = rerank_results(query, search_results, top_k=3)
    if show_steps:
        print(f"  After re-ranking top 3:")
        for idx, score in reranked:
            print(f"    [{score}] [{documents[idx]['section']}] {documents[idx]['text'][:55]}...")

    # Step 4: Generate answer
    answer = generate_answer(query, reranked)
    if show_steps:
        print(f"\n  Answer: {answer}")
    return answer

# ============================================================
# INTERACTIVE MODE
# ============================================================

print("=" * 60)
print("  ADVANCED RAG PIPELINE")
print("  Hybrid search + Re-ranking + Query expansion + Citations")
print("  Type 'quit' to exit")
print("=" * 60)

# Run a few test queries first
ask("How's our customer retention trending?")
ask("What product should new customers start with?")
ask("Which region needs the most help and why?")
ask("What are our growth plans?")
ask("What is our marketing budget?")

print(f"\n{'='*60}")
print("  INTERACTIVE MODE")
print(f"{'='*60}")

while True:
    user_input = input("\nYou: ").strip()
    if not user_input:
        continue
    if user_input.lower() == "quit":
        print("Goodbye!")
        break
    ask(user_input)