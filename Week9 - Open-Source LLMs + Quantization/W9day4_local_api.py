from fastapi import FastAPI
from pydantic import BaseModel
import requests
import time
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import numpy as np

# ============================================================
# SETUP
# ============================================================

print("Loading embedding model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Ready!")

# Documents
documents = [
    {"text": "Q4 2025 revenue was $2.3 million, up 15% year over year.", "source": "Q4 Report"},
    {"text": "North region led with $920K in Q4. 15% growth from Q3.", "source": "Q4 Report"},
    {"text": "South region had weakest performance at $310K, down 8%. Two accounts churned.", "source": "Q4 Report"},
    {"text": "Enterprise Plan is top product at $208K revenue. Average deal size $40K.", "source": "Q4 Report"},
    {"text": "Customer churn dropped to 3.2% in Q4 from 5.1% in Q3.", "source": "Q4 Report"},
    {"text": "Dashboard Pro leads with 9 orders. 40% upgrade to Analytics Suite.", "source": "Q4 Report"},
    {"text": "Amit Patel is highest spending customer at $90,000.", "source": "Q4 Report"},
    {"text": "Management targets 25 new customer acquisitions in 2026.", "source": "Q4 Report"},
]

doc_texts = [d["text"] for d in documents]
doc_embeddings = embed_model.encode(doc_texts)
tokenized = [t.lower().split() for t in doc_texts]
bm25 = BM25Okapi(tokenized)

# ============================================================
# SEARCH + GENERATE FUNCTIONS
# ============================================================

def hybrid_search(query, top_k=3):
    query_vec = embed_model.encode([query])
    neural_scores = cosine_similarity(query_vec, doc_embeddings)[0]
    bm25_scores = bm25.get_scores(query.lower().split())

    n_norm = neural_scores / neural_scores.max() if neural_scores.max() > 0 else neural_scores
    b_norm = bm25_scores / bm25_scores.max() if bm25_scores.max() > 0 else bm25_scores

    combined = 0.5 * n_norm + 0.5 * b_norm
    top_idx = combined.argsort()[-top_k:][::-1]

    return [{"text": documents[idx]["text"], "source": documents[idx]["source"], "score": round(float(combined[idx]), 3)} for idx in top_idx]

def generate_answer(query, results, model="llama3.2:3b"):
    context = "\n".join([f"[{r['source']}] {r['text']}" for r in results])
    prompt = f"""Answer based ONLY on the context. Be concise — 1-2 sentences. If not found, say "Not found."

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=120
    )
    return response.json()["response"].strip()

# ============================================================
# FASTAPI SERVER
# ============================================================

app = FastAPI(title="Local RAG API", description="100% local, 100% free")

class QueryRequest(BaseModel):
    question: str
    model: str = "llama3.2:3b"
    top_k: int = 3

class QueryResponse(BaseModel):
    answer: str
    sources: list
    model: str
    time_seconds: float
    cost: float = 0.0

@app.get("/health")
def health_check():
    return {"status": "healthy", "models_available": ["llama3.2:3b", "mistral"]}

@app.post("/ask", response_model=QueryResponse)
def ask_question(req: QueryRequest):
    start = time.time()

    # Retrieve
    results = hybrid_search(req.question, top_k=req.top_k)

    # Generate
    answer = generate_answer(req.question, results, model=req.model)

    elapsed = round(time.time() - start, 1)

    return QueryResponse(
        answer=answer,
        sources=[{"text": r["text"][:80], "score": r["score"]} for r in results],
        model=req.model,
        time_seconds=elapsed,
        cost=0.0
    )

@app.get("/documents")
def list_documents():
    return {"count": len(documents), "documents": [{"text": d["text"][:60], "source": d["source"]} for d in documents]}

# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":
    import uvicorn
    print("\nStarting Local RAG API server...")
    print("Open http://localhost:8000/docs to see the interactive API docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)