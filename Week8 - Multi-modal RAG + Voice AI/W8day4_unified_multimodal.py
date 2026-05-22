import anthropic
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import numpy as np
from dotenv import load_dotenv

load_dotenv()
claude = anthropic.Anthropic()

print("Loading model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Ready!\n")

# ============================================================
# ALL DOCUMENT TYPES IN ONE STORE
# ============================================================

documents = [
    # Text documents (from PDF report)
    {"text": "Q4 2025 revenue was $2.3 million, up 15% year over year. Strongest quarter ever.", "type": "text", "source": "Q4_report.pdf"},
    {"text": "South region had weakest performance at $310K, down 8% from Q3. Two accounts churned citing Salesforce issues.", "type": "text", "source": "Q4_report.pdf"},
    {"text": "Enterprise Plan is top product at $208K revenue with $40K average deal size.", "type": "text", "source": "Q4_report.pdf"},
    {"text": "Customer churn dropped to 3.2% in Q4 from 5.1% in Q3. New onboarding process helped.", "type": "text", "source": "Q4_report.pdf"},
    {"text": "Dashboard Pro leads with 9 orders. 40% of customers upgrade to Analytics Suite within 12 months.", "type": "text", "source": "Q4_report.pdf"},
    {"text": "Management targets 25 new customer acquisitions in 2026.", "type": "text", "source": "Q4_report.pdf"},

    # Image descriptions (from Claude Vision)
    {"text": "Bar chart showing Q4 revenue by region: North $920K, West $740K, South $310K, East $61.5K.", "type": "image", "source": "revenue_chart.png"},
    {"text": "Line chart showing monthly revenue trend. Peaks in April ($64K) and September ($65K). Dip in June ($22.5K).", "type": "image", "source": "monthly_trend.png"},
    {"text": "Sales funnel: 5000 leads → 1200 qualified (24%) → 800 opportunities (67%) → 200 closed (25%). Overall 4% conversion.", "type": "image", "source": "funnel_diagram.png"},
    {"text": "Salesforce dashboard: churn rate 3.2%, onboarding success 87%, time to first value 14 days.", "type": "image", "source": "churn_dashboard.png"},

    # Table rows (from PDF tables)
    {"text": "North region: Q3 revenue $800K, Q4 revenue $920K, growth 15%.", "type": "table", "source": "revenue_table"},
    {"text": "South region: Q3 revenue $350K, Q4 revenue $310K, growth -8%.", "type": "table", "source": "revenue_table"},
    {"text": "West region: Q3 revenue $600K, Q4 revenue $740K, growth 23%.", "type": "table", "source": "revenue_table"},
    {"text": "East region: Q3 revenue $45K, Q4 revenue $61.5K, growth 37%.", "type": "table", "source": "revenue_table"},

    # Table summary
    {"text": "Revenue table summary: Total Q4 revenue across regions $2.03M. Best region: North ($920K). Fastest growth: East (37%). Declining: South (-8%).", "type": "table_summary", "source": "revenue_table"},
]

# ============================================================
# HYBRID SEARCH SETUP
# ============================================================

doc_texts = [d["text"] for d in documents]
doc_embeddings = embed_model.encode(doc_texts)
tokenized = [t.lower().split() for t in doc_texts]
bm25 = BM25Okapi(tokenized)

print(f"Loaded {len(documents)} documents:")
print(f"  📄 {sum(1 for d in documents if d['type'] == 'text')} text")
print(f"  🖼️ {sum(1 for d in documents if d['type'] == 'image')} images")
print(f"  📊 {sum(1 for d in documents if d['type'] in ['table', 'table_summary'])} table entries")

# ============================================================
# SEARCH + GENERATE
# ============================================================

def hybrid_search(query, top_k=4):
    query_vec = embed_model.encode([query])
    neural_scores = cosine_similarity(query_vec, doc_embeddings)[0]
    bm25_scores = bm25.get_scores(query.lower().split())

    n_norm = neural_scores / neural_scores.max() if neural_scores.max() > 0 else neural_scores
    b_norm = bm25_scores / bm25_scores.max() if bm25_scores.max() > 0 else bm25_scores

    combined = 0.5 * n_norm + 0.5 * b_norm
    top_idx = combined.argsort()[-top_k:][::-1]

    results = []
    for idx in top_idx:
        doc = documents[idx]
        results.append({
            "text": doc["text"],
            "type": doc["type"],
            "source": doc["source"],
            "score": round(float(combined[idx]), 3)
        })
    return results

def generate_answer(query, results, voice_mode=False):
    context = "\n\n".join([f"[{r['type'].upper()} - {r['source']}] {r['text']}" for r in results])

    if voice_mode:
        length_instruction = "Keep answer to 1-2 sentences since it will be read aloud. No markdown, no bullet points."
    else:
        length_instruction = "Be concise — 2-3 sentences. Cite sources."

    prompt = f"""Answer based ONLY on the sources. {length_instruction}

SOURCES:
{context}

QUESTION: {query}

ANSWER:"""

    response = claude.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text

# ============================================================
# VOICE PIPELINE
# ============================================================

def speech_to_text(audio):
    print(f"  🎤 [STT] \"{audio}\"")
    return audio

def text_to_speech(text):
    clean = text.replace("**", "").replace("[", "").replace("]", "")
    print(f"  🔊 [TTS] \"{clean[:100]}...\"")
    return clean

# ============================================================
# UNIFIED ASK FUNCTION
# ============================================================

def ask(query, voice=False):
    if voice:
        print(f"\n  🎤 Voice query")
        query = speech_to_text(query)
    else:
        print(f"\nQ: {query}")
    print("-" * 55)

    results = hybrid_search(query, top_k=4)
    print(f"  Sources ({len(results)}):")
    for r in results:
        icons = {"text": "📄", "image": "🖼️", "table": "📊", "table_summary": "📊"}
        icon = icons.get(r["type"], "📄")
        print(f"    {icon} [{r['score']}] [{r['type']}] {r['source']}: {r['text'][:50]}...")

    answer = generate_answer(query, results, voice_mode=voice)

    if voice:
        print(f"  🤖 {answer}")
        text_to_speech(answer)
    else:
        print(f"\nA: {answer}")

# ============================================================
# DEMO
# ============================================================

print(f"\n{'='*60}")
print("  UNIFIED MULTI-MODAL RAG")
print("  Text + Images + Tables + Voice")
print(f"{'='*60}")

# Text queries
ask("What does the revenue breakdown look like?")
ask("How is our sales funnel?")
ask("Which region is growing fastest?")

# Voice queries — shorter answers
ask("What was our total revenue?", voice=True)
ask("How's customer churn?", voice=True)

# ============================================================
# INTERACTIVE
# ============================================================

print(f"\n{'='*60}")
print("  INTERACTIVE MODE")
print("  Type question, or 'voice: question' for voice mode")
print("  Type 'quit' to exit")
print(f"{'='*60}")

while True:
    user_input = input("\nYou: ").strip()
    if not user_input:
        continue
    if user_input.lower() == "quit":
        print("Goodbye!")
        break
    if user_input.lower().startswith("voice:"):
        ask(user_input[6:].strip(), voice=True)
    else:
        ask(user_input)