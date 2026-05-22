import anthropic
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import numpy as np
import json
from dotenv import load_dotenv

load_dotenv()
claude = anthropic.Anthropic()

print("Loading model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Ready!\n")

# ============================================================
# MULTI-MODAL DOCUMENT STORE
# ============================================================

documents = [
    # Text documents
    {"text": "Q4 2025 revenue was $2.3 million, up 15% year over year. Strongest quarter in company history.", "type": "text", "source": "Q4_report.pdf", "section": "Executive Summary"},
    {"text": "North region led with $920K in Q4, driven by 3 large Enterprise Plan deals and new sales playbook.", "type": "text", "source": "Q4_report.pdf", "section": "Revenue"},
    {"text": "South region had weakest performance at $310K, down 8% from Q3. Two key accounts churned citing Salesforce integration issues.", "type": "text", "source": "Q4_report.pdf", "section": "Revenue"},
    {"text": "West region contributed $740K driven by Analytics Suite adoption among mid-market customers.", "type": "text", "source": "Q4_report.pdf", "section": "Revenue"},
    {"text": "East region generated $61,500. Newest territory launched July 2024 showing 12% month-over-month growth.", "type": "text", "source": "Q4_report.pdf", "section": "Revenue"},
    {"text": "Enterprise Plan is top product at $208K revenue. Average deal size $40K with 6-month sales cycles.", "type": "text", "source": "Q4_report.pdf", "section": "Products"},
    {"text": "Analytics Suite generated $163K across 5 deals. Popular with Tableau and PowerBI users.", "type": "text", "source": "Q4_report.pdf", "section": "Products"},
    {"text": "Dashboard Pro leads with 9 orders but lowest revenue at $12K per unit. 40% upgrade to Analytics Suite within 12 months.", "type": "text", "source": "Q4_report.pdf", "section": "Products"},
    {"text": "Customer churn dropped to 3.2% in Q4 from 5.1% in Q3. New onboarding with dedicated account managers for 90 days helped.", "type": "text", "source": "Q4_report.pdf", "section": "Customers"},
    {"text": "Amit Patel is highest-value customer at $90,000 across 3 orders. Based in north region.", "type": "text", "source": "Q4_report.pdf", "section": "Customers"},
    {"text": "Management targets 25 new customer acquisitions in 2026. Plans include east region expansion and Salesforce connector.", "type": "text", "source": "Q4_report.pdf", "section": "Outlook"},

    # Image descriptions (from Claude Vision)
    {"text": "Bar chart showing Q4 revenue by region: North $920K, West $740K, South $310K, East $61.5K. Total $2.03M.", "type": "image", "source": "revenue_chart.png", "section": "Charts"},
    {"text": "Line chart: monthly revenue Jan-Oct 2025. Peaks April ($64K) and September ($65K). Dip June ($22.5K). Upward trend.", "type": "image", "source": "monthly_trend.png", "section": "Charts"},
    {"text": "Sales funnel: 5000 leads → 1200 qualified (24%) → 800 opportunities (67%) → 200 closed (25%). Overall 4% conversion rate.", "type": "image", "source": "funnel_diagram.png", "section": "Charts"},
    {"text": "Salesforce dashboard: churn 3.2%, onboarding success 87%, time to first value 14 days, NPS score 42.", "type": "image", "source": "churn_dashboard.png", "section": "Charts"},

    # Table rows
    {"text": "North region: Q3 revenue $800K, Q4 revenue $920K, growth 15%.", "type": "table", "source": "revenue_table", "section": "Tables"},
    {"text": "South region: Q3 revenue $350K, Q4 revenue $310K, growth -8%.", "type": "table", "source": "revenue_table", "section": "Tables"},
    {"text": "West region: Q3 revenue $600K, Q4 revenue $740K, growth 23%.", "type": "table", "source": "revenue_table", "section": "Tables"},
    {"text": "East region: Q3 revenue $45K, Q4 revenue $61.5K, growth 37%.", "type": "table", "source": "revenue_table", "section": "Tables"},
    {"text": "Revenue table summary: Best region North ($920K). Fastest growth East (37%). Declining South (-8%).", "type": "table_summary", "source": "revenue_table", "section": "Tables"},
]

# ============================================================
# SEARCH SETUP
# ============================================================

doc_texts = [d["text"] for d in documents]
doc_embeddings = embed_model.encode(doc_texts)
tokenized = [t.lower().split() for t in doc_texts]
bm25 = BM25Okapi(tokenized)

type_counts = {}
for d in documents:
    t = d["type"]
    type_counts[t] = type_counts.get(t, 0) + 1
print(f"Document store: {len(documents)} total")
for t, c in type_counts.items():
    icons = {"text": "📄", "image": "🖼️", "table": "📊", "table_summary": "📊"}
    print(f"  {icons.get(t, '📄')} {t}: {c}")

# ============================================================
# HYBRID SEARCH
# ============================================================

def hybrid_search(query, top_k=4, doc_type=None):
    query_vec = embed_model.encode([query])
    neural_scores = cosine_similarity(query_vec, doc_embeddings)[0]
    bm25_scores = bm25.get_scores(query.lower().split())

    n_norm = neural_scores / neural_scores.max() if neural_scores.max() > 0 else neural_scores
    b_norm = bm25_scores / bm25_scores.max() if bm25_scores.max() > 0 else bm25_scores
    combined = 0.5 * n_norm + 0.5 * b_norm

    # Optional: filter by document type
    if doc_type:
        for i, d in enumerate(documents):
            if d["type"] != doc_type:
                combined[i] = -1

    top_idx = combined.argsort()[-top_k:][::-1]
    results = []
    for idx in top_idx:
        if combined[idx] < 0:
            continue
        doc = documents[idx]
        results.append({
            "text": doc["text"],
            "type": doc["type"],
            "source": doc["source"],
            "section": doc["section"],
            "score": round(float(combined[idx]), 3)
        })
    return results

# ============================================================
# ANSWER GENERATION
# ============================================================

def generate_answer(query, results, voice_mode=False):
    context = "\n\n".join([f"[{r['type'].upper()} | {r['source']} | {r['section']}]\n{r['text']}" for r in results])

    if voice_mode:
        style = "Answer in 1-2 short sentences for spoken output. No markdown, no bullets, no citations."
    else:
        style = "Answer in 2-3 sentences. Cite sources as [Source: filename]. Be specific with numbers."

    prompt = f"""Answer based ONLY on the provided sources. {style}
If the sources don't contain the answer, say "This information is not available in the current documents."

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
# VOICE FUNCTIONS
# ============================================================

def speech_to_text(audio):
    return audio

def text_to_speech(text):
    clean = text.replace("**", "").replace("[", "").replace("]", "")
    return clean

# ============================================================
# MAIN ASK FUNCTION
# ============================================================

def ask(query, voice=False, doc_type=None):
    if voice:
        print(f"\n  🎤 \"{query}\"")
    else:
        print(f"\nQ: {query}")
    print("-" * 55)

    if voice:
        query = speech_to_text(query)

    results = hybrid_search(query, top_k=4, doc_type=doc_type)

    icons = {"text": "📄", "image": "🖼️", "table": "📊", "table_summary": "📊"}
    print(f"  Sources:")
    for r in results:
        icon = icons.get(r["type"], "📄")
        print(f"    {icon} [{r['score']}] [{r['section']}] {r['source']}: {r['text'][:45]}...")

    answer = generate_answer(query, results, voice_mode=voice)

    if voice:
        spoken = text_to_speech(answer)
        print(f"  🤖 {answer}")
        print(f"  🔊 \"{spoken[:80]}...\"")
    else:
        print(f"\nA: {answer}")

    return answer

# ============================================================
# DEMO + INTERACTIVE
# ============================================================

print(f"\n{'='*60}")
print("  MULTI-MODAL RAG SYSTEM (Week 8 Project)")
print("  📄 Text + 🖼️ Images + 📊 Tables + 🎤 Voice")
print(f"{'='*60}")

# Demo queries
ask("What was our Q4 revenue and how did it compare to last year?")
ask("How is our sales funnel performing?")
ask("Which region is growing fastest and which is declining?")
ask("What was our total revenue?", voice=True)
ask("Tell me about customer churn", voice=True)

# Interactive
print(f"\n{'='*60}")
print("  INTERACTIVE MODE")
print("  Commands:")
print("    Type a question for text answer")
print("    'voice: question' for voice-style short answer")
print("    'images: question' to search only image sources")
print("    'tables: question' to search only table sources")
print("    'stats' to see document store summary")
print("    'quit' to exit")
print(f"{'='*60}")

while True:
    user_input = input("\nYou: ").strip()
    if not user_input:
        continue
    elif user_input.lower() == "quit":
        print("Goodbye!")
        break
    elif user_input.lower() == "stats":
        print(f"\n  Total documents: {len(documents)}")
        for t, c in type_counts.items():
            print(f"    {icons.get(t, '📄')} {t}: {c}")
    elif user_input.lower().startswith("voice:"):
        ask(user_input[6:].strip(), voice=True)
    elif user_input.lower().startswith("images:"):
        ask(user_input[7:].strip(), doc_type="image")
    elif user_input.lower().startswith("tables:"):
        ask(user_input[7:].strip(), doc_type="table")
    else:
        ask(user_input)