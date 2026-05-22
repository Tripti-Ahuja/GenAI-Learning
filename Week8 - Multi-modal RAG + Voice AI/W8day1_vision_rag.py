import anthropic
import base64
import os
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
# STEP 1: Simulate image descriptions using Claude Vision
# (In production, you'd send actual images to the API)
# ============================================================

# In a real app, you'd do this:
# with open("chart.png", "rb") as f:
#     image_data = base64.b64encode(f.read()).decode("utf-8")
# response = claude.messages.create(
#     model="claude-haiku-4-5-20251001",
#     max_tokens=300,
#     messages=[{
#         "role": "user",
#         "content": [
#             {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_data}},
#             {"type": "text", "text": "Describe this chart in detail including all data points and trends."}
#         ]
#     }]
# )

# For learning, we'll simulate the image descriptions
# as if Claude Vision already described these charts

image_descriptions = [
    {
        "source": "revenue_bar_chart.png",
        "type": "chart",
        "description": "Bar chart showing Q4 2025 revenue by region. North leads at $920K, followed by West at $740K, South at $310K, and East at $61.5K. Total revenue is $2.3 million. Chart title: Regional Revenue Q4 2025."
    },
    {
        "source": "growth_line_chart.png",
        "type": "chart",
        "description": "Line chart showing monthly revenue from January to October 2025. Revenue starts at $60K in January, peaks at $64K in April, dips to $22.5K in June, and recovers to $65K in September. Overall upward trend with seasonal dip in summer months."
    },
    {
        "source": "product_pie_chart.png",
        "type": "chart",
        "description": "Pie chart showing revenue distribution by product. Enterprise Plan: 43% ($208K), Analytics Suite: 34% ($163K), Dashboard Pro: 23% ($111.5K). Enterprise Plan is the dominant revenue source."
    },
    {
        "source": "churn_dashboard_screenshot.png",
        "type": "dashboard",
        "description": "Salesforce dashboard screenshot showing customer churn metrics. Q4 churn rate: 3.2%, down from 5.1% in Q3. Green indicator showing improvement. Onboarding success rate: 87%. Average time to first value: 14 days."
    },
    {
        "source": "sales_funnel_diagram.png",
        "type": "diagram",
        "description": "Sales funnel diagram showing: 5000 leads at top, 1200 qualified leads (24% conversion), 800 opportunities (66.7% conversion), 200 closed won (25% conversion). Overall lead-to-close rate: 4%. Average deal size: $24,125."
    },
]

# Regular text documents (same as before)
text_documents = [
    {"source": "Q4_report.pdf", "type": "text", "description": "Q4 2025 revenue was $2.3 million, up 15% year over year. Strongest quarter in company history."},
    {"source": "Q4_report.pdf", "type": "text", "description": "South region had weakest performance at $310K, down 8% from Q3. Two key accounts churned citing Salesforce integration issues."},
    {"source": "Q4_report.pdf", "type": "text", "description": "Enterprise Plan is top product at $208K total revenue with $40K average deal size."},
    {"source": "Q4_report.pdf", "type": "text", "description": "Customer churn rate dropped to 3.2% in Q4, down from 5.1% in Q3. New onboarding process helped."},
    {"source": "Q4_report.pdf", "type": "text", "description": "Dashboard Pro leads in order volume with 9 orders. 40% of customers upgrade to Analytics Suite within 12 months."},
]

# ============================================================
# STEP 2: Combine all sources into one unified collection
# ============================================================

all_documents = image_descriptions + text_documents

print(f"Document store: {len(image_descriptions)} image descriptions + {len(text_documents)} text documents = {len(all_documents)} total\n")

# Embed everything — images and text use the same embedding
all_texts = [doc["description"] for doc in all_documents]
all_embeddings = embed_model.encode(all_texts)

# ============================================================
# STEP 3: Unified search across text AND images
# ============================================================

def search(query, top_k=3):
    query_vec = embed_model.encode([query])
    scores = cosine_similarity(query_vec, all_embeddings)[0]
    top_idx = scores.argsort()[-top_k:][::-1]

    results = []
    for idx in top_idx:
        doc = all_documents[idx]
        results.append({
            "source": doc["source"],
            "type": doc["type"],
            "description": doc["description"],
            "score": round(float(scores[idx]), 3)
        })
    return results

def ask(query):
    print(f"Q: {query}")
    print("-" * 55)

    results = search(query, top_k=3)
    print("Sources found:")
    for r in results:
        icon = "🖼️" if r["type"] in ["chart", "dashboard", "diagram"] else "📄"
        print(f"  {icon} [{r['score']}] [{r['type']}] {r['source']}: {r['description'][:55]}...")

    context = "\n\n".join([f"[{r['type'].upper()} - {r['source']}] {r['description']}" for r in results])

    prompt = f"""Answer based ONLY on the provided sources. These sources include both text documents and descriptions of charts/images.
Cite sources as [Source: filename]. Be concise — 2-3 sentences.

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
    print(f"\nA: {response.content[0].text}\n")

# ============================================================
# TEST
# ============================================================

print("=" * 60)
print("  MULTI-MODAL RAG: Text + Images unified search")
print("=" * 60 + "\n")

ask("What does the revenue breakdown look like by region?")
ask("How is our sales funnel performing?")
ask("What's the trend in monthly revenue?")
ask("How did customer churn change?")
ask("Which product generates the most revenue?")