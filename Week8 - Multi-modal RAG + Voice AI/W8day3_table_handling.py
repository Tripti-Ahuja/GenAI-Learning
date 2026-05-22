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
# THE PROBLEM: Raw table extraction is messy
# ============================================================

# This is what you'd get from a PDF text extractor
raw_table_text = """Region Q3 Revenue Q4 Revenue Growth
North $800K $920K 15%
South $350K $310K -8%
West $600K $740K 23%
East $45K $61.5K 37%"""

# Looks okay here but in real PDFs it often becomes:
garbled_table = "Region Q3 Revenue Q4 Revenue Growth North $800K $920K 15% South $350K $310K -8% West $600K $740K 23% East $45K $61.5K 37%"

print("=" * 60)
print("  THE TABLE PROBLEM")
print("=" * 60)
print(f"\n  Raw extraction (garbled):")
print(f"  \"{garbled_table[:80]}...\"")
print(f"\n  Embedding this garbled text = poor search results")

# ============================================================
# APPROACH 1: Convert table to structured descriptions
# ============================================================

def table_to_descriptions(table_data):
    """Convert each row into a natural language sentence"""
    descriptions = []
    for row in table_data:
        desc = f"{row['region']} region had Q3 revenue of {row['q3']} and Q4 revenue of {row['q4']}, representing {row['growth']} growth."
        descriptions.append({
            "text": desc,
            "source": "revenue_table",
            "type": "table_row"
        })

    # Also create a summary row
    total_q4 = sum(row['q4_num'] for row in table_data)
    best = max(table_data, key=lambda x: x['q4_num'])
    worst = min(table_data, key=lambda x: x['q4_num'])
    summary = f"Regional revenue table summary: Total Q4 revenue ${total_q4:,}. Best performing region: {best['region']} at {best['q4']}. Worst performing: {worst['region']} at {worst['q4']}."
    descriptions.append({
        "text": summary,
        "source": "revenue_table",
        "type": "table_summary"
    })
    return descriptions

table_data = [
    {"region": "North", "q3": "$800K", "q4": "$920K", "growth": "15%", "q4_num": 920000},
    {"region": "South", "q3": "$350K", "q4": "$310K", "growth": "-8%", "q4_num": 310000},
    {"region": "West", "q3": "$600K", "q4": "$740K", "growth": "23%", "q4_num": 740000},
    {"region": "East", "q3": "$45K", "q4": "$61.5K", "growth": "37%", "q4_num": 61500},
]

table_descriptions = table_to_descriptions(table_data)

print(f"\n{'='*60}")
print("  APPROACH 1: Table → Natural Language Descriptions")
print(f"{'='*60}")
for desc in table_descriptions:
    print(f"  [{desc['type']}] {desc['text']}")

# ============================================================
# APPROACH 2: Table as markdown (preserves structure)
# ============================================================

markdown_table = """| Region | Q3 Revenue | Q4 Revenue | Growth |
|--------|-----------|-----------|--------|
| North  | $800K     | $920K     | 15%    |
| South  | $350K     | $310K     | -8%    |
| West   | $600K     | $740K     | 23%    |
| East   | $45K      | $61.5K    | 37%    |"""

markdown_description = {
    "text": f"Revenue comparison table by region showing Q3 vs Q4 performance: {markdown_table}",
    "source": "revenue_table_markdown",
    "type": "table_markdown"
}

print(f"\n{'='*60}")
print("  APPROACH 2: Table as Markdown")
print(f"{'='*60}")
print(f"  {markdown_description['text'][:100]}...")

# ============================================================
# STEP 3: Combine all document types and search
# ============================================================

# Regular text documents
text_docs = [
    {"text": "Q4 2025 total revenue was $2.3 million, up 15% year over year", "source": "report", "type": "text"},
    {"text": "Enterprise Plan is top product at $208K total revenue", "source": "report", "type": "text"},
    {"text": "Customer churn rate dropped to 3.2% in Q4", "source": "report", "type": "text"},
    {"text": "Dashboard Pro leads with 9 orders but lowest revenue per unit", "source": "report", "type": "text"},
]

# All documents combined: text + table descriptions + markdown table
all_docs = text_docs + table_descriptions + [markdown_description]

all_embeddings = embed_model.encode([d["text"] for d in all_docs])

def search(query, top_k=3):
    query_vec = embed_model.encode([query])
    scores = cosine_similarity(query_vec, all_embeddings)[0]
    top_idx = scores.argsort()[-top_k:][::-1]
    return [(idx, round(float(scores[idx]), 3)) for idx in top_idx]

def ask(query):
    print(f"\nQ: {query}")
    print("-" * 55)

    results = search(query, top_k=3)
    print("Sources:")
    for idx, score in results:
        doc = all_docs[idx]
        icon = "📊" if "table" in doc["type"] else "📄"
        print(f"  {icon} [{score}] [{doc['type']}] {doc['text'][:60]}...")

    context = "\n\n".join([f"[{all_docs[idx]['type']} - {all_docs[idx]['source']}] {all_docs[idx]['text']}" for idx, _ in results])

    prompt = f"""Answer based ONLY on the sources. Be concise. Cite sources.

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
    print(f"\nA: {response.content[0].text}")

# ============================================================
# TEST
# ============================================================

print(f"\n{'='*60}")
print("  MULTI-SOURCE RAG: Text + Tables")
print(f"{'='*60}")

ask("Which region grew the fastest?")
ask("Compare north and south region performance")
ask("What was the worst performing region?")
ask("How did west region do from Q3 to Q4?")
ask("What is our total Q4 revenue?")