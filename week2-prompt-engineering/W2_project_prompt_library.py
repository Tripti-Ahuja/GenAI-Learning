import anthropic
import json
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic()

def clean_json(text):
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()

def call_claude(prompt, max_tokens=200):
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=max_tokens,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    return json.loads(clean_json(response.content[0].text))

# ============================================================
# TEMPLATE 1: Review Classifier (from Day 5)
# ============================================================
def classify_review(review):
    prompt = f"""Classify this customer review. Return ONLY valid JSON, no markdown.

Review: "{review}"

{{"sentiment": "positive/negative/neutral", "confidence": "high/medium/low", "topic": "main topic in 2-3 words"}}"""
    return call_claude(prompt, 100)

# ============================================================
# TEMPLATE 2: SQL Explainer (from Day 5)
# ============================================================
def explain_sql(query):
    prompt = f"""Explain this SQL query in plain English. Return ONLY valid JSON, no markdown.

Query: {query}

{{"explanation": "what it does in one sentence", "tables_used": ["list"], "complexity": "simple/medium/complex"}}"""
    return call_claude(prompt, 150)

# ============================================================
# TEMPLATE 3: Data Summarizer (from Day 5)
# ============================================================
def summarize_data(data_description, audience):
    prompt = f"""Summarize this data for a {audience}. Return ONLY valid JSON, no markdown.

Data: {data_description}

{{"summary": "2-3 sentence summary", "key_metric": "most important number", "action": "recommended next step"}}"""
    return call_claude(prompt, 200)

# ============================================================
# TEMPLATE 4: Salesforce Ticket Router (NEW)
# ============================================================
def route_ticket(ticket_description):
    prompt = f"""You are a Salesforce support ticket router. Analyze this ticket and return ONLY valid JSON, no markdown.

Ticket: "{ticket_description}"

{{"category": "bug/feature_request/billing/onboarding/general_inquiry", "priority": "P1_critical/P2_high/P3_medium/P4_low", "department": "engineering/sales/support/billing", "estimated_response": "1hr/4hr/24hr/48hr", "auto_reply": "a short 1-2 sentence acknowledgment to send the customer"}}"""
    return call_claude(prompt, 200)

# ============================================================
# TEMPLATE 5: Dashboard Description Generator (NEW)
# ============================================================
def describe_dashboard(metrics, dashboard_name):
    prompt = f"""You are a Tableau dashboard documentation writer. Return ONLY valid JSON, no markdown.

Dashboard: {dashboard_name}
Metrics shown: {metrics}

{{"title": "dashboard title", "purpose": "what business question this answers in one sentence", "key_metrics": ["list of 3-4 most important metrics"], "suggested_filters": ["list of 2-3 filters users should add"], "audience": "who should use this dashboard"}}"""
    return call_claude(prompt, 200)

# ============================================================
# TEST ALL 5 TEMPLATES
# ============================================================

print("=" * 60)
print("  WEEK 2 PROJECT: Prompt Library (5 Templates)")
print("=" * 60)

# Template 1
print("\n--- 1. REVIEW CLASSIFIER ---")
reviews = [
    "Love the new dashboard feature, saves me hours!",
    "App crashed again. Third time this week.",
    "It works. Does what it says.",
]
for review in reviews:
    r = classify_review(review)
    print(f"  {r['sentiment']:<10} ({r['confidence']}) - {r['topic']:<20} | {review[:45]}")

# Template 2
print("\n--- 2. SQL EXPLAINER ---")
q = "SELECT region, SUM(amount) FROM orders WHERE date > '2024-01-01' GROUP BY region HAVING SUM(amount) > 10000"
r = explain_sql(q)
print(f"  [{r['complexity']}] {r['explanation']}")
print(f"  Tables: {', '.join(r['tables_used'])}")

# Template 3
print("\n--- 3. DATA SUMMARIZER ---")
r = summarize_data(
    "Q4 revenue: $2.3M (up 15% YoY). Top region: North ($900K). Weakest: South ($300K, down 8%).",
    "C-level executive"
)
print(f"  Summary:    {r['summary']}")
print(f"  Key Metric: {r['key_metric']}")
print(f"  Action:     {r['action']}")

# Template 4
print("\n--- 4. SALESFORCE TICKET ROUTER ---")
tickets = [
    "Dashboard has been loading for 10 minutes and then crashes. Our whole team is blocked.",
    "Can you add a feature to export reports as PDF? Currently only CSV is available.",
    "I was charged twice for my monthly subscription. Please refund.",
]
for ticket in tickets:
    r = route_ticket(ticket)
    print(f"  [{r['priority']}] {r['category']} → {r['department']} (respond in {r['estimated_response']})")
    print(f"  Auto-reply: {r['auto_reply']}\n")

# Template 5
print("--- 5. DASHBOARD DESCRIPTION ---")
r = describe_dashboard(
    "monthly revenue, customer churn rate, new signups, average deal size, pipeline value",
    "Sales Performance Dashboard"
)
print(f"  Title:     {r['title']}")
print(f"  Purpose:   {r['purpose']}")
print(f"  Metrics:   {', '.join(r['key_metrics'])}")
print(f"  Filters:   {', '.join(r['suggested_filters'])}")
print(f"  Audience:  {r['audience']}")