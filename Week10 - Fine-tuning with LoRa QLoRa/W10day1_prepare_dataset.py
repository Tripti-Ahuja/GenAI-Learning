import json
import anthropic
from dotenv import load_dotenv

load_dotenv()
claude = anthropic.Anthropic()

# ============================================================
# STEP 1: Manual dataset — YOUR domain knowledge
# ============================================================

# These are examples YOU write — domain-specific Q&As
# In production, you'd have 500-1000+ of these from real support tickets,
# documentation, internal wikis, etc.

manual_dataset = [
    # Product knowledge
    {"instruction": "What products does the company offer?", "response": "We offer three products: Enterprise Plan ($40K avg deal, 6-month sales cycle), Analytics Suite ($163K total revenue, popular with Tableau/PowerBI users), and Dashboard Pro ($12K per unit, entry-level product)."},
    {"instruction": "What is Dashboard Pro?", "response": "Dashboard Pro is our entry-level product at $12K per unit. It leads in order volume with 9 orders. 40% of Dashboard Pro customers upgrade to Analytics Suite within 12 months."},
    {"instruction": "What is the Enterprise Plan?", "response": "Enterprise Plan is our flagship product at $208K total revenue. Average deal size is $40K with typical 6-month sales cycles. Customer satisfaction scores average 4.5 out of 5."},
    {"instruction": "What is Analytics Suite?", "response": "Analytics Suite generated $163K across 5 deals. It's popular with data teams already using Tableau or PowerBI. A Salesforce connector is planned for Q1 2026."},
    {"instruction": "Which product has the highest revenue?", "response": "Enterprise Plan has the highest revenue at $208K, followed by Analytics Suite at $163K and Dashboard Pro at $111.5K."},
    {"instruction": "What is the upgrade path for customers?", "response": "40% of Dashboard Pro customers upgrade to Analytics Suite within 12 months. Dashboard Pro serves as an effective entry point into our product ecosystem."},

    # Regional performance
    {"instruction": "How is the north region performing?", "response": "North region leads all regions with $920K in Q4 revenue, representing 15% growth from Q3. This growth is attributed to the new sales playbook introduced in September."},
    {"instruction": "Why is the south region underperforming?", "response": "South region had the weakest performance at $310K, down 8% from Q3. Two key accounts churned during the quarter citing integration issues with their Salesforce workflows."},
    {"instruction": "How is the east region doing?", "response": "East region generated $61,500. It's our newest territory launched in July 2024. Despite being the smallest, it shows promising 12% month-over-month growth."},
    {"instruction": "What is the west region revenue?", "response": "West region contributed $740K in Q4, driven by strong Analytics Suite adoption among mid-market customers transitioning from Tableau and PowerBI."},
    {"instruction": "Which region has the highest revenue?", "response": "North region leads with $920K in Q4 revenue, followed by West ($740K), South ($310K), and East ($61.5K)."},

    # Customer metrics
    {"instruction": "What is our customer churn rate?", "response": "Customer churn rate dropped to 3.2% in Q4, down from 5.1% in Q3. The improvement is attributed to the new onboarding process with dedicated account managers for the first 90 days."},
    {"instruction": "Who is our top customer?", "response": "Amit Patel is our highest-value customer with $90,000 in total spending across 3 orders. He is based in the north region and has been a customer since Q1 2024."},
    {"instruction": "How many new customers do we target for 2026?", "response": "Management targets 25 new customer acquisitions in 2026, up from 10 in 2024."},
    {"instruction": "What is the average order value?", "response": "Average order value across all products is $24,125. Enterprise Plan pulls this average up significantly due to its $40K average deal size."},

    # Company metrics
    {"instruction": "What was Q4 2025 revenue?", "response": "Q4 2025 revenue was $2.3 million, up 15% year over year. This was the strongest quarter in company history."},
    {"instruction": "What is our total revenue?", "response": "Total revenue across all orders is $482,500, with Enterprise Plan contributing $208K, Analytics Suite $163K, and Dashboard Pro $111.5K."},
    {"instruction": "When did revenue peak in 2025?", "response": "Monthly revenue peaked in April 2025 at $64,000, driven by 2 Enterprise Plan renewals."},
    {"instruction": "What are the 2026 plans?", "response": "Key initiatives for 2026 include expanding the east region sales team, launching the Salesforce connector for Analytics Suite, and introducing a mid-tier pricing plan between Dashboard Pro and Enterprise Plan."},

    # Salesforce-related
    {"instruction": "What Salesforce issues are customers facing?", "response": "Two key accounts in the south region churned citing integration issues with their existing Salesforce workflows. A Salesforce connector for Analytics Suite is planned for Q1 2026 to address this."},
    {"instruction": "When is the Salesforce connector launching?", "response": "The Salesforce connector for Analytics Suite is planned for Q1 2026."},
]

# ============================================================
# STEP 2: Generate more training data using Claude
# ============================================================

def generate_more_examples(existing_data, num_new=30):
    examples = "\n".join([f"Q: {d['instruction']}\nA: {d['response']}" for d in existing_data[:10]])

    prompt = f"""Based on these example Q&A pairs about a B2B SaaS company, generate {num_new} NEW and DIFFERENT Q&A pairs.

Cover these topics: products, pricing, regions, customers, metrics, plans, competitors, onboarding, churn, upselling.

Use the same tone, detail level, and company context as the examples.

EXAMPLES:
{examples}

Return ONLY a JSON array. No markdown. Each item has "instruction" and "response" keys.
Format: [{{"instruction": "...", "response": "..."}}, ...]"""

    response = claude.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=4000,
        temperature=0.7,
        messages=[{"role": "user", "content": prompt}]
    )

    text = response.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    if text.endswith("```"):
        text = text[:-3].strip()

    try:
        return json.loads(text)
    except:
        print("Failed to parse generated examples")
        return []

print("=" * 60)
print("  PREPARING FINE-TUNING DATASET")
print("=" * 60)

print(f"\n  Manual examples: {len(manual_dataset)}")

# Generate additional examples
print("  Generating more examples with Claude...")
generated = generate_more_examples(manual_dataset, num_new=30)
print(f"  Generated examples: {len(generated)}")

# Combine
full_dataset = manual_dataset + generated
print(f"  Total dataset size: {len(full_dataset)}")

# ============================================================
# STEP 3: Split into train/test
# ============================================================

import random
random.seed(42)
random.shuffle(full_dataset)

split_point = int(len(full_dataset) * 0.85)
train_data = full_dataset[:split_point]
test_data = full_dataset[split_point:]

print(f"\n  Train set: {len(train_data)} examples")
print(f"  Test set:  {len(test_data)} examples")

# ============================================================
# STEP 4: Save in different formats
# ============================================================

# JSONL format (most common for fine-tuning)
with open("train.jsonl", "w") as f:
    for item in train_data:
        f.write(json.dumps(item) + "\n")

with open("test.jsonl", "w") as f:
    for item in test_data:
        f.write(json.dumps(item) + "\n")

# Alpaca format (instruction-input-output, used by many fine-tuning tools)
alpaca_data = []
for item in train_data:
    alpaca_data.append({
        "instruction": item["instruction"],
        "input": "",
        "output": item["response"]
    })

with open("train_alpaca.json", "w") as f:
    json.dump(alpaca_data, f, indent=2)

print(f"\n  Files saved:")
print(f"    train.jsonl       ({len(train_data)} examples)")
print(f"    test.jsonl        ({len(test_data)} examples)")
print(f"    train_alpaca.json ({len(alpaca_data)} examples, Alpaca format)")

# ============================================================
# STEP 5: Preview the dataset
# ============================================================

print(f"\n{'='*60}")
print("  DATASET PREVIEW")
print(f"{'='*60}")

print("\n  First 5 training examples:")
for i, item in enumerate(train_data[:5], 1):
    print(f"\n  {i}. Q: {item['instruction']}")
    print(f"     A: {item['response'][:80]}...")

print(f"\n  Test examples (for evaluation after fine-tuning):")
for i, item in enumerate(test_data[:3], 1):
    print(f"\n  {i}. Q: {item['instruction']}")
    print(f"     A: {item['response'][:80]}...")

# ============================================================
# STEP 6: Dataset quality stats
# ============================================================

print(f"\n{'='*60}")
print("  DATASET QUALITY STATS")
print(f"{'='*60}")

avg_q_len = sum(len(d["instruction"]) for d in full_dataset) / len(full_dataset)
avg_a_len = sum(len(d["response"]) for d in full_dataset) / len(full_dataset)

print(f"  Total examples:          {len(full_dataset)}")
print(f"  Avg question length:     {round(avg_q_len)} chars")
print(f"  Avg answer length:       {round(avg_a_len)} chars")
print(f"  Shortest answer:         {min(len(d['response']) for d in full_dataset)} chars")
print(f"  Longest answer:          {max(len(d['response']) for d in full_dataset)} chars")