import requests
import json
import time

# ============================================================
# HELPER: Ask any local model via Ollama
# ============================================================

def ask_ollama(prompt, model):
    start = time.time()
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=120
        )
        elapsed = round(time.time() - start, 1)
        result = response.json()
        return result["response"], elapsed
    except Exception as e:
        return f"Error: {str(e)}", 0

def get_model_info(model):
    try:
        response = requests.post(
            "http://localhost:11434/api/show",
            json={"name": model}
        )
        info = response.json()
        size = info.get("size", 0)
        size_gb = round(size / (1024**3), 2) if size else "unknown"
        return {"size_gb": size_gb, "parameters": info.get("details", {}).get("parameter_size", "unknown"), "quantization": info.get("details", {}).get("quantization_level", "unknown")}
    except:
        return {"size_gb": "unknown", "parameters": "unknown", "quantization": "unknown"}

# ============================================================
# STEP 1: Compare model info
# ============================================================

models = ["llama3.2:3b", "mistral"]

print("=" * 60)
print("  MODEL COMPARISON: Llama 3.2 (3B) vs Mistral (7B)")
print("=" * 60)

print(f"\n  Model Info:")
for model in models:
    info = get_model_info(model)
    print(f"    {model}: {info['parameters']} params | {info['size_gb']} GB | Quantization: {info['quantization']}")

# ============================================================
# STEP 2: Quality comparison on same prompts
# ============================================================

test_prompts = [
    {
        "name": "Simple fact",
        "prompt": "What is the capital of Japan? Answer in one sentence."
    },
    {
        "name": "SQL generation",
        "prompt": "Write a SQL query to find the top 3 customers by total spending from tables: customers(id, name) and orders(customer_id, amount). Return ONLY the SQL, no explanation."
    },
    {
        "name": "Reasoning",
        "prompt": "A company had $800K revenue in Q3 and $920K in Q4. What is the percentage growth? Show your calculation."
    },
    {
        "name": "Classification",
        "prompt": "Classify this review as positive, negative, or neutral. Reply with ONLY the classification word.\n\nReview: 'Great tool but the dashboard loads too slowly.'"
    },
    {
        "name": "Summarization",
        "prompt": "Summarize in one sentence: The north region led all regions with $920K in Q4 revenue, representing 15% growth from Q3, largely attributed to the new sales playbook introduced in September and three large Enterprise Plan deals."
    },
]

print(f"\n{'='*60}")
print("  QUALITY + SPEED COMPARISON")
print(f"{'='*60}")

results = []

for test in test_prompts:
    print(f"\n  Task: {test['name']}")
    print(f"  Prompt: \"{test['prompt'][:70]}...\"")

    for model in models:
        answer, elapsed = ask_ollama(test["prompt"], model)
        clean_answer = answer.strip()[:150]
        print(f"    {model:<15} [{elapsed}s]: {clean_answer}...")
        results.append({
            "task": test["name"],
            "model": model,
            "time": elapsed,
            "answer": answer.strip()[:200]
        })

# ============================================================
# STEP 3: Summary scorecard
# ============================================================

print(f"\n{'='*60}")
print("  SPEED SUMMARY")
print(f"{'='*60}")

for model in models:
    model_results = [r for r in results if r["model"] == model]
    avg_time = round(sum(r["time"] for r in model_results) / len(model_results), 1)
    info = get_model_info(model)
    print(f"\n  {model}:")
    print(f"    Size: {info['size_gb']} GB | Quantization: {info['quantization']}")
    print(f"    Avg response time: {avg_time}s")

print(f"\n{'='*60}")
print("  QUANTIZATION EXPLAINED")
print(f"{'='*60}")
print("""
  What is quantization?
  Models store millions of numbers (weights). Precision levels:

  FP16 (16-bit):  Full precision  → 7B model = ~14 GB RAM
  Q8   (8-bit):   Half size       → 7B model = ~7 GB RAM
  Q4   (4-bit):   Quarter size    → 7B model = ~4 GB RAM

  Ollama models are typically Q4 quantized by default.
  That's why Mistral 7B fits in ~4GB instead of 14GB.

  Quality loss from Q4 is minimal for most tasks (<5%).
  Speed improves because smaller files load faster.

  RAM estimation formula:
    Parameters × Bits per param / 8 = Size in bytes
    7B × 4 bits / 8 = 3.5 GB (roughly what you see)
""")