import requests
import json
import time

# ============================================================
# Ollama runs a local API at http://localhost:11434
# Same pattern as Claude API — send request, get response
# But free, local, no internet needed
# ============================================================

def ask_ollama(prompt, model="llama3.2:3b"):
    start = time.time()
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        }
    )
    elapsed = round(time.time() - start, 1)
    result = response.json()
    return result["response"], elapsed

# ============================================================
# TEST 1: Basic conversation
# ============================================================

print("=" * 60)
print("  LOCAL LLM via Ollama (Llama 3.2 3B)")
print("  Running on YOUR laptop — no API, no cost")
print("=" * 60)

questions = [
    "What is a LEFT JOIN in SQL? Answer in 2 sentences.",
    "Explain what an API is to a 10 year old. Keep it short.",
    "What is the capital of France?",
    "Write a SQL query to find the top 5 customers by total spending.",
]

for q in questions:
    print(f"\nQ: {q}")
    answer, elapsed = ask_ollama(q)
    print(f"A: {answer[:200]}")
    print(f"   [{elapsed}s]")

# ============================================================
# TEST 2: Compare response quality vs Claude
# ============================================================

import anthropic
from dotenv import load_dotenv
load_dotenv()
claude = anthropic.Anthropic()

print(f"\n{'='*60}")
print("  LOCAL (Llama 3.2 3B) vs API (Claude Haiku)")
print(f"{'='*60}")

test_prompts = [
    "What is the difference between a data analyst and a data engineer? Answer in 2 sentences.",
    "Write a SQL query to find monthly revenue for 2025 from an orders table with columns: id, amount, order_date.",
]

for prompt in test_prompts:
    print(f"\nQ: {prompt}")

    # Local model
    local_answer, local_time = ask_ollama(prompt)

    # Claude API
    start = time.time()
    claude_response = claude.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=150,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    claude_time = round(time.time() - start, 1)
    claude_answer = claude_response.content[0].text

    print(f"\n  🖥️ Local ({local_time}s): {local_answer[:150]}...")
    print(f"\n  ☁️ Claude ({claude_time}s): {claude_answer[:150]}...")
    print(f"\n  Cost: Local = $0.00 | Claude = ~$0.001")