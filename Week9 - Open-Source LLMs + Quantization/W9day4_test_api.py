import requests
import json

BASE_URL = "http://localhost:8000"

# Health check
print("Health:", requests.get(f"{BASE_URL}/health").json())

# List documents
docs = requests.get(f"{BASE_URL}/documents").json()
print(f"Documents: {docs['count']}")

# Ask questions
questions = [
    "What was Q4 revenue?",
    "Who is our best customer?",
    "What is our marketing budget?",
]

print("\n" + "=" * 50)
for q in questions:
    response = requests.post(f"{BASE_URL}/ask", json={"question": q}).json()
    print(f"\nQ: {q}")
    print(f"A: {response['answer'][:120]}")
    print(f"   [{response['time_seconds']}s | ${response['cost']} | {response['model']}]")