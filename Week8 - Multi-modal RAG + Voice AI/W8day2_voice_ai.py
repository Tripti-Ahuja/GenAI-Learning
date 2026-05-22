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
# STEP 1: Speech-to-Text (STT) — simulated
# In production: Whisper or Deepgram converts audio to text
# ============================================================

def speech_to_text(audio_input):
    """
    In production, this would be:
    
    import whisper
    model = whisper.load_model("tiny")
    result = model.transcribe("question.wav")
    return result["text"]
    
    For learning, we simulate by just returning the text
    as if Whisper already transcribed it.
    """
    print(f"  🎤 [STT] Audio received → Transcribed: \"{audio_input}\"")
    return audio_input

# ============================================================
# STEP 2: Text-to-Speech (TTS) — using Claude to format
# In production: ElevenLabs or Coqui converts text to audio
# ============================================================

def text_to_speech(text):
    """
    In production, this would be:
    
    import requests
    response = requests.post(
        "https://api.elevenlabs.io/v1/text-to-speech/voice_id",
        headers={"xi-api-key": "your_key"},
        json={"text": text, "model_id": "eleven_monolingual_v1"}
    )
    with open("response.mp3", "wb") as f:
        f.write(response.content)
    
    For learning, we simulate by printing what would be spoken.
    """
    # Clean up the text for speech (remove markdown, citations)
    clean_text = text.replace("**", "").replace("[", "").replace("]", "")
    clean_text = clean_text.replace("Source:", "from")
    print(f"  🔊 [TTS] Speaking: \"{clean_text[:100]}...\"")
    return clean_text

# ============================================================
# STEP 3: RAG pipeline (same as before)
# ============================================================

documents = [
    {"text": "Q4 2025 revenue was $2.3 million, up 15% year over year", "source": "Q4 Report"},
    {"text": "North region generated $920K in Q4 2025, leading all regions", "source": "Q4 Report"},
    {"text": "South region had weakest performance at $310K, down 8% from Q3", "source": "Q4 Report"},
    {"text": "Enterprise Plan is top product at $208K total revenue", "source": "Q4 Report"},
    {"text": "Customer churn rate dropped to 3.2% in Q4, down from 5.1% in Q3", "source": "Q4 Report"},
    {"text": "Dashboard Pro leads in order volume with 9 orders. 40% upgrade to Analytics Suite", "source": "Q4 Report"},
    {"text": "Amit Patel is highest spending customer at $90,000", "source": "Q4 Report"},
    {"text": "Management targets 25 new customer acquisitions in 2026", "source": "Q4 Report"},
]

doc_embeddings = embed_model.encode([d["text"] for d in documents])

def retrieve(query, top_k=3):
    query_vec = embed_model.encode([query])
    scores = cosine_similarity(query_vec, doc_embeddings)[0]
    top_idx = scores.argsort()[-top_k:][::-1]
    return [(idx, round(float(scores[idx]), 3)) for idx in top_idx]

def generate_answer(query, results):
    context = "\n".join([f"[{documents[idx]['source']}] {documents[idx]['text']}" for idx, _ in results])

    prompt = f"""Answer the question concisely based on the sources. 
Keep your answer short (1-2 sentences) since it will be read aloud.
Cite sources naturally like "according to the Q4 report."

SOURCES:
{context}

QUESTION: {query}

ANSWER:"""

    response = claude.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=150,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text

# ============================================================
# STEP 4: Full voice pipeline
# ============================================================

def voice_ask(spoken_input):
    print(f"\n{'='*55}")
    print(f"  VOICE PIPELINE")
    print(f"{'='*55}")

    # Step 1: Speech to Text
    transcribed = speech_to_text(spoken_input)

    # Step 2: Retrieve
    results = retrieve(transcribed, top_k=3)
    print(f"  🔍 [RAG] Retrieved {len(results)} documents")
    for idx, score in results:
        print(f"       [{score}] {documents[idx]['text'][:50]}...")

    # Step 3: Generate
    answer = generate_answer(transcribed, results)
    print(f"  🤖 [Answer] {answer}")

    # Step 4: Text to Speech
    spoken_answer = text_to_speech(answer)

    return answer

# ============================================================
# TEST — Simulating voice conversations
# ============================================================

print("=" * 55)
print("  VOICE-ENABLED RAG SYSTEM")
print("  (Simulated: STT and TTS shown as text)")
print("=" * 55)

# These simulate what a user would SAY out loud
voice_ask("What was our revenue last quarter?")
voice_ask("Which region is doing the worst?")
voice_ask("Tell me about our best customer")
voice_ask("What are our plans for next year?")

# ============================================================
# INTERACTIVE MODE
# ============================================================

print(f"\n{'='*55}")
print("  INTERACTIVE VOICE MODE")
print("  (Type what you'd speak, or 'quit')")
print(f"{'='*55}")

while True:
    spoken = input("\n🎤 You say: ").strip()
    if spoken.lower() == "quit":
        print("Goodbye!")
        break
    if spoken:
        voice_ask(spoken)