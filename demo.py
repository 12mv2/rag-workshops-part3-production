import os
import json
import math
import openai
from dotenv import load_dotenv
from pinecone import Pinecone

# === Load environment variables ===
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "runners-index")

# === Init Pinecone ===
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

# === Setup LLM ===
use_openai = bool(OPENAI_API_KEY)
if use_openai:
    openai.api_key = OPENAI_API_KEY
else:
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")

# === Embedding Functions ===
# Step 1: Feature normalization ensures all metrics contribute equally
def normalize_feature(value, min_val, max_val):
    """Scale feature to 0-1 range"""
    return (value - min_val) / (max_val - min_val)

# Step 2: Vector normalization creates unit vector for cosine similarity
def normalize_vector(vector):
    """Convert to unit vector (magnitude=1)"""
    mag = math.sqrt(sum(x ** 2 for x in vector))
    return [x / mag for x in vector] if mag else vector

def embed(cadence, heel_ratio, vertical_osc):
    """Transform gait metrics into normalized 3D vector"""
    vec = [
        normalize_feature(cadence, 50, 250),
        heel_ratio,  # Already 0-1 scale
        normalize_feature(vertical_osc, 6, 20)
    ]
    return normalize_vector(vec)

# === User Input ===
print("\n🏃 Enter gait metrics to find similar runners/animals:\n")
cadence = float(input("  Cadence (steps/min, e.g., 185): "))
heel = float(input("  Heel Strike Ratio (0=toe, 1=heel, e.g., 0.2): "))
vert = float(input("  Vertical Oscillation (cm, e.g., 6.5): "))

query_vec = embed(cadence, heel, vert)
print(f"\n🔎 Query vector: [{query_vec[0]:.3f}, {query_vec[1]:.3f}, {query_vec[2]:.3f}]")

# === Query Pinecone ===
print("\n🔍 Searching Pinecone for similar gaits...")
results = index.query(vector=query_vec, top_k=5, include_metadata=True)
matches = results.get("matches", [])

if not matches:
    print("\n❌ No similar results found.")
    exit()

# === Display Results ===
print("\n📊 Top 5 matches:\n")
for i, match in enumerate(matches, 1):
    name = match['id']
    score = match['score']
    print(f"  {i}. {name:25s} (similarity: {score:.3f})")

# === Build context for LLM ===
context = "Top 5 most similar gaits:\n"
for i, match in enumerate(matches, 1):
    context += f"{i}. {match['id']} (similarity: {match['score']:.3f})\n"

# === LLM Prompt ===
prompt = f"""You are a biomechanics expert. Analyze this gait pattern based on the similar matches found.

Input gait:
- Cadence: {cadence} steps/min
- Heel strike ratio: {heel} (0=toe, 1=heel)
- Vertical oscillation: {vert} cm

{context}

Provide a brief analysis explaining what this gait pattern suggests."""

print("\n🤖 Generating expert analysis...\n")

if use_openai:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a biomechanics expert."},
            {"role": "user", "content": prompt}
        ]
    )
    print(response.choices[0].message.content)
else:
    response = model.generate_content(prompt)
    print(response.text)

print("\n✅ Done!")
 