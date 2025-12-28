import os
import json
import math
from dotenv import load_dotenv
from pinecone import Pinecone

# === Load environment variables ===
load_dotenv()
API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX", "runners-index")

# === Init Pinecone client ===
pc = Pinecone(api_key=API_KEY)
index = pc.Index(INDEX_NAME)

# === Embedding helpers ===
# Step 1: Feature normalization (0-1 scale) - CRITICAL for equal feature weight
def normalize_feature(val, min_val, max_val):
    """Scale feature to 0-1 range so all features contribute equally"""
    return (val - min_val) / (max_val - min_val)

# Step 2: Vector normalization (unit length) - Best practice for cosine similarity
def normalize_vector(vec):
    """Convert to unit vector (magnitude=1) for efficient similarity calculations"""
    mag = math.sqrt(sum(v**2 for v in vec))
    return [v / mag for v in vec] if mag else vec

# Combine both steps: feature normalization → vector normalization
def embed_runner(runner):
    """Transform gait metrics into normalized 3D vector for Pinecone"""
    cadence = normalize_feature(runner["cadence"], 50, 250)
    heel = runner["heel_strike"]  # Already 0-1 scale
    vert = normalize_feature(runner["vertical_oscillation"], 6, 20)
    return normalize_vector([cadence, heel, vert])

# === Load runner data ===
with open("data/runners.json", "r") as f:
    runners = json.load(f)

# === Generate a list of vector embeddings from the runner dicts ===
vectors = []
for r in runners:
    vec = embed_runner(r)
    vectors.append((r["name"], vec))
    print(f"✓ {r['name']:20s} → [{vec[0]:.3f}, {vec[1]:.3f}, {vec[2]:.3f}]")

# === Upload the vectors to Pinecone ===
index.upsert(vectors=vectors)
print(f"\n✅ Uploaded {len(vectors)} vectors to Pinecone index '{INDEX_NAME}'")
