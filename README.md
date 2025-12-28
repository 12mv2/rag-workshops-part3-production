# Workshop 3: Production Pipeline

**Learn:** Scale RAG with vector databases for instant similarity search

## Quick Start
```bash
git clone https://github.com/12mv2/rag-workshops-part3-production.git
cd rag-workshops-part3-production
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Create .env with your keys (from Workshop 1 Pinecone setup)
echo "PINECONE_API_KEY=your-key" > .env
echo "PINECONE_INDEX=runners-index" >> .env
echo "OPENAI_API_KEY=your-openai-key" >> .env

# Upload data to Pinecone
python upload.py

# Query the system
python demo.py
```

## What You'll Do

| **Phase** | **Action** | **Outcome** | **Time** |
|-----------|------------|-------------|----------|
| **Setup** | Run commands above (Pinecone already configured from W1) | Data uploaded to vector DB | 3 min |
| **See It** | Input gait (185, 0.2, 6.5) → "You run like Kipchoge!" | Semantic search found nearest neighbors | 2 min |
| **Concept** | **Normalize features → Cosine similarity finds neighbors** | Equal importance for all metrics | 5 min |
| **Your Turn** | Try 3 inputs: 1) Kipchoge-like 2) Bolt-like 3) Kangaroo-like | Feel vector search at scale | 10 min |
| **Learned** | Vector DBs = instant similarity at any scale | ✅ Production RAG | 2 min |

## The Core Insight

**Workshop 1:** Vectors cluster similar data  
**Workshop 2:** RAG grounds LLM responses  
**Workshop 3:** Vector databases make it scalable

Instead of comparing your query to every vector (slow), Pinecone indexes them for **instant nearest-neighbor search**. This is how production RAG systems handle millions of documents.

## The Pipeline

```
Your gait metrics (cadence, heel strike, oscillation)
         ↓
[Normalize: 0-1 scale] ← CRITICAL: equal feature weight
         ↓
[Unit vector: cosine-ready] ← Best practice for efficiency
         ↓
[Pinecone query: top 5 matches] ← Milliseconds at any scale
         ↓
[LLM analyzes with context] ← Grounded response
```

## Why Normalization Matters

Without normalization, big numbers dominate:
```python
# BAD: Cadence (180) crushes heel strike (0.2)
[180, 0.2, 6.0] → cadence controls everything

# GOOD: All features equal weight
[0.65, 0.2, 0.12] → balanced similarity
```

<details>
<summary><strong>📚 Understanding the Data</strong></summary>

**13 entities uploaded to Pinecone:**

| Name | Cadence | Heel Strike | Vert. Osc |
|------|---------|-------------|-----------|
| Eliud Kipchoge | 185 | 0.2 | 6.2 |
| Usain Bolt | 260 | 0.3 | 4.8 |
| Cheetah | 250 | 0.1 | 12.5 |
| Kangaroo | 70 | 0.0 | 35.0 |
| ... | ... | ... | ... |

Each becomes a 3D unit vector in Pinecone.

</details>

<details>
<summary><strong>🔧 Advanced: Using Gemini Instead of OpenAI</strong></summary>

Update `.env`:
```env
GEMINI_API_KEY=your-gemini-key
# OPENAI_API_KEY=comment-this-out
```

The code auto-detects which key is present.

</details>

<details>
<summary><strong>🎯 No Pinecone Account Yet?</strong></summary>

Should have done this in Workshop 1's optional addon! But if not:

1. Sign up: [pinecone.io](https://pinecone.io) (free tier)
2. Create index:
   - Name: `runners-index`
   - Dimensions: `3`
   - Metric: `cosine`
3. Copy API key to `.env`

</details>

<details>
<summary><strong>💡 Want to modify and save your changes?</strong></summary>

Fork it first:
1. Click "Fork" on GitHub
2. Clone YOUR fork: `git clone https://github.com/YOUR_USERNAME/rag-workshops-part3-production.git`
3. Experiment freely

</details>

---

**Previous:** [Workshop 2 - Retrieval Basics](https://github.com/12mv2/rag-workshop-2-retrieval-basics)

**Next Steps:** Build your own RAG system with:
- Your own data
- Different vector databases (Weaviate, Qdrant, Chroma)
- Text documents (not just structured data)

---

**Questions?** [GitHub Discussions](https://github.com/12mv2/rag-workshops-part3-production/discussions)
