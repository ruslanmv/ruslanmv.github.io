---
title: "Build Embeddings and Semantic Search with Sentence Transformers"
excerpt: "Turn text into vectors and build a fast semantic search engine with FAISS."
description: "Python tutorial to build embeddings and semantic search with Sentence Transformers and FAISS — encode a corpus, index vectors, and run cosine nearest-neighbour queries, with runnable code."
last_modified_at: 2026-06-02
header:
  image: "/assets/images/posts/2026-06-02-Build-Embeddings-and-Semantic-Search-with-Sentence-Transformers/hero.jpg"
  teaser: "/assets/images/posts/2026-06-02-Build-Embeddings-and-Semantic-Search-with-Sentence-Transformers/hero.jpg"
  caption: "Photo via Unsplash"
categories:
  - AI
  - NLP
tags:
  - Embeddings
  - Sentence Transformers
  - FAISS
  - Semantic Search
  - Python
toc: true
toc_label: "Contents"
---

Keyword search matches *letters*; **semantic search** matches *meaning*. The trick is to turn each piece of text into a **vector** (embedding) so that similar meanings land close together — then a query finds its nearest neighbours. This tutorial builds a working semantic search engine with **Sentence Transformers** and **FAISS** in a few lines of Python.

## How it works

![Semantic search: documents encoded to vectors, indexed in FAISS, queried by nearest neighbours]({{ '/assets/images/posts/2026-06-02-Build-Embeddings-and-Semantic-Search-with-Sentence-Transformers/diagram.svg' | relative_url }})

## Prerequisites

- Python 3.10+ (no GPU needed for this example)

## 1. Install

```bash
python -m pip install -U sentence-transformers faiss-cpu
```

## 2. Embed a corpus

`all-MiniLM-L6-v2` is small, fast, and produces 384-dimensional vectors — perfect for getting started.

```python
from sentence_transformers import SentenceTransformer

docs = [
    "How do I reset my password?",
    "The invoice total looks incorrect.",
    "Where can I download my receipt?",
    "My account is locked after too many attempts.",
    "How do I change my billing address?",
]

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(docs, normalize_embeddings=True)   # shape (5, 384)
print(embeddings.shape)   # (5, 384)
```

Normalizing the vectors means a plain dot-product equals **cosine similarity** — convenient for search.

## 3. Build a FAISS index

```python
import faiss, numpy as np

dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)        # inner product == cosine on normalized vectors
index.add(np.asarray(embeddings, dtype="float32"))
print(index.ntotal)   # 5
```

## 4. Search

```python
def search(query, k=3):
    q = model.encode([query], normalize_embeddings=True).astype("float32")
    scores, ids = index.search(q, k)
    return [(docs[i], round(float(s), 3)) for s, i in zip(scores[0], ids[0])]

for hit in search("I forgot my login credentials"):
    print(hit)
```

**Expected output:**

```text
('My account is locked after too many attempts.', 0.62)
('How do I reset my password?', 0.61)
('Where can I download my receipt?', 0.27)
```

Notice the top hits never share the words "forgot" or "login" — they match on **meaning**. That's the whole point of embeddings, and it's exactly what powers **retrieval-augmented generation (RAG)**.

## 5. Scale up

- Swap `IndexFlatIP` for `IndexIVFFlat` or `IndexHNSWFlat` when you have millions of vectors.
- Persist with `faiss.write_index(index, "corpus.faiss")` and reload with `faiss.read_index(...)`.
- For a managed vector database, the same vectors drop straight into Milvus, Pinecone, or pgvector.

## Common errors

- **`faiss` import fails** — install `faiss-cpu` (or `faiss-gpu` on CUDA machines).
- **Poor results** — make sure you encode the query with the **same** model, and normalize both sides.
- **Slow first run** — the model downloads once, then is cached locally.

## FAQ

**Which embedding model should I use?**
`all-MiniLM-L6-v2` for speed; `all-mpnet-base-v2` for higher quality; `multilingual-e5-base` for many languages.

**Is this the same as RAG?**
Retrieval is the first half of RAG — you retrieve relevant chunks by embedding similarity, then feed them to an LLM to answer.

## Related tutorials

- [Hugging Face & Local LLMs guide](/hugging-face/)
- [watsonx Assistant with Milvus as a vector database](/blog/WatsonX-Assistant-with-Milvus-as-Vector-Database)
- [Fine-tune a transformer model for text classification](/blog/Fine-tune-a-Transformer-Model-for-Text-Classification)
- [Deploy a Hugging Face model with FastAPI](/blog/Deploy-a-Hugging-Face-Model-with-FastAPI)
