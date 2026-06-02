---
title: "Use Hugging Face Pipelines for Rapid Prototyping (Python Tutorial)"
excerpt: "Run text, vision, and speech models in one line with Hugging Face pipelines."
description: "Hugging Face pipelines tutorial — prototype text, vision, and speech tasks in one line of Python. Sentiment, zero-shot classification, NER, summarization, and batching, with runnable examples."
last_modified_at: 2026-06-02
header:
  image: "/assets/images/posts/2026-06-02-Use-Hugging-Face-Pipelines-for-Rapid-Prototyping/hero.jpg"
  teaser: "/assets/images/posts/2026-06-02-Use-Hugging-Face-Pipelines-for-Rapid-Prototyping/hero.jpg"
  caption: "Photo via Unsplash"
categories:
  - AI
  - NLP
tags:
  - Hugging Face
  - Pipelines
  - NLP
  - Vision
  - Python
toc: true
toc_label: "Contents"
---

When you want to test an idea *now*, the Hugging Face **`pipeline`** is the fastest path from "what if" to a working result. One function call downloads a sensible default model and runs the whole task — tokenization, inference, and post-processing — so you can prototype text, vision, and speech in minutes.

## What a pipeline does

![A Hugging Face pipeline wraps tokenizer, model, and post-processing behind one call]({{ '/assets/images/posts/2026-06-02-Use-Hugging-Face-Pipelines-for-Rapid-Prototyping/diagram.svg' | relative_url }})

## Prerequisites

- Python 3.10+

## 1. Install

```bash
python -m pip install -U "transformers[torch]"
```

## 2. Sentiment in one line

```python
from transformers import pipeline

clf = pipeline("sentiment-analysis")
print(clf("I love how simple this is!"))
# [{'label': 'POSITIVE', 'score': 0.9999}]
```

No model name, no tokenizer setup — the pipeline picks a good default and handles everything.

## 3. Zero-shot classification (no training!)

Classify text into **your own labels** without fine-tuning anything.

```python
zsc = pipeline("zero-shot-classification")
out = zsc("My invoice total is wrong and I want a refund.",
          candidate_labels=["billing", "technical", "spam"])
print(out["labels"][0], round(out["scores"][0], 3))
# billing 0.972
```

## 4. More tasks, same pattern

```python
ner = pipeline("ner", grouped_entities=True)
print(ner("Ruslan works on AI in Genoa, Italy."))
# [{'entity_group': 'PER', 'word': 'Ruslan', ...}, {'entity_group': 'LOC', 'word': 'Genoa', ...}]

summarizer = pipeline("summarization")
print(summarizer("Long article text ...", max_length=60, min_length=20)[0]["summary_text"])

# Vision works too — just point at an image:
captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
print(captioner("https://images.unsplash.com/photo-1518770660439-4636190af475"))
```

## 5. Choose a specific model and batch for speed

Defaults are great for prototyping; name a model when you need a specific one, and pass a **list** to process many inputs at once.

```python
clf = pipeline("sentiment-analysis",
               model="distilbert-base-uncased-finetuned-sst-2-english",
               device=0)            # device=0 = first GPU, -1 = CPU

results = clf(["Great support!", "Worst experience ever.", "It was fine."])
for r in results:
    print(r["label"], round(r["score"], 3))
```

**Expected output:**

```text
POSITIVE 0.999
NEGATIVE 0.999
POSITIVE 0.715
```

## From prototype to product

Pipelines are perfect for exploration. When you're ready to ship:

- wrap the pipeline in an API — see [Deploy a Hugging Face model with FastAPI](/blog/Deploy-a-Hugging-Face-Model-with-FastAPI);
- if accuracy matters, [fine-tune your own model](/blog/Fine-tune-a-Transformer-Model-for-Text-Classification) and pass it to the pipeline;
- for search/RAG, switch to [embeddings and semantic search](/blog/Build-Embeddings-and-Semantic-Search-with-Sentence-Transformers).

## Common errors

- **Slow first call** — the default model downloads once, then is cached in `~/.cache/huggingface`.
- **Out of memory** — set `device=-1` for CPU, or pick a smaller model.
- **Wrong task name** — see the full list in the Transformers docs (`sentiment-analysis`, `zero-shot-classification`, `ner`, `summarization`, `translation`, `image-to-text`, `automatic-speech-recognition`, …).

## FAQ

**Are pipelines production-ready?**
They're ideal for prototyping and light production. For scale, load the model once and serve it behind FastAPI with batching.

**Can I run pipelines offline?**
Yes — once a model is cached, set `HF_HUB_OFFLINE=1` to run without network access.

## Related tutorials

- [Hugging Face & Local LLMs guide](/hugging-face/)
- [Fine-tune a transformer model for text classification](/blog/Fine-tune-a-Transformer-Model-for-Text-Classification)
- [Build embeddings and semantic search with Sentence Transformers](/blog/Build-Embeddings-and-Semantic-Search-with-Sentence-Transformers)
- [Deploy a Hugging Face model with FastAPI](/blog/Deploy-a-Hugging-Face-Model-with-FastAPI)
