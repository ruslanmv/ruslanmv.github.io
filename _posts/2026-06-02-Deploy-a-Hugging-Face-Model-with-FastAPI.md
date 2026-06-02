---
title: "Deploy a Hugging Face Model with FastAPI (Production-Ready Tutorial)"
excerpt: "Wrap a Hugging Face model in a fast, typed REST API with FastAPI and Uvicorn."
description: "Python tutorial to deploy a Hugging Face model with FastAPI — build a typed /predict endpoint, load the pipeline once, add a health check, and containerize it for production."
last_modified_at: 2026-06-02
header:
  image: "/assets/images/posts/2026-06-02-Deploy-a-Hugging-Face-Model-with-FastAPI/hero.jpg"
  teaser: "/assets/images/posts/2026-06-02-Deploy-a-Hugging-Face-Model-with-FastAPI/hero.jpg"
  caption: "Photo via Unsplash"
categories:
  - AI
  - MLOps
tags:
  - FastAPI
  - Hugging Face
  - Deployment
  - API
  - Python
toc: true
toc_label: "Contents"
---

A model in a notebook helps one person; a model behind an **API** helps every app in your company. **FastAPI** is the modern, fast, type-checked way to serve a Hugging Face model over HTTP. This tutorial builds a production-minded inference service — typed requests, a single shared model, a health check — and shows how to containerize it.

## Architecture

![FastAPI serving a Hugging Face model: client posts text, FastAPI runs the pipeline, returns JSON]({{ '/assets/images/posts/2026-06-02-Deploy-a-Hugging-Face-Model-with-FastAPI/diagram.svg' | relative_url }})

## Prerequisites

- Python 3.10+

## 1. Install

```bash
python -m pip install -U fastapi "uvicorn[standard]" "transformers[torch]"
```

## 2. The service (`app.py`)

Load the model **once** at startup (not per request), validate input with Pydantic, and expose a `/health` endpoint so orchestrators know the service is alive.

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

ml = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # load the model once when the server starts
    ml["clf"] = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    yield
    ml.clear()

app = FastAPI(title="Sentiment API", lifespan=lifespan)

class TextIn(BaseModel):
    text: str

class PredictionOut(BaseModel):
    label: str
    score: float

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionOut)
def predict(body: TextIn):
    result = ml["clf"](body.text)[0]
    return {"label": result["label"], "score": round(result["score"], 4)}
```

## 3. Run it

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Then call it — and open `http://localhost:8000/docs` for the auto-generated Swagger UI.

```bash
curl -s -X POST localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"This tutorial is fantastic!"}'
```

**Expected output:**

```json
{"label":"POSITIVE","score":0.9998}
```

## 4. Containerize for production

Pin dependencies in `requirements.txt`, then build a small image (the multi-stage `Dockerfile` and best practices are covered in detail in [Deploy AI agents to production with Docker](/blog/Deploy-AI-Agents-to-Production-with-Docker)).

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
RUN useradd -m appuser && chown -R appuser /app
USER appuser
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t sentiment-api:1.0 .
docker run --rm -p 8000:8000 sentiment-api:1.0
```

## Production tips

- **Load once** (the `lifespan` pattern) so each request reuses the model in memory.
- **Batch** requests where possible — pipelines accept lists for higher throughput.
- **Add timeouts** around inference and return `503` if the model isn't ready.
- **Cache the model** in the image or a mounted volume to avoid re-downloading on every cold start.
- **Scale** with multiple Uvicorn workers (`--workers`) or by running more containers behind a load balancer.

## Common errors

- **Model re-downloads on every start** — bake it into the image or mount a `HF_HOME` cache volume.
- **High latency on first request** — that's the model warming up; call `/predict` once at startup.
- **422 Unprocessable Entity** — your JSON body doesn't match the Pydantic schema (`{"text": "..."}`).

## FAQ

**FastAPI vs Flask?**
FastAPI is async, faster, and gives you typed validation plus free OpenAPI docs — a better fit for ML services.

**Where do I deploy this?**
Any container platform: Google Cloud Run, AWS ECS/Fargate, or Kubernetes. Cloud Run can run this exact image with autoscaling.

## Related tutorials

- [Hugging Face & Local LLMs guide](/hugging-face/)
- [Deploy AI agents to production with Docker](/blog/Deploy-AI-Agents-to-Production-with-Docker)
- [Use Hugging Face pipelines for rapid prototyping](/blog/Use-Hugging-Face-Pipelines-for-Rapid-Prototyping)
- [Fine-tune a transformer model for text classification](/blog/Fine-tune-a-Transformer-Model-for-Text-Classification)
