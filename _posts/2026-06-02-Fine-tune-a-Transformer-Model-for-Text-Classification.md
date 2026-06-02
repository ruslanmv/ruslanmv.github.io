---
title: "Fine-tune a Transformer Model for Text Classification (Python Tutorial)"
excerpt: "Train, evaluate, and save your own text classifier with Hugging Face Transformers."
description: "Step-by-step Python tutorial to fine-tune a transformer (DistilBERT) for text classification with the Hugging Face Trainer — data prep, training, evaluation (accuracy/F1), and inference."
last_modified_at: 2026-06-02
header:
  image: "/assets/images/posts/2026-06-02-Fine-tune-a-Transformer-Model-for-Text-Classification/hero.jpg"
  teaser: "/assets/images/posts/2026-06-02-Fine-tune-a-Transformer-Model-for-Text-Classification/hero.jpg"
  caption: "Photo via Unsplash"
categories:
  - AI
  - NLP
tags:
  - Hugging Face
  - Transformers
  - Fine-tuning
  - NLP
  - Python
toc: true
toc_label: "Contents"
---

Pretrained transformers already "understand" language — **fine-tuning** teaches them *your* labels with a short training run on a modest dataset. In this tutorial you'll fine-tune **DistilBERT** for text classification with the Hugging Face `Trainer`, evaluate it with accuracy and F1, and save a model you can reuse anywhere.

## The pipeline at a glance

![Fine-tuning pipeline: data, tokenizer, pretrained model, trainer, evaluation]({{ '/assets/images/posts/2026-06-02-Fine-tune-a-Transformer-Model-for-Text-Classification/diagram.svg' | relative_url }})

## What you'll build

A sentiment classifier trained on movie reviews — but the exact same code fine-tunes *any* labeled text dataset (support tickets, emails, intents).

## Prerequisites

- Python 3.10+
- A GPU is helpful but not required for this small example

## 1. Install

```bash
python -m pip install -U "transformers[torch]" datasets evaluate scikit-learn
```

## 2. Load and tokenize the data

```python
from datasets import load_dataset
from transformers import AutoTokenizer

# A small slice keeps the demo fast; drop the slicing for the full run.
ds = load_dataset("imdb")
ds["train"] = ds["train"].shuffle(seed=42).select(range(2000))
ds["test"]  = ds["test"].shuffle(seed=42).select(range(1000))

model_id = "distilbert-base-uncased"
tok = AutoTokenizer.from_pretrained(model_id)

def tokenize(batch):
    return tok(batch["text"], truncation=True, max_length=256)

ds = ds.map(tokenize, batched=True)
```

## 3. Define the model and metrics

```python
import numpy as np, evaluate
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)

acc = evaluate.load("accuracy")
f1  = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": acc.compute(predictions=preds, references=labels)["accuracy"],
            "f1": f1.compute(predictions=preds, references=labels, average="macro")["f1"]}
```

## 4. Train

```python
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding

args = TrainingArguments(
    output_dir="distilbert-imdb",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_steps=50,
)

trainer = Trainer(
    model=model, args=args,
    train_dataset=ds["train"], eval_dataset=ds["test"],
    tokenizer=tok, data_collator=DataCollatorWithPadding(tok),
    compute_metrics=compute_metrics,
)

trainer.train()
print(trainer.evaluate())
```

**Expected output (abridged):**

```text
{'eval_accuracy': 0.90, 'eval_f1': 0.90, 'epoch': 2.0}
```

Even on 2,000 examples DistilBERT reaches ~90% — the power of starting from a pretrained model.

## 5. Save and run inference

```python
trainer.save_model("distilbert-imdb")          # weights + config + tokenizer
from transformers import pipeline

clf = pipeline("text-classification", model="distilbert-imdb", tokenizer=tok)
print(clf("An absolute masterpiece — I was hooked the whole time."))
# [{'label': 'LABEL_1', 'score': 0.98}]   # LABEL_1 = positive
```

To get human-readable labels, set `id2label`/`label2id` on the model config before training (e.g. `{0: "negative", 1: "positive"}`).

## Common errors

- **CUDA out of memory** — lower `per_device_train_batch_size`, reduce `max_length`, or use gradient accumulation.
- **`eval_strategy` not recognized** — older Transformers use `evaluation_strategy`; upgrade or rename.
- **Loss not decreasing** — check your labels are 0..N-1 integers and `num_labels` matches.

## FAQ

**Which base model should I pick?**
`distilbert-base-uncased` is a great default (fast, small). For higher accuracy try `roberta-base`; for multilingual data use `xlm-roberta-base`.

**How much data do I need?**
A few hundred well-labeled examples per class already work; more data and balanced classes help most.

## Related tutorials

- [Hugging Face & Local LLMs guide](/hugging-face/)
- [Use Hugging Face pipelines for rapid prototyping](/blog/Use-Hugging-Face-Pipelines-for-Rapid-Prototyping)
- [Build embeddings and semantic search with Sentence Transformers](/blog/Build-Embeddings-and-Semantic-Search-with-Sentence-Transformers)
- [Deploy a Hugging Face model with FastAPI](/blog/Deploy-a-Hugging-Face-Model-with-FastAPI)
