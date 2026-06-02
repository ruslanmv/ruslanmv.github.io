---
title: "Hugging Face & Local LLM Tutorials"
layout: single
permalink: /hugging-face/
sitemap: true
author_profile: false
toc: true
toc_label: "On this page"
toc_icon: "smile"
last_modified_at: 2026-06-02
excerpt: "A curated hub of Hugging Face and local LLM tutorials — custom cache folders, GGUF conversion, and running FLAN-T5 / GPT models locally with Python."
description: "Hugging Face and local LLM tutorials: save models to a custom cache, convert models to GGUF in Google Colab, and run FLAN-T5 and GPT large language models locally with Python."
header:
  overlay_filter: "0.6"
  overlay_image: /assets/images/header-home.webp
---

Practical guides for working with **Hugging Face Transformers** and running **large language models locally**.

## Models & storage

- [How to download Hugging Face models to a custom cache folder (Python)](/blog/How-to-download-and-save-HuggingFace-models-to-custom-path) — `cache_dir`, `HF_HOME`, `TRANSFORMERS_CACHE`, and `save_pretrained`.

## Run LLMs locally

- [Convert a Hugging Face model to GGUF in Google Colab](/blog/How-to-convert-Models-to-GGUF-in-Google-Colab) — quantize and export for llama.cpp / Ollama.
- [Run FLAN-T5 and GPT large language models locally with Gradio](/blog/How-to-run-Large-Language-Model-FLAN-T5-and-GPT-locally) — a local web UI in Python.

## Build models from scratch

- [Build a basic LLM (GPT) from scratch in Python](/blog/How-to-Build-a-basic-LLM-GPT-from-Scratch-in-Python)
- [Text generation from scratch: build a generative AI model in Python](/blog/Text-Generation-from-Scratch)

## Related topics

Continue with [Generative AI](/generative-ai/), [AI Agents](/ai-agents/), and the daily [AI Rankings](/Best-of-the-Best/) of top AI repositories, papers, and packages.

## FAQ

**How do I stop Hugging Face from filling up my default drive?**
Point the cache elsewhere with `cache_dir` / `HF_HOME` — see [downloading models to a custom cache folder](/blog/How-to-download-and-save-HuggingFace-models-to-custom-path).

**How do I run a Hugging Face model offline / locally?**
Convert it to [GGUF in Google Colab](/blog/How-to-convert-Models-to-GGUF-in-Google-Colab), then run it locally, or use a [local Gradio app for FLAN-T5 / GPT](/blog/How-to-run-Large-Language-Model-FLAN-T5-and-GPT-locally).
