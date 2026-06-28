#!/usr/bin/env python3
"""A tiny, reusable OllaBridge Cloud client.

OllaBridge Cloud is an OpenAI-compatible *router*: one endpoint that picks the
best model behind a logical alias (free-best, fable-5, local-private, ...) with
automatic failover. We talk to it with plain `requests` — no SDK, no surprises,
works behind any firewall.

Configuration (environment variables):
    OLLABRIDGE_URL    e.g. https://app.ollabridge.com   (NO trailing /v1)
    OLLABRIDGE_TOKEN  the device token from pair.py
"""
from __future__ import annotations

import os

import requests

BASE = os.environ.get("OLLABRIDGE_URL", "https://app.ollabridge.com").rstrip("/")
TOKEN = os.environ.get("OLLABRIDGE_TOKEN", "not-needed")


def _headers() -> dict:
    return {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}


def list_models() -> list[str]:
    """Return the model aliases the router can reach right now."""
    r = requests.get(f"{BASE}/v1/models", headers=_headers(), timeout=30)
    r.raise_for_status()
    return [m["id"] for m in r.json().get("data", [])]


def chat(prompt: str, *, model: str = "free-best", system: str | None = None,
         temperature: float = 0.3) -> str:
    """One-shot chat completion through the router. Returns the reply text."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    r = requests.post(
        f"{BASE}/v1/chat/completions",
        headers=_headers(),
        json={"model": model, "messages": messages, "temperature": temperature, "stream": False},
        timeout=120,
    )
    r.raise_for_status()
    return (r.json()["choices"][0]["message"]["content"] or "").strip()


if __name__ == "__main__":
    print("Models reachable:")
    for mid in list_models()[:12]:
        print("  -", mid)
    print("\nfree-best says:", chat("Reply with exactly: ROUTER OK", temperature=0))
