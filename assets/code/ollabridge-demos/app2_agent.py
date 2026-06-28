#!/usr/bin/env python3
"""Program 2 — A small *agentic workflow* on the router LLM.

This is the same three-node shape you would wire visually in LangFlow, written
in Python so you can run it and read every step:

    Plan  ->  Research (per sub-question)  ->  Synthesize

Every step is one call to OllaBridge Cloud. Swap the model alias (free-best ->
fable-5 -> local-private) and the workflow is unchanged — that is the whole point
of routing through one gateway.

    python app2_agent.py "How do I start learning to build AI agents?"
"""
from __future__ import annotations

import json
import re
import sys

from ollabridge_client import chat


def plan(question: str, model: str) -> list[str]:
    """Planner agent: break the question into 2-3 focused sub-questions."""
    raw = chat(
        f'Break this into at most 3 focused research sub-questions. '
        f'Return ONLY a JSON array of strings.\n\nQuestion: {question}',
        model=model,
        system="You are a meticulous research planner.",
        temperature=0.2,
    )
    raw = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", raw.strip()).strip()
    try:
        subs = json.loads(raw)
        return [str(s) for s in subs][:3] if isinstance(subs, list) else [question]
    except json.JSONDecodeError:
        return [question]


def research(subq: str, model: str) -> str:
    """Researcher agent: answer one sub-question concisely."""
    return chat(
        f"Answer in 2-3 sentences, concrete and practical:\n{subq}",
        model=model,
        system="You are a concise domain expert.",
        temperature=0.3,
    )


def synthesize(question: str, notes: list[tuple[str, str]], model: str) -> str:
    """Writer agent: merge the notes into one clear answer."""
    bundle = "\n\n".join(f"Q: {q}\nA: {a}" for q, a in notes)
    return chat(
        f"Using these notes, write a clear, friendly final answer to the user's "
        f"original question.\n\nORIGINAL: {question}\n\nNOTES:\n{bundle}",
        model=model,
        system="You are a helpful writer. Be encouraging and concrete.",
        temperature=0.4,
    )


def main() -> int:
    question = sys.argv[1] if len(sys.argv) > 1 else "How do I start learning to build AI agents?"
    model = sys.argv[2] if len(sys.argv) > 2 else "free-best"

    print(f"❓ Question: {question}")
    print(f"🤖 Model   : {model}  (routed by OllaBridge Cloud)\n")

    print("🧭 [Planner] decomposing the question…")
    subs = plan(question, model)
    for i, s in enumerate(subs, 1):
        print(f"   {i}. {s}")

    print("\n🔎 [Researcher] answering each sub-question…")
    notes: list[tuple[str, str]] = []
    for i, s in enumerate(subs, 1):
        a = research(s, model)
        notes.append((s, a))
        print(f"   ✓ sub-question {i} answered ({len(a)} chars)")

    print("\n🖊️  [Writer] synthesizing the final answer…\n")
    answer = synthesize(question, notes, model)
    print("=" * 60)
    print(answer)
    print("=" * 60)
    print("\n✅ Agentic workflow complete — 1 plan + "
          f"{len(subs)} research + 1 synthesis calls, all through one router.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
