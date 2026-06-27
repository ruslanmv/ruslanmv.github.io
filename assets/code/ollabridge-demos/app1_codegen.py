#!/usr/bin/env python3
"""Program 1 — "Describe it, get a running project."

A tiny coding assistant in the spirit of GitPilot: you describe a program in one
sentence, an LLM (routed through OllaBridge Cloud) writes the files, we save them
to ./generated/, and then we RUN the result so you can see it actually works.

    python app1_codegen.py "a CLI that prints the first 10 Fibonacci numbers"

It demonstrates the core loop every AI coder uses:
    describe  ->  plan + files (JSON)  ->  write to disk  ->  run + show output
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

from ollabridge_client import chat

OUT = Path(__file__).parent / "generated"

SYSTEM = (
    "You are a senior Python engineer. Given a task, return ONLY a strict JSON "
    "object (no markdown fences) with this exact shape:\n"
    '{"files": [{"path": "main.py", "content": "..."}], "run": "python main.py"}\n'
    "Keep it to a single self-contained file named main.py with no third-party "
    "dependencies. The program must print its result to stdout when run."
)


def extract_json(text: str) -> dict:
    """Models sometimes wrap JSON in prose or ```json fences. Be forgiving."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            raise
        return json.loads(m.group(0))


def main() -> int:
    task = sys.argv[1] if len(sys.argv) > 1 else "a CLI that prints the first 10 Fibonacci numbers"
    model = sys.argv[2] if len(sys.argv) > 2 else "free-best"

    print(f"📝 Task   : {task}")
    print(f"🤖 Model  : {model}  (routed by OllaBridge Cloud)\n")

    print("→ Asking the router to write the code…")
    raw = chat(task, model=model, system=SYSTEM, temperature=0.1)
    spec = extract_json(raw)

    OUT.mkdir(exist_ok=True)
    written = []
    for f in spec["files"]:
        p = OUT / f["path"]
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f["content"])
        written.append(p)
        print(f"📄 Wrote  : {p.relative_to(OUT.parent)}  ({len(f['content'])} bytes)")

    run_cmd = spec.get("run", "python main.py")
    print(f"\n▶  Running: {run_cmd}")
    proc = subprocess.run(run_cmd, shell=True, cwd=OUT, capture_output=True, text=True, timeout=60)
    print("─" * 56)
    print(proc.stdout.rstrip() or "(no stdout)")
    if proc.stderr.strip():
        print("[stderr]", proc.stderr.rstrip())
    print("─" * 56)
    print("✅ Done — the router wrote it, your machine ran it." if proc.returncode == 0
          else f"⚠️ exited with code {proc.returncode}")
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
