---
title: "Give Your AI Coder a Contract: A 2-Minute Tour of Matrix Builder (mb)"
excerpt: "AI writes code in seconds — then forgets the rules. Matrix Builder gives every AI coder a contract, not a prompt, and versions every change like Git. Here's the whole loop in your terminal."
date: 2026-06-14
permalink: /matrix-builder-cli/
header:
  image: "/assets/images/posts/2026-06-14-matrix-builder-cli/hero.svg"
  teaser: "/assets/images/posts/2026-06-14-matrix-builder-cli/hero.svg"
  caption: "Matrix Builder — Git for AI build contracts."
tags:
  - matrix-builder
  - agent-matrix
  - agent-generator
  - ai-coding
  - cli
  - claude-code
toc: true
toc_label: "On this page"
toc_sticky: true
---

AI writes code in seconds — then promptly forgets the rules, wanders off your architecture, and quietly edits files it shouldn't touch. We've all been there: a single loose prompt turns a clean codebase into an un-reviewable, 400-line diff nightmare.

**Matrix Builder** fixes that structural breakdown. It shifts the paradigm by giving every AI coder a **contract, not a prompt** — and versioning every single change exactly like Git.

The local tool is `mb`. It's completely offline, deterministic, and requires nothing but a standard Python environment. Let's run it end-to-end.

## Install (one line)

```bash
pip install agent-generator        # ships the `mb` CLI + the core engine
mb --version
# mb (agent-generator) 0.2.0
```

No remote servers, no mandatory database, and no prerequisite API keys.

---

## The whole loop, in your terminal

Here's how the control loop works when you want to build a feature cleanly:

```bash
# 1) Turn an idea into a controlled blueprint (housed locally in ./.mb/)
$ mb init "A GitHub repo intelligence agent" --quality standard
Initialized .mb/ for Standard Matrix Bundle
  project    bp-56d0c6f90021
  version    v1.0.0  ·  quality standard
  blueprint  1 task(s)  ·  stack fastapi/nextjs

# 2) Plan one scoped batch — it explicitly declares allowed files + acceptance checks
$ mb next "Add repo ingestion"
Batch 01  Add repo ingestion  (add-feature)
  TASK-002: Apply the requested update
    - backend/app/api/routes.py
    - backend/tests/test_routes.py
  acceptance: pytest -q, ruff check ., npm run build

# 3) Generate a contract-bound prompt for your AI coder (Claude Code, Cursor, GitPilot…)
$ mb prompt --coder claude-code
You are Claude Code, an expert software engineer.
…
Governing rules: RMD-101 … RMD-120.
End your response with: MATRIX_STATUS: approved | needs_repair | rejected
# (paste this into your AI coder — the framework forces it to stay inside allowed files)

# 4) Validate what the AI changed — this is where the magic happens
$ mb check backend/app/api/routes.py
MATRIX_STATUS: approved  score=100
  committed mc-454fa2735cd7        # ← an immutable, versioned Matrix Commit
```

### Watching the contract hold

What happens if the AI tries to color outside the lines? Let's intentionally tell it to modify a locked configuration file:

```bash
$ mb check MATRIX_BLUEPRINT.yaml
MATRIX_STATUS: rejected  score=60
  -  RMD-002: Modified a forbidden contract file: MATRIX_BLUEPRINT.yaml
  next      mb repair --copy
```

Instantly rejected. `mb repair` handles the failure gracefully by generating a tightly scoped repair prompt to feed back to the AI. And you can audit your entire structural progress at any time:

```bash
$ mb timeline
Standard Matrix Bundle  v1.0.0  ·  A GitHub repo intelligence agent
  Batch 01  Add repo ingestion  (add-feature)
           ✓ commit 001 mc-454fa2735cd7
```

---

## What just happened behind the scenes?

Every accepted change automatically matures into a **Matrix Commit** — an immutable checkpoint pinning the exact prompt, the generated diff, the standards lock, and the validation result.

Because `mb check` is rigorously **fail-closed** (exit codes: `0` approved · `1` needs-repair · `2` rejected), you can drop it straight into GitHub Actions or any CI pipeline to gate rogue AI edits.

And the guardrails aren't arbitrary. Every build is locked to the signed **Ruslan Magana Definitions (RMD)**, which bake in industry security baselines:

* **NIST SSDF** — Secure Software Development Framework
* **OWASP Top 10** & **OWASP LLM Top 10**
* **SLSA** — Supply-chain Levels for Software Artifacts

Your AI coder finally behaves like a disciplined engineer following a strict blueprint — not a rogue agent guessing its way through production code.

---

## Direct evaluation: testing with a free cloud LLM

The terminal loop above hands the prompt to *you* to feed your favorite IDE tool. But since `mb` is fundamentally an engine, I pushed it further: I wired a lightweight **AI bridge** that captures the output of `mb prompt`, dispatches it to a remote model, writes the returned code to disk, and runs `mb check` to close the loop automatically.

The beauty is in the routing — **only one step ever leaves your machine.** Everything else (planning, validation, committing) stays local and deterministic:

![End-to-end architecture: idea → mb → contract prompt → OllaBridge cloud LLM → files → mb check → Matrix Commit](/assets/images/posts/2026-06-14-matrix-builder-cli/e2e-architecture.svg)

For the AI engine, I opted for a deliberately brutal stress test: **[OllaBridge](https://github.com/ruslanmv/ollabridge)'s free cloud tier** running `qwen2.5:1.5b` — an OpenAI-compatible endpoint that needs **no API key**. The question: can a tiny, heavily constrained 1.5B model navigate the contract framework at all?

```text
$ mb init "A simple hello world web page" --quality starter
Initialized .mb/ for Starter controlled blueprint   ·   stack fastapi/nextjs

$ mb next "Create the hello world landing page"
Batch 01  Create the hello world landing page  (add-feature)
  allowed files: backend/app/api/routes.py · backend/tests/test_routes.py

$ mb prompt --coder generic --file prompt.md      # generates the contract prompt
#  → AI bridge posts prompt.md to OllaBridge free tier (qwen2.5:1.5b) … ~5s
#  → writes the files it returned, safely inside allowed paths

$ mb check backend/app/api/routes.py backend/tests/test_routes.py
MATRIX_STATUS: approved  score=100
  committed mc-f5e6ed8fb23e
```

The tiny model even abided by the contract's stop condition, ending its raw stream with exactly `MATRIX_STATUS: approved`. The result was a working FastAPI route serving our target page:

![The Hello World page generated end-to-end by the free OllaBridge model and approved by mb](/assets/images/posts/2026-06-14-matrix-builder-cli/hello-world.png)

> **An honest post-mortem:** a 1.5B-parameter model isn't flawless — on the first pass it slipped on a minor import (`HTMLResponse` belongs in `fastapi.responses`). But that *is* the thesis: **kept strictly inside a build contract, even a micro-sized model produced a safe, auditable, committed patch in seconds.** Scaling this exact pipeline to a larger *paired* OllaBridge model — or an elite frontier coder — makes the output production-clean immediately, and the governance layer stays perfectly unchanged. (That's a topic for the next post.)

---

## Why this matters: from "prompting" to "engineering"

| | Prompt-and-pray AI coding | Contract-driven engineering (`mb`) |
| --- | --- | --- |
| **Blast radius** | Can modify, corrupt, or hallucinate any file in the repo | Strictly bound to declared files; forbidden edits drop out immediately |
| **Auditability** | Lost in shell history or an ephemeral chat tab | Immutable **Matrix Commits** pinning prompt, diff, and compliance score |
| **CI compatibility** | Subjective; hard to automate | **Fail-closed** binary exit codes built for deployment gates |
| **Vendor lock-in** | Tied to a specific tool's workspace | Fully model-agnostic — free local LLMs or frontier APIs, same contract |
| **Reproducibility** | Same prompt, different result each run | Deterministic — same idea replays into the same controlled build |

---

## Prefer a visual interface?

If the `mb` CLI is the *Git* of AI generations, our upcoming web platform is the *GitHub*. A collaborative cloud is in **active development** at [build.matrixhub.io](https://build.matrixhub.io) — visual build timelines, audit logs, and shared blueprints. Today the CLI is the rock-solid bedrock path, and the web app is evolving alongside it fast.

## Take it for a spin

Get your AI coders under control right now:

```bash
pip install agent-generator
mb init "your idea here" --quality standard
```

* 🌐 **Docs:** [agent-matrix.github.io/matrix-builder](https://agent-matrix.github.io/matrix-builder/site/)
* 💻 **Repo:** [agent-matrix/matrix-builder](https://github.com/agent-matrix/matrix-builder)
* ⚙️ **Core engine:** [ruslanmv/agent-generator](https://github.com/ruslanmv/agent-generator)

*If deterministic AI development matches your philosophy, drop a ⭐ on the repo — it goes a long way. And tell me in the comments: which AI coder are you locking into a contract first?*
