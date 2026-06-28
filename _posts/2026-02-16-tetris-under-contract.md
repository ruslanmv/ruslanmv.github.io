---
title: "I Made Claude Opus 4.8 Build a Full Tetris — in 5 Batches, Each Under Contract"
excerpt: "Not a one-shot prompt. I gave an AI a contract and let it build a complete, colorful neon Tetris one governed batch at a time — foundation, controls, scoring, juice, polish. Every batch validated before it could land. Play it, then see exactly how (and why) it works."
description: "A step-by-step, fully auditable build: Matrix Builder turns one sentence into a contract, GitPilot drives Claude Opus 4.8 to write a single-file Tetris across 5 scoped batches, and mb check validates each one (approved, score 100) before commit. With diagrams and a real headless-browser screenshot."
date: 2026-02-16
permalink: /blog/tetris-under-contract/
header:
  image: "/assets/images/posts/2026-06-16-tetris-under-contract/hero.svg"
  teaser: "/assets/images/posts/2026-06-16-tetris-under-contract/hero.svg"
  caption: "Neon Tetris — built by Claude Opus 4.8 across 5 governed Matrix Builder batches."
tags:
  - matrix-builder
  - gitpilot
  - claude
  - ai-coding
  - multi-agent
  - game-dev
toc: true
toc_label: "Contents"
---

A while back I let an AI build [a Pong game under contract]({{ '/blog/pong-under-contract/' | relative_url }}). Pong is one file and a couple of hundred lines — a good proof, but small. So I raised the bar: could an AI build something *real* — a complete, polished, colorful **Tetris** — not in a single heroic prompt, but the way an engineer actually ships software: **one governed batch at a time**, with a validator standing between the model and `main`?

It could. Here's the whole thing, step by step — and a diagram of how (and why) it works.

## 🎮 Play it first

> **▶ [Play Neon Tetris](https://ruslanmv.github.io/tetris-under-contract/)** — one self-contained HTML file, desktop and mobile. Arrows/WASD move, **↑/X** rotate, **Space** hard drop, **C** hold, **P** pause.

![Neon Tetris — hero](/assets/images/posts/2026-06-16-tetris-under-contract/hero.svg)

Here's the real thing running (a headless-browser capture, not a mock-up): live HUD, hold + next queue, ghost piece, neon glow.

![Neon Tetris gameplay screenshot](/assets/images/posts/2026-06-16-tetris-under-contract/gameplay.png)

Everything — including the contract that governed each batch — is open at **[github.com/ruslanmv/tetris-under-contract](https://github.com/ruslanmv/tetris-under-contract)**.

## The idea: 5 batches, each under contract

Most "AI built X" demos are one giant prompt and a shrug. This was the opposite. I used two tools from the terminal:

- **[Matrix Builder](https://github.com/agent-matrix/matrix-builder)** (`mb`) — turns an idea into a *contract*: a locked blueprint, pinned standards, an **allow-list of exactly which files may change**, and acceptance criteria. It also **validates** the result.
- **[GitPilot](https://gitpilot.ruslanmv.com)** — the coder. I pointed it at **Claude Opus 4.8** and let it write the game, bound by each batch's contract.

Then I ran the same loop five times, growing the game from a board into a finished arcade title.

## Step by step (everything in the terminal)

### Step 0 — Install + point GitPilot at Claude Opus 4.8

```bash
pip install agent-generator gitcopilot crewai

export GITPILOT_PROVIDER=claude
export GITPILOT_CLAUDE_MODEL=claude-opus-4-8
export ANTHROPIC_API_KEY=sk-ant-…
```

### Step 1 — One sentence becomes a contract

```bash
mb init "A polished neon Tetris game … single self-contained HTML file, runs on GitHub Pages" \
        --quality standard --title "Tetris Under Contract"
```

### Steps 2–6 — Five governed batches

Each batch ran the **same four commands**: plan a scoped batch, render the contract-bound prompt, let Claude write the code, validate fail-closed.

```bash
# (repeated, once per batch)
mb next "<the batch goal>"                 # plan a scoped batch (allow-list: frontend/index.html)
mb prompt --coder gitpilot                 # render the contract-bound prompt
gitpilot generate -m "$(cat coder-prompts/gitpilot.md)" -o .
mb check frontend/index.html               # validate → approved / needs-repair / rejected
```

Here's what each batch asked Claude to add, the file size after it, and the immutable Matrix Commit it produced:

| # | Batch | What Opus 4.8 added | Size | Matrix Commit |
|---|---|---|---|---|
| 1 | **Foundation** | 10×20 board, 7 colored tetrominoes, render loop, gravity, lock, 7-bag | 11 KB | `mc-e5da1ec40b74` |
| 2 | **Controls** | move/soft/hard drop, **SRS rotation + wall kicks**, lock delay, DAS | 20 KB | `mc-ed8f08a81d49` |
| 3 | **Scoring** | line clears, scoring, levels/speed, **next** preview, **hold** | 28 KB | `mc-1e2903a575aa` |
| 4 | **Juice** | ghost piece, particles + flash, **WebAudio** SFX, mobile touch, responsive | 44 KB | `mc-11c9835f1a93` |
| 5 | **Polish** | start / pause / game-over, **high score** (localStorage), accessibility | 56 KB | `mc-c186f714b6a1` |

Every single batch returned the same verdict:

```text
$ mb check frontend/index.html
MATRIX_STATUS: approved  score=100
  committed mc-…
```

And the timeline tells the whole story at a glance:

```text
$ mb timeline
Tetris Under Contract  v1.0.0
  Batch 01  Foundation                             ✓ mc-e5da1ec40b74
  Batch 02  Controls and SRS rotation              ✓ mc-ed8f08a81d49
  Batch 03  Line clears, scoring, levels, next/hold ✓ mc-1e2903a575aa
  Batch 04  Juice: ghost, particles, sound, mobile  ✓ mc-11c9835f1a93
  Batch 05  Game states, high score, accessibility  ✓ mc-c186f714b6a1
```

The crucial detail: across all five batches, the model wrote to **only `frontend/index.html`** — never a stray file. That's not luck; it's the allow-list. If Claude had written anywhere else, `mb check` returns `needs-repair` (exit 1) or `rejected` (exit 2) and the change is blocked.

### Step 7 — Did it actually work?

Syntax-checking every batch (`node --check`) wasn't enough for me, so I ran the finished game in a real headless **Chromium** — started it, moved, rotated, hard-dropped, held a piece:

```text
RUNTIME JS ERRORS: NONE ✓
```

Zero errors. The screenshot above *is* that run. The full transcript — provider, model, byte counts, and every verdict — lives in [`EVIDENCE.md`](https://github.com/ruslanmv/tetris-under-contract/blob/main/EVIDENCE.md).

> One honest note: GitPilot's Anthropic path didn't set `max_tokens`, so the first big generation got truncated. I added a configurable `max_tokens` to GitPilot (a real upstream fix) — and the batches flowed.

## How it works — and why Matrix Builder

Here's the whole loop in one picture, plus what `mb` actually buys you:

![How it works: Matrix Builder × GitPilot × Claude, and the advantages of mb](/assets/images/posts/2026-06-16-tetris-under-contract/how-it-works.svg)

The shift is from **prompting** to **engineering**. A prompt is a guess: the model wanders, invents dependencies, and nobody can prove what it was *allowed* to do. A contract flips that:

- **A contract, not a prompt** — a locked blueprint and pinned standards, not vibes.
- **Allow-list scope** — the AI edits only the files you permit; everything else is off-limits.
- **Fail-closed validation** — `mb check` returns *approved / needs-repair / rejected* (exit 0 / 1 / 2), so it gates CI.
- **Immutable Matrix Commits** — every accepted change pins the prompt, the diff, and the verdict: a real audit trail.
- **Standards baked in** — NIST SSDF, OWASP Top 10 & LLM Top 10, SLSA.
- **Provider-agnostic** — Claude here, but OpenAI, Watsonx, or local Ollama work the same; the governance never changes.
- **Offline & deterministic** — the planning and validation stay on your machine.

You keep the raw speed of AI generation, and you reclaim the safety, auditability, and peace of mind production software demands. Five batches, one file, **zero out-of-scope edits** — and a game you can actually play.

## Reproduce it

```bash
pip install agent-generator gitcopilot crewai
export GITPILOT_PROVIDER=claude GITPILOT_CLAUDE_MODEL=claude-opus-4-8 ANTHROPIC_API_KEY=sk-ant-…
mb init "your game idea" --quality standard
# then, per batch:
mb next "the next feature" && mb prompt --coder gitpilot
gitpilot generate -m "$(cat coder-prompts/gitpilot.md)" -o .
mb check frontend/index.html
```

## Take it for a spin

- **Play / fork:** [github.com/ruslanmv/tetris-under-contract](https://github.com/ruslanmv/tetris-under-contract) — read `EVIDENCE.md`, audit the `.mb/` bundle (5 commits).
- **Matrix Builder:** [agent-matrix/matrix-builder](https://github.com/agent-matrix/matrix-builder)
- **GitPilot:** [gitpilot.ruslanmv.com](https://gitpilot.ruslanmv.com)
- **Engine + `mb` CLI:** [ruslanmv/agent-generator](https://github.com/ruslanmv/agent-generator)
- **Also:** [Pong, under contract]({{ '/blog/pong-under-contract/' | relative_url }})

*The game was never really the point — the method was. Five scoped batches, each validated before it could land, show that AI can keep its raw speed without giving up control, auditability, or review. That is the kind of governed workflow I believe production AI coding will increasingly depend on, and it is the same approach behind Matrix Builder and GitPilot.*
