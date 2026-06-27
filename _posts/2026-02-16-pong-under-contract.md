---
title: "I Didn't Write This Pong Game — Claude Opus 4.8 Did, Under Contract"
excerpt: "Most 'AI built X' demos can't prove the AI stayed in scope. So I gave Claude a contract instead of a prompt: Matrix Builder locked the blueprint and the allow-list, GitPilot drove Claude Opus 4.8 to write the game, and mb check validated it — approved, score 100 — before anything could land. With a full, reproducible evidence trail."
description: "A real, auditable run: Matrix Builder turns one sentence into a controlled contract, GitPilot drives Claude Opus 4.8 to code a retro Pong game touching only the allowed file, and mb check returns MATRIX_STATUS approved before commit. Provider, model, command, and verdict are all recorded."
date: 2026-02-16
permalink: /blog/pong-under-contract/
header:
  image: "/assets/images/posts/2026-06-16-pong-under-contract/hero.svg"
  teaser: "/assets/images/posts/2026-06-16-pong-under-contract/hero.svg"
  caption: "Pong Under Contract — coded by Claude Opus 4.8, driven by GitPilot, under a Matrix Builder contract."
tags:
  - matrix-builder
  - gitpilot
  - agent-generator
  - claude
  - ai-coding
  - multi-agent
toc: true
toc_label: "Contents"
---

You've seen the headline a hundred times: *"I asked an AI to build X and it did!"* — and then you open the repo and it's a 600-line diff that touched twelve files you never mentioned, invented two dependencies, and edited the CI config "to help." It works, maybe. But could you **prove** what the AI was *allowed* to do?

This time I wanted the proof. So I let **Claude Opus 4.8** build a complete retro **Pong** game — but I refused to give it a prompt. I gave it a **contract**. And I kept the receipts.

## 🎮 Play it first

> **▶ [Play Pong Under Contract](https://ruslanmv.github.io/pong-under-contract/)** — one self-contained HTML file an AI wrote. Beat the CPU to 11. Move with `W`/`S`, arrows, or touch.

![Pong Under Contract — coded by Claude Opus 4.8, under a Matrix Builder contract](/assets/images/posts/2026-06-16-pong-under-contract/hero.svg)

And here's the **actual game** running in a browser — a real headless capture, not a mock-up: glowing neon paddles and ball, a dashed center net, CRT scanlines, and the build credit baked right in.

![The real neon Pong running — glowing green on a dark CRT](/assets/images/posts/2026-06-16-pong-under-contract/gameplay.png)

Everything is open at **[github.com/ruslanmv/pong-under-contract](https://github.com/ruslanmv/pong-under-contract)** — including the Matrix Bundle that governed the build and an [`EVIDENCE.md`](https://github.com/ruslanmv/pong-under-contract/blob/main/EVIDENCE.md) recording the exact run.

## Two tools, one idea: control, not vibes

- **[Matrix Builder](https://github.com/agent-matrix/matrix-builder)** gives AI coders **a contract, not a prompt** — a *Matrix Bundle* with a locked blueprint, a pinned standards file, an **allow-list of exactly which files may change**, and acceptance criteria. The `mb` CLI ships in `agent-generator` and runs offline.
- **[GitPilot](https://gitpilot.ruslanmv.com)** is the open-source multi-agent coding assistant and the **native worker for Matrix Builder**. It runs any LLM — OpenAI, **Anthropic Claude**, IBM Watsonx, or Ollama (local & free). Claude is in fact its default provider.

One plans. One executes. And a validator stands between the model and your `main` branch.

## Exactly how I built it — step by step, in the terminal

I did the whole thing from the **command line**, not the VS Code extension or the web app — because the CLI is scriptable and every step is reproducible. Two CLIs do the work: **`mb`** (Matrix Builder, ships inside `agent-generator`) and **`gitpilot`** (the GitPilot CLI). Here is every step I ran.

### Step 0 — Install the three tools

```bash
pip install agent-generator   # the `mb` CLI — Matrix Builder + the contract engine
pip install gitcopilot        # the `gitpilot` CLI — the AI coder
pip install crewai            # the agent runtime GitPilot uses to call the LLM
```

### Step 1 — Point GitPilot at Claude Opus 4.8

GitPilot reads three environment variables to choose the provider, model, and key — that's the entire configuration:

```bash
export GITPILOT_PROVIDER=claude            # Claude is GitPilot's default provider
export GITPILOT_CLAUDE_MODEL=claude-opus-4-8
export ANTHROPIC_API_KEY=sk-ant-…          # your Anthropic key
```

Swap these three for `openai`, `watsonx`, or `ollama` (local & free) and **nothing else in the process changes**.

### Step 2 — Turn one sentence into a contract (`mb`)

```bash
mb init "A retro Pong arcade game … single self-contained HTML file …" --quality standard
mb next "Build the complete Pong game in one self-contained frontend/index.html"
```

`mb next` plans a scoped batch. I reviewed it and set the allow-list to the single file the game needs — `frontend/index.html` — then rendered the contract-bound prompt for the coder:

```bash
mb prompt --coder gitpilot     # writes coder-prompts/gitpilot.md  (+ .gitpilotrules)
```

That prompt isn't "please build Pong." It's a contract: *implement TASK-002 only; **edit ONLY `frontend/index.html`** (RMD-002, RMD-107); don't add dependencies (RMD-105); end with `MATRIX_STATUS`.* Those `RMD-xxx` rules bake in NIST SSDF, OWASP Top 10 & OWASP LLM Top 10, and SLSA.

### Step 3 — Let Claude write the game, through GitPilot (`gitpilot generate`)

I piped the contract prompt straight into GitPilot's local generator. This is the line where the AI actually writes the code:

```bash
gitpilot generate -m "$(cat coder-prompts/gitpilot.md)" -o .
```

```text
Provider: claude · Model: claude-opus-4-8
Generating... done
  Created: frontend/index.html (15027 bytes)
Generated 1 file(s)
```

GitPilot called **Claude Opus 4.8** and wrote **exactly one file** — the allow-listed one. No stray configs, no invented dependencies. The model produced a genuinely good game, too: angle-based paddle bounce, a ball that speeds up, a capped-speed AI, keyboard **and** touch controls, and a win screen at 11.

### Step 4 — Validate the AI's output — *before* it can land (`mb check`)

```bash
mb check frontend/index.html
```

```text
MATRIX_STATUS: approved  score=100
  committed mc-caf49981d56a        # ← an immutable, versioned Matrix Commit
```

`approved`, score 100 — so the change matured into a **Matrix Commit** pinning the prompt, the diff, the standards lock, and the verdict. Everything above — provider, model, command, byte count, verdict — is recorded verbatim in [`EVIDENCE.md`](https://github.com/ruslanmv/pong-under-contract/blob/main/EVIDENCE.md). This is the difference between *"looks done"* and **proof**.

### Step 5 — Sanity-check that it actually runs

```bash
node --check script.js     # extracted <script> → valid JavaScript, no syntax errors
```

A quick `grep` confirmed the canvas, game loop, input handlers, score-to-11, and win screen were all present.

### Step 6 — Ship it

Commit, push, and let CI re-run the very same `mb check` as a gate before deploying the game to GitHub Pages ([`.github/workflows/contract.yml`](https://github.com/ruslanmv/pong-under-contract/blob/main/.github/workflows/contract.yml)). If the contract isn't satisfied, the deploy never happens.

> **CLI or desktop?** I used the headless **`gitpilot generate`** command in a terminal. The same GitPilot engine also runs as a **VS Code extension** and as a **web app** at [gitpilot.ruslanmv.com](https://gitpilot.ruslanmv.com). There, instead of piping the prompt on the command line, you'd paste the `mb prompt` output into GitPilot's chat and **approve each diff in the UI** (Ask mode). The contract (`mb`) and the verdict (`mb check`) are identical either way — only the surface changes.

## "But what if the model colors outside the lines?"

Then it doesn't ship. `mb check` is **fail-closed** — exit `0` approved · `1` needs-repair · `2` rejected. If the model writes to a file outside the allow-list, the verdict flips to `needs-repair` and `mb repair` generates a tightly-scoped fix prompt. The demo repo enforces exactly this in CI ([`.github/workflows/contract.yml`](https://github.com/ruslanmv/pong-under-contract/blob/main/.github/workflows/contract.yml)): the pipeline re-runs `mb check` and refuses to deploy unless the verdict is `approved`. In this build it ran green on a real GitHub Actions runner.

## From *prompting* to *engineering*

| Prompting | Engineering — Matrix Builder + GitPilot + Claude |
|---|---|
| "Build me Pong" → 🤞 | A blueprint, an allow-list, acceptance criteria |
| Edits anything, anywhere | Claude wrote **only** `frontend/index.html` |
| Un-reviewable diff | One scoped diff + a recorded run |
| "Looks done?" | `MATRIX_STATUS: approved score=100` + a signed commit |
| Trust | **Proof** |

It works the same whether the coder is Claude Opus 4.8 or a tiny local model. In an [earlier post]({{ '/matrix-builder-cli/' | relative_url }}) I ran the identical loop with a 1.5B model on [OllaBridge](https://github.com/ruslanmv/ollabridge)'s free tier — kept inside the contract, even a micro-model produced a safe, committed patch. **Swap the model; the governance never changes.**

## Reproduce it

```bash
pip install agent-generator gitcopilot crewai
mb init "your idea here" --quality standard && mb next "first feature"
mb prompt --coder gitpilot
export GITPILOT_PROVIDER=claude GITPILOT_CLAUDE_MODEL=claude-opus-4-8 ANTHROPIC_API_KEY=sk-ant-…
gitpilot generate -m "$(cat coder-prompts/gitpilot.md)" -o .
mb check <files you allowed it to touch>
```

## Take it for a spin

- **Play / fork:** [github.com/ruslanmv/pong-under-contract](https://github.com/ruslanmv/pong-under-contract) — read `EVIDENCE.md` and audit `.mb/`.
- **Matrix Builder:** [agent-matrix/matrix-builder](https://github.com/agent-matrix/matrix-builder)
- **GitPilot:** [gitpilot.ruslanmv.com](https://gitpilot.ruslanmv.com)
- **Engine + `mb`:** [ruslanmv/agent-generator](https://github.com/ruslanmv/agent-generator)

The next time an AI writes your code, don't ask it to behave — give it a contract, and keep the record of what it changed.

*If auditable, in-scope AI development is the direction you care about, the whole project is open — the code, the evidence, and the Matrix bundle — at [pong-under-contract](https://github.com/ruslanmv/pong-under-contract).*
