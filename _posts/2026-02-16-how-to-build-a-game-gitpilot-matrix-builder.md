---
title: "How to Build a Governed Arcade Game with GitPilot + Matrix Builder"
excerpt: "A practical guide: turn one sentence into a controlled contract, let an AI coder build the game one validated batch at a time, and grow from a single-file prototype to a real Vite + TypeScript + Phaser engine — with the art pipeline that makes it look premium. Built with Claude or watsonx, your choice."
description: "Step-by-step tutorial for building a browser arcade platformer (Contract Quest) with GitPilot under Matrix Builder contracts: the governed loop, single-file vs Phaser architecture, the asset pipeline, and how to reach concept-art quality."
date: 2026-02-16
permalink: /blog/how-to-build-a-game-gitpilot-matrix-builder/
header:
  image: "/assets/images/posts/2026-06-16-how-to-build-a-game-gitpilot-matrix-builder/contract-quest.png"
  teaser: "/assets/images/posts/2026-06-16-how-to-build-a-game-gitpilot-matrix-builder/contract-quest.png"
  caption: "Contract Quest — coded by GitPilot, under a Matrix Builder contract."
tags:
  - matrix-builder
  - gitpilot
  - game-dev
  - tutorial
  - watsonx
toc: true
toc_label: "Contents"
---

This is the playbook behind the whole *under-contract* arcade — Pong, Tetris, Match-3, a neon climber, and **Contract Quest**, a governed platformer. The thesis is simple: **give the AI a contract, not a prompt.** Here's exactly how to do it, from one sentence to a polished, deployable game.

![Contract Quest running](/assets/images/posts/2026-06-16-how-to-build-a-game-gitpilot-matrix-builder/contract-quest.png)

## The two tools

- **[Matrix Builder](https://github.com/agent-matrix/matrix-builder)** (`mb`) — turns an idea into a *Matrix Bundle*: a locked blueprint, pinned standards, an **allow-list of exactly which files may change**, and acceptance criteria. It also **validates** the result (`mb check`: approved / needs-repair / rejected).
- **[GitPilot](https://gitpilot.ruslanmv.com)** — the coding agent that follows the contract. Point it at **any** model — Claude, OpenAI, or **IBM watsonx** — and it writes the code inside the allow-list.

![How it works](/assets/images/posts/2026-06-16-how-to-build-a-game-gitpilot-matrix-builder/how-it-works.svg)

## Step 1 — Install and pick a model

```bash
pip install agent-generator gitcopilot crewai

# Option A — Claude
export GITPILOT_PROVIDER=claude GITPILOT_CLAUDE_MODEL=claude-opus-4-8 ANTHROPIC_API_KEY=sk-ant-…
# Option B — watsonx (open models)
export GITPILOT_PROVIDER=watsonx GITPILOT_WATSONX_MODEL=openai/gpt-oss-120b \
       WATSONX_API_KEY=…  WATSONX_PROJECT_ID=…  WATSONX_URL=https://us-south.ml.cloud.ibm.com
```

The governance never changes — only the model does. (Tip: raise `GITPILOT_MAX_TOKENS` for big single-file generations.)

## Step 2 — The governed loop (per batch)

```bash
mb init "Contract Quest — a side-scrolling arcade platformer …" --quality standard
# then repeat, once per feature batch:
mb next "<the next batch goal>"            # plan a scoped batch (allow-list + acceptance)
mb prompt --coder gitpilot                 # render the contract-bound prompt
gitpilot generate -m "$(cat coder-prompts/gitpilot.md)" -o .
mb check frontend/index.html               # validate, fail-closed → Matrix Commit
```

Build the game in **batches** — foundation, controls, collectibles, enemies, levels, polish — each one validated before it can land. If the model writes outside the allow-list, `mb check` returns `needs-repair` and nothing ships. That's the whole point: **proof, not vibes.**

## Step 3 — Two ways to build (pick by ambition)

**A. Single self-contained `frontend/index.html`** — canvas + JS, no build step. Perfect for prototypes and game jams; instantly playable on GitHub Pages. This is how the watsonx **[Contract Quest prototype](https://github.com/ruslanmv/contract-quest-watsonx)** was built — 6 governed batches + a self-repair.

**B. A real engine** — **Vite + TypeScript + Phaser 3**, for a production game with scenes, entities, tilemaps, animation, and CI. The architecture lives in **[ruslanmv/contract-quest](https://github.com/ruslanmv/contract-quest)**:

```
src/
  main.ts                 # Phaser config (pixelArt, Arcade physics, FIT scale)
  scenes/{Boot,Preload,Game}.ts
  entities/{Player,BugBot,Slime,Coin}.ts
  ui/{HUD,TouchControls}.ts
  utils/{constants,textures}.ts
public/assets/            # sprites, tiles, backgrounds (loaded at runtime)
.github/workflows/deploy.yml   # typecheck → build → deploy to Pages
MATRIX_*.{yaml,lock,md}        # the contract identity + RMD governance
```

Reproduce it locally:

```bash
git clone https://github.com/ruslanmv/contract-quest && cd contract-quest
npm install && npm run dev      # → http://localhost:5173/
```

## Step 4 — Make it *look* premium (the part everyone misses)

Here's the hard truth: an LLM that writes **code** can't *paint* pixel art. The gorgeous look is an **asset problem**, not a prompt problem. Same engine, different art = a totally different game:

![Before / after — placeholder shapes vs pixel-art assets](/assets/images/posts/2026-06-16-how-to-build-a-game-gitpilot-matrix-builder/before-after.png)

The pipeline:

1. **Generate the art** with a pixel-art **image model** (Midjourney / SDXL / DALL·E) or **Aseprite**; lay out levels in **Tiled**. (Contract Quest ships `scripts/gen_assets.py` to generate an *original* placeholder set programmatically — hero sprite sheet, tiles, parallax skylines, coins, gate, HUD icons — so it looks good immediately.)
2. **Drop the PNGs into `public/assets/`** with the same names/sizes — the engine's `PreloadScene` loads them, builds animations, and `GameScene` renders parallax layers, glow/bloom, particles, a color-grade, and an icon HUD with **zero code changes**.
3. To match a concept render exactly, follow the prompt pack in [`docs/ASSET_PROMPTS.md`](https://github.com/ruslanmv/contract-quest/blob/main/docs/ASSET_PROMPTS.md).

**The division of labor:** GitPilot (any model, under contract) writes the *engineering*; an image model / artist provides the *art*. Matrix Builder keeps both honest.

## Step 5 — Ship it

CI (`typecheck → build → deploy`) publishes `dist/` to GitHub Pages. Set `base` in `vite.config.ts` to `/<repo>/`, enable **Settings → Pages → GitHub Actions**, and you're live.

## Step 6 — Scale safely: an 8-episode campaign, *additive by design*

A prototype is one level. A real game is a **campaign** — and this is where most "AI built a game" demos fall apart: every new level means another sprawling, ungoverned edit. The fix is the same discipline as the batches: make content **data, not code**.

Contract Quest grew into a **Super Mario Bros-style 8-episode quest** — a story (the *Builder* restoring corrupted realms), per-episode narrative cards, power-ups, a *Rogue Architect* boss, and end credits — **without rewriting the engine**. Each episode is one small config object:

```ts
// src/levels/episodes.ts  — adding an episode = adding ONE entry. The engine never changes.
{
  id: 4, title: "IV — The Parallax Heights", subtitle: "Ascend",
  story: ["The spires of the old build pierce the dusk.", "Climb. The higher contracts await."],
  width: 3000, platforms: 10, enemies: 5, slimeRatio: 0.5, moving: true,
  coinArcs: 4, djump: true, accent: 0x00f0ff, seed: 44,
}
```

![Contract Quest — 8-episode campaign: title, story card, gameplay](/assets/images/posts/2026-06-16-how-to-build-a-game-gitpilot-matrix-builder/campaign.png)

This is the scalability story in one pattern. **An episode is just a batch in another form:** small, scoped, validated, additive. A whole team — or a fleet of contract-bound AI agents — can add episodes, enemies, and mechanics in parallel; the engine, the standards, and the `mb check` gate stay fixed. You grow a 56 KB demo into a real title the same way you'd grow a 10-million-line engine: **content scales, governance holds.** That's how this technology takes complex projects — up to AAA — *safely*.

## Take it further

- **Contract Quest (engine):** [github.com/ruslanmv/contract-quest](https://github.com/ruslanmv/contract-quest)
- **Contract Quest (watsonx prototype):** [github.com/ruslanmv/contract-quest-watsonx](https://github.com/ruslanmv/contract-quest-watsonx)
- **Matrix Builder:** [agent-matrix/matrix-builder](https://github.com/agent-matrix/matrix-builder) · **GitPilot:** [gitpilot.ruslanmv.com](https://gitpilot.ruslanmv.com)
- **The arcade:** [Pong](https://github.com/ruslanmv/pong-under-contract) · [Tetris](https://github.com/ruslanmv/tetris-under-contract) · [Match-3](https://github.com/ruslanmv/match-3-under-contract) · [Neon Climber](https://github.com/ruslanmv/doodle-jump-climber-under-contract)

*The takeaway is the method, not the game: give an AI a contract instead of a prompt, and keep the evidence of what it changed. A small demo and a large engine are built the same way — the content scales, and the governance holds.*
