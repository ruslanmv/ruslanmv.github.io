---
title: "Contract Quest: Reproducing a Governed AI Game Build with Matrix Designer, Matrix Builder, GitPilot, and watsonx.ai"
excerpt: "A step-by-step reproduction guide for building Contract Quest from zero with a production build script: design first, govern the batches, generate with GitPilot on watsonx.ai, verify every stage, and publish only with evidence."
description: "A practical end-to-end tutorial for reproducing Contract Quest, a governed static browser game generated through Matrix Designer, Matrix Builder, GitPilot, and openai/gpt-oss-120b on IBM watsonx.ai. Includes setup commands, the build workflow, debugging lessons, verification gates, and the final production build.sh script."
date: 2026-06-23
permalink: /blog/contract-quest-watsonx/
header:
  image: "/assets/images/posts/2026-06-16-contract-quest-watsonx/hero.svg"
  teaser: "/assets/images/posts/2026-06-16-contract-quest-watsonx/hero.svg"
  caption: "Contract Quest — designed first, governed batch by batch, generated with GitPilot on IBM watsonx.ai."
tags:
  - matrix-designer
  - matrix-builder
  - gitpilot
  - watsonx
  - watsonx-ai
  - gpt-oss
  - game-dev
  - tutorial
  - vercel
  - webaudio
  - qa
  - reproducible-builds
toc: true
toc_label: "Contents"
---

What if an AI-generated game could be more than a quick demo?

What if the entire build could be designed, scoped, generated, verified, repaired, and explained in a way that another developer can reproduce?

That is the story behind **Contract Quest**, a governed arcade platformer built with **Matrix Designer**, **Matrix Builder**, **GitPilot**, and **IBM watsonx.ai**.

The visible artifact is a browser game: a small robot hero runs through a contract-themed pixel world, collects Contract Coins and Validation Gems, reads RMD panels, activates checkpoints, uses Shield and Double Jump power-ups, fights Bug Bots and Prompt Slimes, and finally defeats the Rogue Architect before entering the Matrix Gate.

The deeper artifact is the pipeline.

This article is not only about the game. It is about the method we used to build it from zero:

```text
Design the product.
Govern the implementation.
Generate bounded code changes.
Verify every batch.
Repair only when evidence says repair is needed.
Publish with logs and evidence.
```

The final result is summarized by one script: `build.sh`. That script is the practical version of this whole tutorial. At the end of this post, I share the production-ready script that reproduces the workflow.

## Why this matters

Most AI coding workflows begin with a prompt like this:

```text
Make me a game.
```

That can produce something interesting, but it usually has weak boundaries. The model may invent architecture, add files freely, skip tests, hide assumptions, or generate code that only works once on one machine.

For Contract Quest, I wanted the opposite.

I wanted a process where the AI cannot simply wander through the repository. Every stage should have a contract:

- What is being built?
- Which files may change?
- Which commands prove it works?
- Which checks decide whether we continue?
- Which evidence supports the final blog post and release claims?

That is why the workflow starts with **Matrix Designer**, not GitPilot.

Matrix Designer defines the product. Matrix Builder governs the batches. GitPilot writes code from bounded prompts. watsonx.ai runs the model. Verification decides whether the next batch is allowed to continue.

## The architecture of the workflow

The tools have separate responsibilities.

| Layer | Tool | Responsibility |
|---|---|---|
| Product design | Matrix Designer | Creates the blueprint, design bundle, visual target, architecture, entity contracts, acceptance criteria, and batch roadmap. |
| Governance | Matrix Builder | Converts the design into scoped implementation batches with allowed files and checks. |
| Code generation | GitPilot | Executes each implementation prompt locally and writes files to disk. |
| Model runtime | IBM watsonx.ai | Provides `openai/gpt-oss-120b` as the generation model. |
| Verification | npm, Playwright, Matrix checks | Proves that the generated game still builds, runs, and respects the contract. |
| Evidence | `EVIDENCE.md` and logs | Records what happened so the release can be audited. |

The important point is the order:

```text
Matrix Designer -> Matrix Builder -> GitPilot -> watsonx.ai -> tests -> evidence
```

GitPilot does not invent the game. It implements a batch derived from a design artifact.

## What we are building

Contract Quest is a static browser game that can be deployed on Vercel. It does not need a backend, database, API key, or runtime watsonx credential to play.

The target includes:

- a premium arcade platformer feel;
- original deterministic art;
- a contract-themed world;
- mobile and desktop controls;
- generated WebAudio music and SFX;
- RMD panels and Matrix Gates;
- checkpoints, coins, validation gems, shields, and double jump;
- enemies and a Rogue Architect boss;
- a final gate that cannot be completed while the boss is alive;
- static verification scripts;
- Playwright smoke tests;
- Vercel deployment support;
- and evidence logs for the full build.

The final game can be simple. The workflow is the product.

## Prerequisites

The reproduction assumes WSL/Linux, Node.js, Python 3.11 or 3.12, `uv`, npm, and Git.

Install `uv` if needed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
uv --version
```

Check Python and Node:

```bash
python3.11 --version
node --version
npm --version
```

If your default Python is 3.13, prefer Python 3.11 for this workflow:

```bash
make install PYTHON=python3.11
```

## Environment configuration

Create a `.env` file in the project root.

Do not add trailing spaces after values.

```bash
cat > .env <<'EOF'
GITPILOT_PROVIDER=watsonx
WATSONX_API_KEY=replace-with-your-ibm-cloud-api-key
WATSONX_PROJECT_ID=replace-with-your-watsonx-project-id
PROJECT_ID=replace-with-your-watsonx-project-id
WATSONX_URL=https://us-south.ml.cloud.ibm.com
WATSONX_BASE_URL=https://us-south.ml.cloud.ibm.com
GITPILOT_WATSONX_MODEL=openai/gpt-oss-120b
GITPILOT_MAX_TOKENS=24000
EOF
```

Load and inspect the environment without printing secrets:

```bash
set -a
source .env
set +a

python - <<'PY'
import os
print('WATSONX_URL =', repr(os.getenv('WATSONX_URL')))
print('WATSONX_PROJECT_ID =', repr(os.getenv('WATSONX_PROJECT_ID')))
print('PROJECT_ID =', repr(os.getenv('PROJECT_ID')))
print('API key configured =', bool(os.getenv('WATSONX_API_KEY')))
PY
```

If you see a space like this, fix `.env`:

```text
'https://us-south.ml.cloud.ibm.com '
```

It must be:

```text
'https://us-south.ml.cloud.ibm.com'
```

If watsonx returns `container_not_found`, the API key cannot access the project ID in that region. Fix the project ID or region before rerunning the build.

## Install the governed tools

The build needs three local commands:

```text
mdesign
mb
gitpilot
```

They must come from `.venv/bin`, not from `~/.local/bin`.

The final Makefile should create the virtual environment with `uv`, clone/install the tools, and verify command paths.

Run:

```bash
make install
```

Then verify:

```bash
which mdesign
which mb
which gitpilot

.venv/bin/python - <<'PY'
import agent_generator.mb
import litellm
import tokenizers
print('governed Python imports ok')
PY
```

Expected command paths:

```text
/mnt/c/workspace/contract-quest-watsonx/.venv/bin/mdesign
/mnt/c/workspace/contract-quest-watsonx/.venv/bin/mb
/mnt/c/workspace/contract-quest-watsonx/.venv/bin/gitpilot
```

## Clone or prepare the game repository

For the finished project:

```bash
git clone https://github.com/ruslanmv/contract-quest-watsonx
cd contract-quest-watsonx
```

For a fresh local reproduction, create the project root and add your `Makefile`, `.env`, and `build.sh`:

```bash
mkdir -p contract-quest-watsonx
cd contract-quest-watsonx
mkdir -p design frontend scripts tests
```

The important point is that `build.sh` is the orchestrator. It is not only a convenience script; it is the executable summary of the workflow.

## The complete end-to-end command path

After `.env`, `Makefile`, and `build.sh` are ready, the full reproduction is:

```bash
dos2unix Makefile build.sh .env
chmod +x build.sh
make install
make build
```

`make build` should call:

```bash
PATH="$PWD/.venv/bin:$PATH" PYTHONNOUSERSITE=1 REQUIRE_GOVERNED_TOOLS=1 ./build.sh from-zero
```

The script then runs this pipeline:

```text
load .env
force .venv command isolation
validate governed inputs
check GitPilot generate support
check watsonx/GitPilot config
run Matrix Designer blueprint/design/validate/export
normalize Matrix Builder export
reset Matrix Builder state
run npm install
run Matrix Designer-exported batches
verify each implementation batch
optionally deploy to Vercel
write EVIDENCE.md
```

## Stage 1 — Matrix Designer creates the design

The script starts with Matrix Designer.

Manual equivalent:

```bash
mdesign blueprints \
  --idea "Contract Quest: a premium static browser platformer rebuilt through Matrix Designer, Matrix Builder, GitPilot, and IBM watsonx.ai governance." \
  -o design/blueprint.json

mdesign design \
  --idea "Contract Quest: a premium static browser platformer rebuilt through Matrix Designer, Matrix Builder, GitPilot, and IBM watsonx.ai governance." \
  --blueprint design/blueprint.json \
  -o design/contract-quest-design-bundle.json

mdesign validate design/contract-quest-design-bundle.json

mdesign export \
  design/contract-quest-design-bundle.json \
  -o design/contract-quest-mb-export.json
```

The design bundle captures the player promise, visual style, architecture, entity contracts, audio constraints, acceptance criteria, and batch roadmap.

## Stage 2 — Normalize the Matrix Builder export

Matrix Designer's export is normalized into a Matrix Builder plan with governed batches.

The final plan has twelve batches:

| Batch | Purpose |
|---|---|
| `D-1-design` | Validate the design bundle and export. |
| `00-contract-scaffold` | Create contracts, package scripts, Vercel config, scaffold, verifier, and README. |
| `01-framework` | Build the runtime foundation and viewport invariant. |
| `02-art` | Add deterministic original art. |
| `03-levels` | Add the level and story model. |
| `04-hero` | Add movement, physics, death, and respawn. |
| `05-rewards-gates` | Add collectibles, panels, HUD, and Matrix Gates. |
| `06-enemies-powerups` | Add enemies, damage, Shield, and Double Jump. |
| `07-boss` | Add the Rogue Architect and final gate lock. |
| `08-audio-polish` | Add lazy WebAudio music, SFX, particles, and atmosphere. |
| `09-mobile-a11y` | Add touch controls, accessibility, and responsive UI. |
| `10-evidence-release` | Add README, evidence, release checks, and tutorial material. |

This normalization also records that implementation uses watsonx through GitPilot, while the deployed game remains static and credential-free.

## Stage 3 — Matrix Builder initializes governance

Manual equivalent:

```bash
rm -rf .mb
mb init \
  "Contract Quest: a premium static browser platformer rebuilt through Matrix Designer, Matrix Builder, GitPilot, and IBM watsonx.ai governance." \
  --quality standard \
  --title "Contract Quest"
```

The script removes stale `.mb` state and generated frontend outputs before starting a clean governed build.

## Stage 4 — D-1 validates design only

This is one of the most important lessons.

`D-1-design` is not an implementation batch. It must not call `mb next`, and it must not call GitPilot.

Manual equivalent:

```bash
mdesign validate design/contract-quest-design-bundle.json
mdesign export design/contract-quest-design-bundle.json -o design/contract-quest-mb-export.json
```

The script then normalizes the export and moves on.

This avoids a class of failures where Matrix Builder tries to invent implementation tasks for a design-only goal.

## Stage 5 — Implementation batches run through GitPilot

Every real implementation batch follows this pattern:

```text
assemble batch prompt
start Matrix Builder batch
run GitPilot generation
run npm verification
run smoke test when applicable
run mb check only on allowed batch files
record evidence
```

The key command is:

```bash
gitpilot generate -m "$(cat .build/evidence/prompts/<batch-id>.md)" -o .
```

Some GitPilot builds support `--prompt-file`. The production script detects that automatically. If supported, it uses:

```bash
gitpilot generate --prompt-file .build/evidence/prompts/<batch-id>.md -o .
```

Otherwise, it falls back to `-m`.

## Stage 6 — Verification gates

Each batch must prove itself.

For most batches:

```bash
npm install
npm run build
npm run verify
npm run smoke
mb check --changed <allowed files only>
```

For `00-contract-scaffold`, the script skips both smoke and `mb check`. This batch creates the contract files themselves (`MATRIX_*`, `package.json`, `vercel.json`, and the static verifier), so smoke has nothing to run yet and `mb check` would be circular — it would reject the very files the batch just generated (`RMD-002` / `RMD-107`). The implementation batches (`01`–`10`) keep the full gate.

For `D-1-design`, the script skips npm, GitPilot, `mb next`, and `mb check` because the correct proof is `mdesign validate`.

## Stage 7 — Why `mb check --changed` matters

Do not run this inside the governed batch loop:

```bash
mb check --repo .
```

That scans everything, including local tool checkouts:

```text
.tools/agent-generator
.tools/matrix-builder
.tools/matrix-designer
.venv
node_modules
```

The production script computes allowed files for each batch and runs:

```bash
mb check --changed file1 file2 file3
```

That prevents false failures like:

```text
SEC-001: Possible secret committed in .tools/...
RMD-116: New dependency without an approval record
RMD-107: Changed a file outside the allowed change scope
```

Those messages are useful when they refer to project code. They are noise when caused by local tool checkouts.

## Stage 8 — Repairs are evidence-driven

If a real implementation batch fails, the script creates a repair prompt.

Repair categories:

| Repair | Trigger |
|---|---|
| Repair D | Design schema or bundle validation failure. |
| Repair A | Runtime build or static verification failure. |
| Repair B | Viewport, layout, mobile, screenshot, or smoke failure. |
| Repair C | Gameplay, audio, progression, boss, gate, or Matrix check failure. |

A repair is not a random retry. It is a new bounded prompt that says:

```text
Repair only within the original batch allow-list.
Preserve all working features.
Do not print, request, or commit secrets.
Record no invented Matrix commit IDs.
```

## Stage 9 — Evidence is written automatically

At the end, or on failure, the script writes:

```text
EVIDENCE.md
```

It records:

- generation timestamp;
- provider and model;
- GitPilot config directory;
- design bundle path;
- Matrix Builder export path;
- batch list;
- batch outcomes;
- command logs;
- screenshots if available;
- known limitations.

This keeps the blog honest. If a test did not run, the evidence should say so.

## Common errors and fixes

### `ModuleNotFoundError: No module named 'agent_generator'`

Meaning: the wrong `mb` command is running.

Fix:

```bash
make install
which mb
```

Expected:

```text
$PWD/.venv/bin/mb
```

### `/mnt/.../.venv/bin/uv: No such file or directory`

Meaning: `.venv/bin` was prepended before recreating `.venv`.

Fix the Makefile to use the external `uv` path when creating and installing into the virtual environment.

### `provenance.provider was unexpected`

Meaning: the design bundle was modified with schema-invalid metadata.

Fix: keep provider/model in `contract-quest-mb-export.json`, not in `provenance` inside the validated design bundle.

### `SEC-001` inside `.tools/`

Meaning: Matrix Builder scanned tool checkouts.

Fix:

```bash
mb check --changed <allowed batch files>
```

and ignore local tool directories:

```bash
cat >> .gitignore <<'EOF'
.tools/
.venv/
.build/
node_modules/
dist/
test-results/
playwright-report/
.pytest_cache/
EOF
```

### `Segmentation fault` after D-1 export

Meaning: D-1 was still flowing into Matrix Builder checks or implementation behavior.

Fix: skip `mb next`, GitPilot, and `mb check` for `D-1-design`.

### `container_not_found` from watsonx

Meaning: the API key cannot access the configured project ID in that watsonx region.

Fix `.env`:

```bash
WATSONX_URL=https://us-south.ml.cloud.ibm.com
WATSONX_PROJECT_ID=your-real-project-id
PROJECT_ID=your-real-project-id
```

Then reset project-local GitPilot config:

```bash
rm -rf .build/gitpilot_config
```

## Definition of done

Contract Quest is ready to publish only when all of these are true:

- Matrix Designer Design Bundle exists.
- `mdesign validate` passes.
- Matrix Builder batches are derived from the exported design.
- `npm run build` passes.
- `npm run verify` passes.
- `npm run smoke` passes or the evidence explains why it could not run.
- Matrix checks run only on scoped files.
- Public Vercel deployment returns HTTP 200 if deployment is enabled.
- The final game needs no runtime watsonx credentials.
- The boss gate cannot be skipped.
- WebAudio starts only after user gesture.
- Evidence supports every production claim.

## The shortest reproduction

Once the repository contains the production `Makefile` and `build.sh`, the whole tutorial becomes this:

```bash
git clone https://github.com/ruslanmv/contract-quest-watsonx
cd contract-quest-watsonx
cp .env.example .env
# Edit .env with your real watsonx values.

dos2unix Makefile build.sh .env
chmod +x build.sh
make install
make build
```

If you are replacing only the build script:

```bash
cp build_ready_production_v2.sh build.sh
dos2unix build.sh
chmod +x build.sh
make build
```

## The production build script

The script below is the executable summary of this article. It loads the environment, isolates `.venv`, validates the Matrix Designer bundle, normalizes the Matrix Builder export, skips implementation behavior for D-1, runs GitPilot for real implementation batches, checks only allowed files, and writes evidence.

Save it as `build.sh`:
<script src="https://gist.github.com/ruslanmv/b0ab0252cd8e10f29fb29a772e4a35d3.js"></script>


## Final word

Contract Quest begins as a game idea, but it becomes a repeatable engineering pattern.

The game is the visible artifact. The pipeline is the real product.

> Matrix Designer defines the game we want. Matrix Builder governs how the AI may build it. GitPilot executes the work. watsonx.ai generates the code. Verification decides what is safe to publish.

That is how we move from prompt-driven demos toward reproducible AI software production.
