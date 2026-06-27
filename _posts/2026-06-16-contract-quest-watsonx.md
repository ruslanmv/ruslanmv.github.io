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

````bash
#!/usr/bin/env bash
# -E (errtrace): the ERR trap is inherited by functions/subshells, so on_error fires
# and writes evidence when a governed step fails inside a function (e.g. gitpilot/mb),
# instead of exiting silently.
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

VENV_DIR="$ROOT_DIR/.venv"
VENV_BIN="$VENV_DIR/bin"

prepend_venv_path() {
  if [ -d "$VENV_BIN" ]; then
    case ":$PATH:" in
      *":$VENV_BIN:"*) ;;
      *) export PATH="$VENV_BIN:$PATH" ;;
    esac
  fi
  export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
  hash -r 2>/dev/null || true
}

prepend_venv_path

MODE="${1:-verify}"
STRICT_GENERATION=0
IDEA="${IDEA:-Contract Quest: a premium static browser platformer rebuilt through Matrix Designer, Matrix Builder, GitPilot, and IBM watsonx.ai governance.}"
PROJECT_TITLE="${PROJECT_TITLE:-Contract Quest}"
DESIGN_BLUEPRINT="design/blueprint.json"
DESIGN_BUNDLE="design/contract-quest-design-bundle.json"
MATRIX_EXPORT="design/contract-quest-mb-export.json"
EVIDENCE_FILE="EVIDENCE.md"
EVIDENCE_DIR=".build/evidence"
MAX_REPAIR_ATTEMPTS="${MAX_REPAIR_ATTEMPTS:-2}"
PUBLISHED_URL=""
KNOWN_LIMITATIONS=()
BATCH_EVIDENCE=()
COMMAND_EVIDENCE=()
EVIDENCE_WRITTEN=0

section() {
  printf '\n==> %s\n' "$1"
}

record_command() {
  COMMAND_EVIDENCE+=("$1|$2|$3")
}

record_batch() {
  BATCH_EVIDENCE+=("$1|$2|$3|$4")
}

trim_value() {
  local value="${1:-}"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "$value"
}

ensure_local_gitignore() {
  local gitignore=".gitignore"
  touch "$gitignore"

  local entry
  for entry in \
    ".tools/" \
    ".venv/" \
    ".build/" \
    "node_modules/" \
    "dist/" \
    "test-results/" \
    "playwright-report/" \
    ".pytest_cache/"; do
    if ! grep -qxF "$entry" "$gitignore"; then
      printf '%s\n' "$entry" >> "$gitignore"
    fi
  done
}

load_env_file() {
  prepend_venv_path

  if [ -f ".env" ]; then
    while IFS= read -r line || [ -n "$line" ]; do
      case "$line" in
        ''|'#'*) continue ;;
        export\ *) line="${line#export }" ;;
      esac
      case "$line" in
        *=*) ;;
        *) continue ;;
      esac
      local key="${line%%=*}"
      case "$key" in
        ''|*[!A-Za-z0-9_]*) continue ;;
      esac
      if [ -z "${!key+x}" ]; then
        local value="${line#*=}"
        if [ "${value:0:1}" = '"' ] && [ "${value: -1}" = '"' ]; then
          value="${value:1:${#value}-2}"
        elif [ "${value:0:1}" = "'" ] && [ "${value: -1}" = "'" ]; then
          value="${value:1:${#value}-2}"
        fi
        export "$key=$value"
      fi
    done < ".env"
  fi

  if [ -n "${PROJECT_ID:-}" ] && [ -z "${WATSONX_PROJECT_ID:-}" ]; then
    export WATSONX_PROJECT_ID="$PROJECT_ID"
  fi

  if [ -n "${WATSXON_API_KEY:-}" ] && [ -z "${WATSONX_API_KEY:-}" ]; then
    export WATSONX_API_KEY="$WATSXON_API_KEY"
  fi

  if [ -n "${WATSONX_API_KEY:-}" ] && [ -z "${WATSONX_APIKEY:-}" ]; then
    export WATSONX_APIKEY="$WATSONX_API_KEY"
  fi

  export GITPILOT_PROVIDER="${GITPILOT_PROVIDER:-watsonx}"
  export GITPILOT_MODEL_PROVIDER="${GITPILOT_MODEL_PROVIDER:-watsonx}"

  export WATSONX_URL="${WATSONX_URL:-https://us-south.ml.cloud.ibm.com}"
  export WATSONX_BASE_URL="${WATSONX_BASE_URL:-$WATSONX_URL}"
  export WATSONX_URL="$(trim_value "$WATSONX_URL")"
  export WATSONX_BASE_URL="$(trim_value "$WATSONX_BASE_URL")"
  export WATSONX_URL="${WATSONX_URL%/}"
  export WATSONX_BASE_URL="${WATSONX_BASE_URL%/}"

  export GITPILOT_WATSONX_MODEL="${GITPILOT_WATSONX_MODEL:-openai/gpt-oss-120b}"
  export GITPILOT_WATSONX_MODEL="$(trim_value "$GITPILOT_WATSONX_MODEL")"
  export GITPILOT_MODEL="${GITPILOT_MODEL:-$GITPILOT_WATSONX_MODEL}"
  export GITPILOT_MODEL_ID="${GITPILOT_MODEL_ID:-$GITPILOT_WATSONX_MODEL}"
  export WATSONX_MODEL_ID="${WATSONX_MODEL_ID:-$GITPILOT_WATSONX_MODEL}"
  export WATSONX_MODEL="${WATSONX_MODEL:-$GITPILOT_WATSONX_MODEL}"
  export LITELLM_MODEL="${LITELLM_MODEL:-watsonx/$GITPILOT_WATSONX_MODEL}"

  export MB_PROVIDER="${MB_PROVIDER:-watsonx}"
  export MB_MODEL="${MB_MODEL:-$GITPILOT_WATSONX_MODEL}"

  # Matrix Designer currently falls back to deterministic planning if no agentic
  # backend is usable. For this governed build we want stable deterministic design
  # followed by watsonx generation through GitPilot, so disable the agentic backend
  # by default. Override MATRIX_DESIGNER_BACKEND manually if you intentionally want
  # an agentic Matrix Designer run.
  export MATRIX_DESIGNER_BACKEND="${MATRIX_DESIGNER_BACKEND:-off}"
  export MATRIX_DESIGNER_PROVIDER="${MATRIX_DESIGNER_PROVIDER:-watsonx}"
  export MATRIX_DESIGNER_MODEL="${MATRIX_DESIGNER_MODEL:-$GITPILOT_WATSONX_MODEL}"

  export GITPILOT_MAX_TOKENS="${GITPILOT_MAX_TOKENS:-24000}"
  export GITPILOT_MAX_TOKENS="$(trim_value "$GITPILOT_MAX_TOKENS")"
  export OTEL_SDK_DISABLED="${OTEL_SDK_DISABLED:-true}"
  export CREWAI_DISABLE_TELEMETRY="${CREWAI_DISABLE_TELEMETRY:-true}"
  # Never show the interactive "view execution traces?" prompt during a governed build.
  export CREWAI_TRACING_ENABLED="${CREWAI_TRACING_ENABLED:-false}"
  export LITELLM_LOG="${LITELLM_LOG:-ERROR}"

  # Force project-local GitPilot config. Do not use ~/.gitpilot during this build,
  # because global config can point to Ollama or stale model settings.
  export GITPILOT_CONFIG_DIR="$ROOT_DIR/.build/gitpilot_config"
  mkdir -p "$GITPILOT_CONFIG_DIR"

  prepend_venv_path
}

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    printf 'Missing required command: %s\n' "$1" >&2
    return 1
  fi
}

need_env() {
  local name="$1"
  if [ -z "${!name:-}" ]; then
    printf 'Missing required environment variable: %s\n' "$name" >&2
    return 1
  fi
}

command_path() {
  command -v "$1" 2>/dev/null || true
}

require_venv_command() {
  local cmd="$1"
  local path
  path="$(command_path "$cmd")"
  if [ -z "$path" ]; then
    printf 'Missing required command: %s\n' "$cmd" >&2
    return 1
  fi

  case "$path" in
    "$VENV_BIN"/*)
      printf 'Using %s: %s\n' "$cmd" "$path"
      ;;
    *)
      printf 'Wrong %s command selected: %s\n' "$cmd" "$path" >&2
      printf 'Expected %s to come from: %s\n' "$cmd" "$VENV_BIN" >&2
      printf 'Fix by running: make install\n' >&2
      return 1
      ;;
  esac
}

# Probe that a governed CLI actually runs (its package imports). Catches the case
# where the entry point exists in .venv/bin but the package is broken/missing.
require_tool_runs() {
  local cmd="$1"
  if "$cmd" --help >/dev/null 2>&1 || "$cmd" --version >/dev/null 2>&1; then
    return 0
  fi
  printf '%s is installed in .venv but failed to run.\n' "$cmd" >&2
  printf 'Its Python package is probably not importable (e.g. matrix-designer/matrix-builder install is broken).\n' >&2
  printf 'Reinstall the governed tools, then retry:\n' >&2
  printf '       make install\n' >&2
  printf 'Diagnose with:\n' >&2
  printf '       %s --help\n' "$cmd" >&2
  return 1
}

ensure_evidence_dir() {
  mkdir -p "$EVIDENCE_DIR" design frontend
}

run_capture() {
  prepend_venv_path

  local label="$1"
  shift
  local slug log status
  slug="$(printf '%s' "$label" | tr '[:upper:] /:' '[:lower:]---' | tr -cd 'a-z0-9._-')"
  log="$EVIDENCE_DIR/${slug}.log"

  section "$label"

  set +e
  # stdin from /dev/null: the governed tools are non-interactive here, and some
  # (CrewAI under mdesign/gitpilot) otherwise prompt "view execution traces?" and
  # then crash at shutdown on a stdin/daemon-thread lock. EOF makes them skip it.
  "$@" </dev/null 2>&1 | tee "$log"
  status=${PIPESTATUS[0]}
  set -e

  record_command "$label" "$status" "$log"
  return "$status"
}

run_shell_capture() {
  prepend_venv_path

  local label="$1"
  shift
  local slug log status
  slug="$(printf '%s' "$label" | tr '[:upper:] /:' '[:lower:]---' | tr -cd 'a-z0-9._-')"
  log="$EVIDENCE_DIR/${slug}.log"

  section "$label"

  set +e
  env PATH="$PATH" PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}" bash -c "$*" </dev/null 2>&1 | tee "$log"
  status=${PIPESTATUS[0]}
  set -e

  record_command "$label" "$status" "$log"
  return "$status"
}

npm_install_if_needed() {
  if [ ! -f package.json ]; then
    KNOWN_LIMITATIONS+=("No package.json was present, so npm install was skipped.")
    return 0
  fi

  need_cmd npm

  if [ -f package-lock.json ]; then
    run_capture "npm install" npm install
  else
    run_capture "npm install" npm install --package-lock=false
  fi
}

run_npm_script_if_present() {
  local script="$1"
  [ -f package.json ] || return 1
  node -e "const p=require('./package.json'); process.exit(p.scripts && p.scripts['$script'] ? 0 : 1)" 2>/dev/null
}

verify_existing_artifacts() {
  npm_install_if_needed

  if run_npm_script_if_present build; then
    run_capture "npm run build" npm run build
  fi

  if run_npm_script_if_present verify; then
    run_capture "npm run verify" npm run verify
  fi

  if run_npm_script_if_present smoke; then
    run_capture "npm run smoke" npm run smoke
  else
    KNOWN_LIMITATIONS+=("No npm smoke script is defined.")
  fi
}

require_governed_inputs() {
  section "Check governed build inputs"

  prepend_venv_path

  local missing=0

  need_cmd node || missing=1
  need_cmd npm || missing=1

  require_venv_command mdesign || missing=1
  require_venv_command mb || missing=1
  require_venv_command gitpilot || missing=1

  # A command can be present in .venv/bin yet fail to import its package (e.g. a
  # broken matrix-designer install: `mdesign` exists but `matrix_designer` is not
  # importable). Probe that each tool actually runs, so the failure is a clear
  # message here instead of a mid-build ModuleNotFoundError traceback.
  require_tool_runs mdesign || missing=1
  require_tool_runs mb || missing=1
  require_tool_runs gitpilot || missing=1

  need_env GITPILOT_PROVIDER || missing=1
  need_env WATSONX_API_KEY || missing=1
  need_env WATSONX_PROJECT_ID || missing=1
  need_env WATSONX_URL || missing=1
  need_env GITPILOT_WATSONX_MODEL || missing=1
  need_env GITPILOT_MAX_TOKENS || missing=1

  if [ "$missing" -ne 0 ]; then
    printf 'Strict governed build cannot continue until all commands and environment variables are available.\n' >&2
    return 1
  fi

  printf 'Provider: %s\n' "$GITPILOT_PROVIDER"
  printf 'watsonx URL: %s\n' "$WATSONX_URL"
  printf 'Model: %s\n' "$GITPILOT_WATSONX_MODEL"
  printf 'Max tokens: %s\n' "$GITPILOT_MAX_TOKENS"
  printf 'GitPilot config dir: %s\n' "$GITPILOT_CONFIG_DIR"
  printf 'API key: configured (value hidden)\n'
  printf 'Project ID: configured (value hidden)\n'

  if [ -n "${WATSONX_API_KEY:-}" ]; then
    printf 'Reminder: rotate any watsonx API key that was pasted into chat or logs before using this final build.\n'
  fi
}

GP_GENERATE_MODE="${GP_GENERATE_MODE:-}"

detect_gitpilot_generate_mode() {
  if [ -n "$GP_GENERATE_MODE" ]; then
    return 0
  fi

  prepend_venv_path

  local help
  help="$(gitpilot generate --help 2>/dev/null || true)"

  if printf '%s' "$help" | grep -q -- '--prompt-file'; then
    GP_GENERATE_MODE="prompt-file"
  else
    GP_GENERATE_MODE="message"
  fi
}

check_gitpilot_connection() {
  section "GitPilot CLI health check"

  prepend_venv_path

  require_venv_command gitpilot

  gitpilot --help >/dev/null

  if ! gitpilot generate --help >/dev/null 2>&1; then
    printf 'GitPilot is installed, but this build needs the local "gitpilot generate" command.\n' >&2
    printf 'Install the local-generate GitPilot build into .venv, then rerun make install.\n' >&2
    gitpilot --help >&2
    return 1
  fi

  detect_gitpilot_generate_mode
  printf 'GitPilot local generate command is available (prompt mode: %s).\n' "$GP_GENERATE_MODE"
}

check_watsonx_connection() {
  section "watsonx/GitPilot configuration health check"

  prepend_venv_path

  printf 'PATH: %s\n' "$PATH"
  printf 'GitPilot config dir: %s\n' "$GITPILOT_CONFIG_DIR"

  if command -v gitpilot >/dev/null 2>&1; then
    run_capture "gitpilot config" gitpilot config || true
    run_capture "gitpilot doctor offline" gitpilot doctor --workspace . --offline || true
  fi
}

run_design_repair() {
  local attempt="$1"
  local reason="$2"
  local prompt_file="$EVIDENCE_DIR/repair-d-design-${attempt}.md"

  section "Repair D: design bundle/schema repair attempt $attempt"

  cat > "$prompt_file" <<EOF_REPAIR
Repair D: Matrix Designer design bundle/schema repair.

Reason:
$reason

Repair only these files if needed:
- $DESIGN_BUNDLE
- $MATRIX_EXPORT
- $DESIGN_BLUEPRINT

Do not print or request secrets. Make the design bundle validate with the installed Matrix Designer schema.
EOF_REPAIR

  run_gitpilot_prompt "$prompt_file" "D design repair $attempt"
}

ensure_governed_design_export() {
  section "Normalize Matrix Designer output for watsonx Matrix Builder batches"

  node - "$DESIGN_BUNDLE" "$MATRIX_EXPORT" "$DESIGN_BLUEPRINT" "$PROJECT_TITLE" "$GITPILOT_WATSONX_MODEL" <<'NODE'
const fs = require('fs');
const path = require('path');

const [designPath, exportPath, blueprintPath, projectTitle, model] = process.argv.slice(2);

const batches = [
  ['D-1-design','Design Bundle',['design/contract-quest-design-bundle.json','design/contract-quest-mb-export.json','design/README.md'],'mdesign validate passes'],
  ['00-contract-scaffold','Contract reset and scaffold',['MATRIX_BLUEPRINT.yaml','MATRIX_TASKS.md','MATRIX_ACCEPTANCE_CRITERIA.md','MATRIX_VALIDATION.md','MATRIX_ALLOWED_CHANGES.md','package.json','vercel.json','frontend/index.html','scripts/verify-static.js','tests/**','README.md'],'npm run build and npm run verify pass'],
  ['01-framework','Framework foundation and viewport invariant',['frontend/index.html','scripts/verify-static.js','tests/**'],'DPR 1 and DPR 2 debug state is sane'],
  ['02-art','Art direction and deterministic asset pipeline',['frontend/index.html','scripts/verify-static.js','tests/**'],'original deterministic art markers remain'],
  ['03-levels','Level and story model',['frontend/index.html','tests/**','README.md'],'three-level campaign metadata exists'],
  ['04-hero','Hero movement, physics, death, and respawn',['frontend/index.html','tests/**'],'jump edge and pit/respawn markers exist'],
  ['05-rewards-gates','Rewards, panels, HUD, and Matrix Gates',['frontend/index.html','tests/**'],'collectibles and gates are explicit'],
  ['06-enemies-powerups','Enemies and power-ups',['frontend/index.html','tests/**'],'central damage and power-up behavior remains testable'],
  ['07-boss','Rogue Architect boss and final validation',['frontend/index.html','tests/**'],'final gate is locked while bossAlive is true'],
  ['08-audio-polish','WebAudio music, SFX, and atmosphere',['frontend/index.html','scripts/verify-static.js','tests/**'],'audio initializes only after user gesture'],
  ['09-mobile-a11y','Mobile controls, accessibility, and UX polish',['frontend/index.html','tests/**'],'touch start and jump work'],
  ['10-evidence-release','QA, evidence, tutorial, and release package',['README.md','EVIDENCE.md','build.sh','scripts/**','tests/**'],'claims match evidence']
];

const roadmap = batches.map(([id, title, allowed_files, acceptance], index) => ({
  id: `batch-${String(index + 1).padStart(2, '0')}`,
  name: title,
  purpose: `Implement and verify the ${title} scope for Contract Quest.`,
  allowed_files,
  acceptance: [acceptance]
}));

const requiredFunctional = [
  'npm run build passes and validates the static game artifact.',
  'npm run verify passes and checks design, Matrix export, Vercel configuration, and frontend markers.',
  'npm run smoke passes where Playwright and browser binaries are installed, or reports the environment limitation without requiring runtime credentials.',
  'The game exposes desktop and mobile controls and supports DPR 1 and DPR 2 viewport states.',
  'The final Matrix Gate cannot be completed while bossAlive is true.',
  'Audio initializes lazily only after a user gesture and uses generated WebAudio rather than external audio files.',
  'No watsonx API key or project credential is committed or required by the deployed static game.'
];

let design = {};
try {
  design = JSON.parse(fs.readFileSync(designPath, 'utf8'));
} catch {}

design = {
  schema_version: design.schema_version || 'matrix.designer.bundle/v1',
  design_id: design.design_id || 'contract-quest-watsonx',
  project: projectTitle,
  source: design.source || { type: 'repo', path: designPath },
  provenance: {
    ...(design.provenance || {}),
    created_by: 'Matrix Designer governed workflow',
    ai_assisted: true
  },
  goal_analysis: {
    real_goal: 'Build Contract Quest as a premium static browser platformer that can be reproduced through a governed Matrix Designer -> Matrix Builder -> GitPilot -> IBM watsonx.ai workflow and deployed as static frontend assets.',
    complexity: 'large',
    ...(design.goal_analysis || {})
  },
  framework_decision: design.framework_decision || {
    stack: ['HTML5 Canvas', 'vanilla JavaScript', 'static Vercel hosting'],
    rationale: 'Keep the game static, reproducible, and free of runtime credentials while Matrix Builder and GitPilot use watsonx.ai only during generation.'
  },
  visual_target: design.visual_target || {
    style: 'warm governed arcade pixel quest with deterministic original art, contract coins, validation gems, RMD panels, Matrix Gates, and a Rogue Architect boss.'
  },
  architecture: design.architecture || {
    systems: ['Viewport', 'Input', 'Physics', 'LevelModel', 'HUD', 'AudioBus', 'Particles', 'Progression', 'DebugHooks']
  },
  contracts: design.contracts || {},
  acceptance: {
    ...(design.acceptance || {}),
    functional: requiredFunctional
  },
  batch_roadmap: roadmap
};

const exported = {
  schema_version: '1.0',
  source_design_bundle: designPath,
  matrix_builder: {
    project: projectTitle,
    provider: 'watsonx',
    model,
    default_verification: ['npm run build', 'npm run verify', 'npm run smoke', 'mb check --changed <allowed files>'],
    batches: batches.map(([id, title, allowed_files, acceptance]) => ({
      id,
      title,
      allowed_files,
      acceptance
    }))
  },
  shared_prompt_header: 'You are implementing a Matrix Designer-designed Contract Quest production batch with GitPilot on IBM watsonx.ai. Follow the design bundle and Matrix export, stay in the allowed files, preserve logical viewport coordinates, jumpPressed/jumpHeld separation, boss-gate lock, lazy generated WebAudio, and evidence-backed claims.'
};

fs.mkdirSync(path.dirname(designPath), { recursive: true });
fs.writeFileSync(designPath, JSON.stringify(design, null, 2) + '\n');
fs.writeFileSync(exportPath, JSON.stringify(exported, null, 2) + '\n');

if (!fs.existsSync(blueprintPath)) {
  fs.writeFileSync(
    blueprintPath,
    JSON.stringify({
      project: 'contract-quest-watsonx',
      provider_rule: 'watsonx-only',
      generation: { provider: 'watsonx', model }
    }, null, 2) + '\n'
  );
}

console.log(`normalized ${roadmap.length} governed batches for provider watsonx and model ${model}`);
NODE
}

generate_design() {
  ensure_evidence_dir

  section "Matrix Designer: blueprint, design, validate, export"

  run_capture "mdesign blueprints" \
    mdesign blueprints --idea "$IDEA" -o "$DESIGN_BLUEPRINT"

  run_capture "mdesign design" \
    mdesign design --idea "$IDEA" --blueprint "$DESIGN_BLUEPRINT" -o "$DESIGN_BUNDLE"

  local attempt=0

  until run_capture "mdesign validate" mdesign validate "$DESIGN_BUNDLE"; do
    if [ "$attempt" -ge "$MAX_REPAIR_ATTEMPTS" ]; then
      return 1
    fi

    attempt=$((attempt + 1))
    run_design_repair "$attempt" "mdesign validate failed; see $EVIDENCE_DIR/mdesign-validate.log"
  done

  run_capture "mdesign export" \
    mdesign export "$DESIGN_BUNDLE" -o "$MATRIX_EXPORT"

  ensure_governed_design_export
}

reset_matrix_builder_state() {
  section "Reset Matrix Builder state and generated outputs"

  rm -rf .mb
  rm -f frontend/index.html
  rm -rf dist test-results playwright-report

  run_capture "mb init" \
    mb init "$IDEA" --quality standard --title "$PROJECT_TITLE"
}

batch_ids_from_export() {
  node -e "const fs=require('fs'); const plan=JSON.parse(fs.readFileSync(process.argv[1],'utf8')); const batches=plan.matrix_builder?.batches; if(!Array.isArray(batches)||!batches.length){throw new Error('Matrix export contains no batches')} for (const batch of batches) console.log(batch.id);" "$MATRIX_EXPORT"
}

assemble_batch_prompt() {
  local batch_id="$1"
  local prompt_file="$2"
  local meta_file="$3"

  node - "$MATRIX_EXPORT" "$DESIGN_BUNDLE" "$batch_id" "$prompt_file" "$meta_file" <<'NODE'
const fs = require('fs');

const [exportPath, designPath, batchId, promptPath, metaPath] = process.argv.slice(2);

const plan = JSON.parse(fs.readFileSync(exportPath, 'utf8'));
const designText = fs.readFileSync(designPath, 'utf8');
const exportText = fs.readFileSync(exportPath, 'utf8');

const batch = plan.matrix_builder.batches.find((item) => item.id === batchId);
if (!batch) throw new Error(`Unknown batch ${batchId}`);

const allowed = Array.isArray(batch.allowed_files) ? batch.allowed_files : [];
const acceptance = Array.isArray(batch.acceptance) ? batch.acceptance : [batch.acceptance || 'Batch acceptance must pass.'];

const current = [];
for (const file of allowed) {
  if (file.includes('*')) continue;
  if (fs.existsSync(file) && fs.statSync(file).isFile()) {
    current.push(`\n## ${file}\n\`\`\`\n${fs.readFileSync(file, 'utf8')}\n\`\`\``);
  }
}

const prompt = `${plan.shared_prompt_header || 'You are implementing a Matrix Designer-designed Contract Quest production batch.'}

# Governed batch
Batch ID: ${batch.id}
Batch title: ${batch.title || batch.id}
Batch purpose: ${batch.purpose || batch.title || batch.id}

# Allowed files
${allowed.map((f) => `- ${f}`).join('\n')}

# Acceptance criteria
${acceptance.map((a) => `- ${a}`).join('\n')}

# Required verification commands
- npm install
- npm run build
- npm run verify
- npm run smoke
- mb check --changed <allowed files>

# Output contract
- Output only files in the allowed files list.
- Do not modify files outside the batch allow-list.
- Preserve all previous working features.
- If frontend/index.html is allowed or being extended, output the complete frontend/index.html.
- Do not output partial code for single-file canvas changes.
- Do not print, request, or commit secrets.

# Design Bundle
\`\`\`json
${designText}
\`\`\`

# Matrix Builder export
\`\`\`json
${exportText}
\`\`\`

# Current allowed file content
${current.length ? current.join('\n') : 'No existing allowed file content is available for this batch.'}
`;

fs.writeFileSync(promptPath, prompt);
fs.writeFileSync(metaPath, JSON.stringify({
  id: batch.id,
  title: batch.title || batch.id,
  purpose: batch.purpose || '',
  allowed_files: allowed,
  acceptance
}, null, 2));

console.log(`Title: ${batch.title || batch.id}`);
console.log(`Allowed files: ${allowed.join(', ')}`);
console.log(`Acceptance: ${acceptance.join('; ')}`);
NODE
}

tailor_latest_mb_batch() {
  local title="$1"
  local allowed_json="$2"
  local acceptance_json="$3"
  local latest

  latest="$(find .mb/batches -maxdepth 1 -mindepth 1 -type d | sort | tail -1)"

  if [ -z "$latest" ] || [ ! -f "$latest/batch.json" ]; then
    printf 'Matrix Builder did not create a batch.json to tailor.\n' >&2
    return 1
  fi

  node - "$latest/batch.json" "$title" "$allowed_json" "$acceptance_json" <<'NODE'
const fs = require('fs');

const file = process.argv[2];
const title = process.argv[3];
const allowed = JSON.parse(process.argv[4]);
const acceptance = JSON.parse(process.argv[5]);

const batch = JSON.parse(fs.readFileSync(file, 'utf8'));

batch.title = title;
batch.goal_md = title;

batch.plan = batch.plan || {};
batch.plan.title = title;
batch.plan.goal_md = title;
batch.plan.allowed_files = allowed;

if (Array.isArray(batch.plan.tasks) && batch.plan.tasks.length) {
  batch.plan.tasks[0].title = title;
  batch.plan.tasks[0].allowed_files = allowed;
  batch.plan.tasks[0].acceptance_criteria = acceptance;
}

fs.writeFileSync(file, JSON.stringify(batch, null, 2) + "\n");
NODE
}

start_matrix_batch() {
  local title="$1"
  local allowed_json="$2"
  local acceptance_json="$3"

  export MB_TITLE="$title"

  run_shell_capture "mb next ${title}" \
    'mb next "$MB_TITLE"'

  tailor_latest_mb_batch "$title" "$allowed_json" "$acceptance_json"
}

run_gitpilot_prompt() {
  local prompt_file="$1"
  local title="$2"

  export GP_PROMPT_FILE="$prompt_file"

  detect_gitpilot_generate_mode

  if [ "$GP_GENERATE_MODE" = "prompt-file" ]; then
    run_shell_capture "gitpilot generate ${title}" \
      'gitpilot generate --prompt-file "$GP_PROMPT_FILE" -o .'
  else
    run_shell_capture "gitpilot generate ${title}" \
      'gitpilot generate -m "$(cat "$GP_PROMPT_FILE")" -o .'
  fi
}

mb_changed_files_for_batch() {
  local batch_id="$1"

  node - "$MATRIX_EXPORT" "$batch_id" <<'NODE'
const fs = require('fs');
const path = require('path');

const [exportPath, batchId] = process.argv.slice(2);
const plan = JSON.parse(fs.readFileSync(exportPath, 'utf8'));
const batch = (plan.matrix_builder?.batches || []).find((item) => item.id === batchId);

if (!batch) process.exit(0);

const excludedPrefixes = [
  '.tools/',
  '.venv/',
  '.build/',
  'node_modules/',
  'dist/',
  'test-results/',
  'playwright-report/',
  '.pytest_cache/',
  '.git/'
];

const out = new Set();

function normalize(file) {
  return file.replace(/\\/g, '/').replace(/^\.\//, '');
}

function excluded(file) {
  file = normalize(file);
  return excludedPrefixes.some((prefix) => file === prefix.slice(0, -1) || file.startsWith(prefix));
}

function addFile(file) {
  file = normalize(file);
  if (!file || excluded(file)) return;
  if (fs.existsSync(file) && fs.statSync(file).isFile()) {
    out.add(file);
  }
}

function walk(target) {
  target = normalize(target);
  if (!target || excluded(target) || !fs.existsSync(target)) return;

  const stat = fs.statSync(target);
  if (stat.isFile()) {
    addFile(target);
    return;
  }

  if (!stat.isDirectory()) return;

  for (const name of fs.readdirSync(target)) {
    walk(path.join(target, name));
  }
}

for (const raw of batch.allowed_files || []) {
  if (!raw || typeof raw !== 'string') continue;

  const item = normalize(raw);
  if (excluded(item)) continue;

  if (item.includes('**')) {
    walk(item.split('/**')[0]);
  } else if (item.includes('*')) {
    const base = item.split('*')[0].replace(/\/$/, '');
    walk(base || '.');
  } else {
    addFile(item);
  }
}

for (const file of [...out].sort()) {
  console.log(file);
}
NODE
}

run_mb_check_for_batch() {
  local batch_id="$1"
  local changed=()
  local file

  while IFS= read -r file; do
    [ -n "$file" ] || continue
    changed+=("$file")
  done < <(mb_changed_files_for_batch "$batch_id")

  if [ "${#changed[@]}" -eq 0 ]; then
    printf 'No allowed files were found for mb check in batch %s; skipping mb check.\n' "$batch_id"
    KNOWN_LIMITATIONS+=("mb check skipped for $batch_id because no existing allowed files were found.")
    return 0
  fi

  run_capture "mb check ${batch_id}" \
    mb check --changed "${changed[@]}"
}

run_batch_verification() {
  local batch_id="$1"

  if [ "$batch_id" = "D-1-design" ]; then
    run_capture "mdesign validate ${batch_id}" \
      mdesign validate "$DESIGN_BUNDLE"

    run_capture "mdesign export ${batch_id}" \
      mdesign export "$DESIGN_BUNDLE" -o "$MATRIX_EXPORT"

    ensure_governed_design_export

    # D-1 is a design/export validation batch. Do not run Matrix Builder check
    # here: some Matrix Builder versions segfault or scan unrelated tool files
    # when checking design-only metadata. The generated implementation batches
    # run mb check with their own allowed changed files.
    printf 'Skipping mb check for %s: design-only validation already passed.\n' "$batch_id"
    KNOWN_LIMITATIONS+=("mb check skipped for $batch_id because it is a design-only batch validated by mdesign.")

    return
  fi

  npm_install_if_needed || return $?

  # This function runs inside a `... || { ... }` context where `set -e` does not
  # apply, so each step must explicitly stop on failure and return its status so
  # the batch repair loop can react instead of masking earlier failures.
  run_capture "npm run build ${batch_id}" \
    npm run build || return $?

  run_capture "npm run verify ${batch_id}" \
    npm run verify || return $?

  # 00-contract-scaffold generates the contract itself (MATRIX_* files, package.json,
  # vercel.json, the verifier). Smoke needs more runtime than the scaffold has, and
  # mb check is circular for this batch — it would reject the very contract files the
  # batch just created (RMD-002 "forbidden contract file" / RMD-107 "outside scope").
  # Both are skipped for the scaffold; the implementation batches (01-10) keep them.
  if [ "$batch_id" != "00-contract-scaffold" ]; then
    run_capture "npm run smoke ${batch_id}" \
      npm run smoke || return $?
    run_mb_check_for_batch "$batch_id" || return $?
  else
    printf 'Skipping smoke and mb check for %s: the scaffold establishes the contract files.\n' "$batch_id"
    KNOWN_LIMITATIONS+=("smoke and mb check skipped for 00-contract-scaffold because it creates the contract/scaffold files.")
  fi
}

repair_category_for_failure() {
  local failed_label="$1"

  case "$failed_label" in
    *mdesign*) printf 'Repair D = design bundle/schema repair' ;;
    *build*|*verify*) printf 'Repair A = runtime repair' ;;
    *smoke*) printf 'Repair B = viewport/layout repair' ;;
    *mb\ check*) printf 'Repair C = gameplay/audio progression repair' ;;
    *) printf 'Repair A = runtime repair' ;;
  esac
}

run_repair_batch() {
  local batch_id="$1"
  local attempt="$2"
  local failed_label="$3"
  local prompt_file="$4"
  local category repair_prompt

  category="$(repair_category_for_failure "$failed_label")"
  repair_prompt="$EVIDENCE_DIR/${batch_id}-repair-${attempt}.md"

  section "$category for $batch_id (attempt $attempt)"

  # Build the repair prompt as: repair header + the FULL original batch prompt content.
  # Embedding the path alone (the old bug) left the model with no context, so repairs
  # drifted into unrelated generic projects. Using a quoted heredoc + cat keeps the
  # original prompt's backticks/JSON literal (no shell expansion).
  {
    cat <<EOF_REPAIR_HEAD
$category

The previous batch failed at: $failed_label

Repair only within the original batch allow-list. Preserve all working features.
Do not print, request, or commit secrets. Record no invented Matrix commit IDs.
Implement the SAME batch below — do not change direction or invent a new project.

# Original batch prompt (implement this):

EOF_REPAIR_HEAD
    cat "$prompt_file"
  } > "$repair_prompt"

  run_gitpilot_prompt "$repair_prompt" "$batch_id repair $attempt"
}

run_exported_batch() {
  local batch_id="$1"
  local prompt_file meta_file title allowed_json acceptance_json attempt failed_label status

  prompt_file="$EVIDENCE_DIR/prompts/${batch_id}.md"
  meta_file="$EVIDENCE_DIR/prompts/${batch_id}.json"

  mkdir -p "$(dirname "$prompt_file")"

  assemble_batch_prompt "$batch_id" "$prompt_file" "$meta_file"

  title="$(node -e "const m=require('./' + process.argv[1]); console.log(m.title)" "$meta_file")"
  allowed_json="$(node -e "const m=require('./' + process.argv[1]); console.log(JSON.stringify(m.allowed_files || []))" "$meta_file")"
  acceptance_json="$(node -e "const m=require('./' + process.argv[1]); console.log(JSON.stringify(m.acceptance || []))" "$meta_file")"

  attempt=0

  if [ "$batch_id" = "D-1-design" ]; then
    # D-1 is already produced by Matrix Designer. Do not call `mb next` for it:
    # Matrix Builder may synthesize unrelated backend tasks for a design-only goal.
    # We only validate and export the design bundle, then continue to real batches.
    status=0
    failed_label=""

    run_batch_verification "$batch_id" || {
      status=$?
      failed_label="verification for $batch_id"
    }

    if [ "$status" -eq 0 ]; then
      record_batch "$batch_id" "$title" "passed" "$prompt_file"
      return 0
    fi

    record_batch "$batch_id" "$title" "failed" "$prompt_file"
    return "$status"
  fi

  while :; do
    start_matrix_batch "$title" "$allowed_json" "$acceptance_json"
    run_gitpilot_prompt "$prompt_file" "$title"

    status=0
    failed_label=""

    run_batch_verification "$batch_id" || {
      status=$?
      failed_label="verification for $batch_id"
    }

    if [ "$status" -eq 0 ]; then
      record_batch "$batch_id" "$title" "passed" "$prompt_file"
      return 0
    fi

    if [ "$attempt" -ge "$MAX_REPAIR_ATTEMPTS" ]; then
      record_batch "$batch_id" "$title" "failed after repairs" "$prompt_file"
      return "$status"
    fi

    attempt=$((attempt + 1))
    record_batch "$batch_id" "$title" "repair attempt $attempt after $failed_label" "$prompt_file"
    run_repair_batch "$batch_id" "$attempt" "$failed_label" "$prompt_file"
  done
}

run_all_batches() {
  section "Run Matrix Designer-exported governed batches"

  local batch_id

  while IFS= read -r batch_id; do
    [ -n "$batch_id" ] || continue
    run_exported_batch "$batch_id"
  done < <(batch_ids_from_export)
}

run_final_verification() {
  section "Final verification"

  npm_install_if_needed

  run_capture "final npm run build" \
    npm run build

  run_capture "final npm run verify" \
    npm run verify

  run_capture "final npm run smoke" \
    npm run smoke
}

maybe_deploy() {
  section "Optional Vercel deploy"

  if [ -z "${VERCEL_TOKEN:-}" ]; then
    printf 'Skipping Vercel deploy: VERCEL_TOKEN not set\n'
    KNOWN_LIMITATIONS+=("Vercel deployment skipped because VERCEL_TOKEN was not set.")
    return 0
  fi

  need_cmd npx

  run_capture "vercel deploy" \
    npx vercel deploy --prod --token "$VERCEL_TOKEN"

  if [ -n "${PUBLIC_GAME_URL:-}" ]; then
    PUBLISHED_URL="$PUBLIC_GAME_URL"
    run_capture "verify public URL" curl -I "$PUBLIC_GAME_URL"
  else
    KNOWN_LIMITATIONS+=("Vercel deploy ran, but PUBLIC_GAME_URL was not set for public URL verification.")
  fi
}

write_evidence() {
  section "Write evidence"

  local generated_at
  generated_at="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"

  {
    printf '# Contract Quest Governed Build Evidence\n\n'
    printf -- '- Generated at: `%s`\n' "$generated_at"
    printf -- '- Provider: `%s`\n' "${GITPILOT_PROVIDER:-unset}"
    printf -- '- Model: `%s`\n' "${GITPILOT_WATSONX_MODEL:-unset}"
    printf -- '- GitPilot config dir: `%s`\n' "${GITPILOT_CONFIG_DIR:-unset}"
    printf -- '- Matrix Designer bundle: `%s`\n' "$DESIGN_BUNDLE"
    printf -- '- Matrix Builder export: `%s`\n' "$MATRIX_EXPORT"

    if [ -n "$PUBLISHED_URL" ]; then
      printf -- '- Public game URL: `%s`\n' "$PUBLISHED_URL"
    else
      printf -- '- Public game URL: not recorded\n'
    fi

    printf '\n## Batch List\n\n'

    if [ -f "$MATRIX_EXPORT" ]; then
      node -e "const fs=require('fs'); const p=JSON.parse(fs.readFileSync(process.argv[1],'utf8')); for (const b of p.matrix_builder?.batches || []) console.log('- \`'+b.id+'\` — '+(b.title || b.id));" "$MATRIX_EXPORT"
    else
      printf -- '- Matrix export not available.\n'
    fi

    printf '\n## Batch Outcomes\n\n'

    if [ "${#BATCH_EVIDENCE[@]}" -eq 0 ]; then
      printf -- '- No batch outcomes recorded.\n'
    else
      local item id title outcome prompt
      for item in "${BATCH_EVIDENCE[@]}"; do
        IFS='|' read -r id title outcome prompt <<< "$item"
        printf -- '- `%s` — %s — %s — prompt `%s`\n' "$id" "$title" "$outcome" "$prompt"
      done
    fi

    printf '\n## Command Evidence\n\n'

    if [ "${#COMMAND_EVIDENCE[@]}" -eq 0 ]; then
      printf -- '- No command output recorded.\n'
    else
      local item cmd status log
      for item in "${COMMAND_EVIDENCE[@]}"; do
        IFS='|' read -r cmd status log <<< "$item"
        printf -- '- `%s` exited `%s`; output: `%s`\n' "$cmd" "$status" "$log"
      done
    fi

    printf '\n## Screenshots\n\n'

    if compgen -G "test-results/**/*.png" >/dev/null 2>&1 || compgen -G "playwright-report/**/*.png" >/dev/null 2>&1; then
      find test-results playwright-report -name '*.png' -print 2>/dev/null | sed 's/^/- `/; s/$/`/'
    else
      printf -- '- No screenshots were produced by the available checks.\n'
    fi

    printf '\n## Known Limitations\n\n'

    if [ "${#KNOWN_LIMITATIONS[@]}" -eq 0 ]; then
      printf -- '- None recorded.\n'
    else
      local limitation
      for limitation in "${KNOWN_LIMITATIONS[@]}"; do
        printf -- '- %s\n' "$limitation"
      done
    fi
  } > "$EVIDENCE_FILE"

  EVIDENCE_WRITTEN=1
  printf 'Evidence written to %s\n' "$EVIDENCE_FILE"
}

on_error() {
  local status=$?

  if [ "$STRICT_GENERATION" = "1" ] && [ "$EVIDENCE_WRITTEN" != "1" ]; then
    KNOWN_LIMITATIONS+=("Build stopped before completion with exit status $status.")
    write_evidence || true
  fi

  exit "$status"
}

from_zero() {
  STRICT_GENERATION=1

  ensure_local_gitignore
  ensure_evidence_dir
  require_governed_inputs
  check_gitpilot_connection
  check_watsonx_connection
  generate_design
  reset_matrix_builder_state
  npm_install_if_needed
  run_all_batches
  run_final_verification
  maybe_deploy
  write_evidence
}

print_help() {
  cat <<'HELP'
Usage: ./build.sh [verify|design|from-zero|generate|all|help]

Modes:
  verify     Run local npm/static checks against checked-in artifacts.
  design     Strictly generate blueprint/design, validate, and export with mdesign.
  from-zero  Strict design -> govern -> generate -> verify -> publish/evidence pipeline.
  generate   Alias for from-zero.
  all        Alias for from-zero.
  help       Show this help message.

Canonical .env names:
  GITPILOT_PROVIDER=watsonx
  WATSONX_API_KEY=...
  WATSONX_PROJECT_ID=...
  WATSONX_URL=https://us-south.ml.cloud.ibm.com
  GITPILOT_WATSONX_MODEL=openai/gpt-oss-120b
  GITPILOT_MAX_TOKENS=24000

Optional deployment names:
  VERCEL_TOKEN=...
  VERCEL_PROJECT_ID=...
  VERCEL_ORG_ID=...
  PUBLIC_GAME_URL=...

Secrets are never printed by this script. Rotate any watsonx key that was
pasted into chat or logs before using the final governed build.
HELP
}

load_env_file
trap on_error ERR

case "$MODE" in
  verify)
    ensure_local_gitignore
    ensure_evidence_dir
    verify_existing_artifacts
    ;;
  design)
    STRICT_GENERATION=1
    ensure_local_gitignore
    ensure_evidence_dir
    require_governed_inputs
    generate_design
    ;;
  from-zero|generate|all)
    from_zero
    ;;
  help|-h|--help)
    print_help
    ;;
  *)
    printf 'Unknown mode: %s\n' "$MODE" >&2
    printf 'Run ./build.sh help for usage.\n' >&2
    exit 2
    ;;
esac
````

## Final word

Contract Quest begins as a game idea, but it becomes a repeatable engineering pattern.

The game is the visible artifact. The pipeline is the real product.

> Matrix Designer defines the game we want. Matrix Builder governs how the AI may build it. GitPilot executes the work. watsonx.ai generates the code. Verification decides what is safe to publish.

That is how we move from prompt-driven demos toward reproducible AI software production.
