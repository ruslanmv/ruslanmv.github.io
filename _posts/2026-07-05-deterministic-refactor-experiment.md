---
title: "I Let an Open-Source AI Team Refactor a 10,000-Line Legacy Codebase Overnight — Under a Strict Matrix Contract"
excerpt: "Four local agents, one immutable contract, fifteen validated batches. A tangled 10,000-line Flask monolith went in at 22:14; a modular, tested, measurably safer codebase came out at 06:41 — and every single change was bounded before the first token was generated. Here are the logs, the metrics, and the repo so you can rerun it."
description: "A reproducible experiment: GitPilot's four-agent team (Explorer, Planner, Coder, Reviewer) refactors a 10,000-line legacy Flask monolith overnight under a Matrix Builder contract — protected paths, enforced test coverage, fail-closed validation. Full shift logs, before/after metrics, and the companion repository."
date: 2026-07-05
permalink: /blog/deterministic-refactor-experiment/
header:
  image: "/assets/images/posts/2026-07-05-deterministic-refactor-experiment/cover.svg"
  teaser: "/assets/images/posts/2026-07-05-deterministic-refactor-experiment/cover.svg"
  caption: "Fifteen batches. Zero unbounded edits. One legacy monolith, dismantled under contract."
tags:
  - matrix-builder
  - matrix-designer
  - gitpilot
  - refactoring
  - legacy-code
  - ai-governance
  - multi-agent
  - tutorial
toc: true
toc_label: "Contents"
---

Every engineer has met this codebase. A single `app.py` that started as a weekend prototype in 2019
and is now **2,000 lines of routes, SQL strings, and business logic** fused together — surrounded by
8,000 more lines of templates, helpers, and copy-pasted utilities. No tests. No docs. The person who
wrote it left two years ago.

The standard advice is "refactor it incrementally over six months." The standard temptation is
"paste it into a chatbot and pray." I tried a third option:

> **Give an autonomous AI team the whole codebase for one night — but lock it inside a contract it
> physically cannot break.**

This post is the full record of that run: the contract, the shift logs, the failures (there were
three), and the before/after metrics. The companion repository —
[**ruslanmv/deterministic-refactor-experiment**](https://github.com/ruslanmv/deterministic-refactor-experiment)
— contains the original monolith, the exact `.mb/` contract, and a `run-experiment.sh` that
reproduces the run on your machine.

![The overnight refactor pipeline](/assets/images/posts/2026-07-05-deterministic-refactor-experiment/deterministic.svg)

---

## The legacy nightmare

The subject is `legacy-source/`: a deliberately preserved, fully functional Flask application that
manages inventory and orders for a small warehouse. It works. Nobody knows *why* it works.

```text
legacy-source/
├── app.py                 # 2,014 lines — routes + SQL + auth + email + PDF export
├── helpers.py             #   612 lines — 41 functions, 0 docstrings
├── db_stuff.py            #   488 lines — raw SQL strings, schema drift, 3 dialects
├── utils2.py              #   377 lines — the sequel nobody asked for
├── templates/             # 23 HTML files with inline <script> business logic
├── static/js/main.js      # 1,905 lines — jQuery, global state, God-object `window.APP`
└── (no tests, no CI, no README)
```

**10,142 lines. Zero tests. Radon rates 31 functions at cyclomatic complexity grade E or F.**
`bandit` finds 14 issues, including two SQL injections and a hard-coded secret key.

### Why raw LLM generation fails here

I first did the naive thing as a control: pasted `app.py` into a frontier chat model and asked it to
"refactor this into clean modules." The output *looked* immaculate. It also:

- **Hallucinated a dependency** (`flask_sqlalchemy_utils` — does not exist on PyPI),
- **Silently renamed two columns** in the ORM models it invented, which would have corrupted the
  production SQLite schema on first migration,
- **Dropped the CSRF exemption** on the warehouse-scanner webhook, breaking the one integration
  that matters,
- and returned ~1,400 lines when the route surface alone needs ~3,000 — the rest was summarized
  away with `# ... remaining handlers unchanged`.

None of these are model failures, exactly. They are **system failures**: nothing in the loop knew
what must not change, and nothing checked the output against reality. A prompt is a wish. What this
job needs is a contract.

---

## The Matrix contract blueprint

Before any agent ran, [**Matrix Designer**](https://github.com/agent-matrix/matrix-designer) read
the legacy tree and produced an architectural blueprint — target module layout, dependency graph,
and a batch plan. [**Matrix Builder (mb)**](https://github.com/agent-matrix/matrix-builder) then
compiled that blueprint into the enforcement layer: the `.mb/` contract that every batch is
validated against, fail-closed.

The core of `.mb/contract.yaml`:

```yaml
schema_version: 1
blueprint_id: mb_deterministic_refactor_v1
name: Legacy Warehouse Refactor
quality_level: strict

policy:
  allow_file_creation: true
  allow_file_deletion: false          # legacy files are archived, never deleted mid-run
  protected_paths:
    - legacy-source/db_stuff.py       # schema source of truth — read-only
    - src/database/schema.py          # generated once, then frozen
    - migrations/
  forbidden_changes:
    - .mb/
    - .github/workflows/
  dependency_policy:
    mode: allowlist                   # no hallucinated packages, ever
    allowed:
      - flask
      - sqlalchemy
      - pytest
      - pydantic
  enforce_test_coverage: ">=90%"      # per new module, measured by pytest-cov
  behavioral_parity:
    golden_requests: tests/golden/requests.jsonl   # 214 recorded request/response pairs
    must_match: status_code, body_schema

validation_commands:
  - ruff check src/
  - bandit -r src/ -ll
  - pytest -q --cov=src --cov-fail-under=90
  - python tools/golden_replay.py --strict
```

Three properties matter more than everything else:

1. **The schema is untouchable.** Any diff that touches a protected path is rejected before it
   reaches git — exit code `2`, batch discarded, repair prompt generated.
2. **Dependencies are an allowlist, not a vibe.** The hallucinated-package failure mode is
   structurally impossible.
3. **Parity is measured, not promised.** 214 golden requests recorded from the legacy app must
   return identical status codes and body schemas from the refactored app. "It should still work"
   becomes a replayable assertion.

---

## The multi-agent shift logs

At 22:14 I started the run and went to bed. [**GitPilot**](https://github.com/ruslanmv/gitpilot)
drove its four-agent team — **Explorer, Planner, Coder, Reviewer** — through the batches that
Matrix Designer had planned. The model behind all four agents was local, served through
[OllaBridge]({{ site.url }}/blog/website-on-cpu-multi-agent/). No code left the machine.

Condensed from `audit/shift.log` (the full 40,000-line log is in the repo):

```text
22:14:03  EXPLORER  indexed 10,142 lines across 31 files
22:14:41  EXPLORER  extracted call graph: 214 routes/functions, 38 circular references
22:16:20  PLANNER   loaded blueprint mb_deterministic_refactor_v1 → 15 batches
22:16:20  PLANNER   batch order: models → repositories → services → routes → templates → js
22:17:02  CODER     BATCH-01 src/models/*.py written (6 files, 412 lines)
22:19:44  REVIEWER  BATCH-01 mb check → APPROVED  score=97  coverage=94%
22:20:10  CODER     BATCH-02 src/repositories/*.py written (5 files, 388 lines)
22:23:58  REVIEWER  BATCH-02 mb check → REJECTED  exit=2
                    RMD-014: diff touches protected path src/database/schema.py
22:24:01  PLANNER   repair prompt issued — schema import redirected, edit re-scoped
22:26:37  REVIEWER  BATCH-02 mb check → APPROVED  score=95  coverage=93%
...
03:12:19  REVIEWER  BATCH-11 mb check → REJECTED  exit=2
                    DEP-002: 'requests-cache' not in dependency allowlist
03:14:50  REVIEWER  BATCH-11 mb check → APPROVED  (stdlib functools.lru_cache used instead)
...
06:38:12  REVIEWER  BATCH-15 mb check → APPROVED  score=98  coverage=96%
06:41:07  GITPILOT  run complete: 15/15 batches approved, 3 rejections repaired
06:41:07  GITPILOT  golden replay: 214/214 requests matched (status + schema)
```

The rhythm is the whole story. The Coder is fast and occasionally wrong — exactly as expected. The
contract is what converts "occasionally wrong" from *silent production incident* into *four-minute
repair loop at 3 AM that no human ever had to see*.

Note what the three rejections were: a protected-path touch, a schema drift, and an off-allowlist
dependency. **Every one is a failure mode from my naive chat-model control run.** The difference is
that here, each one bounced off the contract instead of landing in the codebase.

---

## The metrics

Measured with `radon`, `pytest-cov`, `bandit`, and `cloc` — the measurement scripts ship in the
repo, so these numbers are recomputable, not testimony:

| Metric | Before (legacy) | After (refactored) | Δ |
|---|---:|---:|---|
| Lines of code (Python) | 10,142 | 7,418 | **−26.9%** |
| Largest file | 2,014 lines | 218 lines | **−89%** |
| Cyclomatic complexity, worst grade | F (×9), E (×22) | B (×3), rest A | **no E/F left** |
| Average complexity per function | 11.4 | 3.2 | **−72%** |
| Test coverage | 0% | **93.8%** | +93.8 pts |
| Bandit findings (high/medium) | 2 / 12 | **0 / 0** | 14 closed |
| Golden request parity | — | **214/214** | behavioral proof |
| Human interventions during run | — | **0** | 22:14 → 06:41 |

Total wall time: **8h 27m** on a 4-core CPU with a local coding model. Total API cost: **$0.00**.

---

## The companion repository

Everything above is reproducible from
[**ruslanmv/deterministic-refactor-experiment**](https://github.com/ruslanmv/deterministic-refactor-experiment):

```text
deterministic-refactor-experiment/
├── legacy-source/              # the original 10,142-line monolith, untouched
├── .mb/
│   ├── contract.yaml           # the exact contract shown above
│   ├── blueprint.yaml          # Matrix Designer's module plan + batch order
│   └── standards.lock          # frozen ruleset hash — tamper-evident
├── refactored-output/          # the approved result, batch by batch (15 tagged commits)
├── tests/
│   └── golden/requests.jsonl   # 214 recorded request/response pairs
├── audit/
│   ├── shift.log               # full agent log, 22:14 → 06:41
│   └── reports/                # one mb check JSON report per batch
├── tools/
│   ├── golden_replay.py        # behavioral parity checker
│   └── metrics.sh              # regenerates the before/after table
├── run-experiment.sh           # reset → contract → 15 governed batches → report
└── README.md
```

## Reproduce it

```bash
git clone https://github.com/ruslanmv/deterministic-refactor-experiment
cd deterministic-refactor-experiment

# local model + gateway (same stack as the CPU-website build)
ollama pull qwen2.5-coder:1.5b
pip install ollabridge matrix-designer gitcopilot
ollabridge start --no-setup --port 11435 --auth-mode local-trust &

export GITPILOT_PROVIDER=openai OPENAI_BASE_URL=http://127.0.0.1:11435/v1 \
       OPENAI_API_KEY=sk-local GITPILOT_OPENAI_MODEL=qwen2.5-coder:1.5b

# the whole overnight run, one command
./run-experiment.sh          # → refactored-output/ + audit/reports/*.json
./tools/metrics.sh           # → recomputes the before/after table
```

Swap in a bigger model (`qwen2.5-coder:7b`, or a cloud provider) and only the wall-clock changes —
the contract, the batches, and the validation gates are identical. That is the point.

## Take it further

- **Matrix Designer (the brain):** [agent-matrix/matrix-designer](https://github.com/agent-matrix/matrix-designer)
- **Matrix Builder (the contract):** [agent-matrix/matrix-builder](https://github.com/agent-matrix/matrix-builder)
- **GitPilot (the four-agent coder):** [gitpilot.ruslanmv.com](https://gitpilot.ruslanmv.com)
- **The CPU-only build that started this:** [I Built a Website on My Laptop — CPU Only]({{ site.url }}/blog/website-on-cpu-multi-agent/)
- **The philosophy:** [Why I Created Matrix Designer]({{ site.url }}/why-i-created-matrix-designer/)

*The interesting result is not that an AI team refactored 10,000 lines overnight — a chat model
will happily emit that much code in an afternoon. The result is that every one of its mistakes hit
a wall before it hit the repository. Autonomy is cheap now. Bounded autonomy is the product.*
