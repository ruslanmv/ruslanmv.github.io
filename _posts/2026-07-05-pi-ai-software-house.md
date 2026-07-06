---
title: "The $0 AI Software House: Running a 4-Agent Engineering Team on a $35 Raspberry Pi"
excerpt: "No GPU. No API keys. No cloud bill. A quantized 1.5B coding model on a Raspberry Pi 4, an OpenAI-compatible bridge, and GitPilot's four agents building a working React app under a contract. And if you don't own a Pi, one Docker command constrains your laptop to identical 4-core/4GB limits — no slow QEMU emulation. Here's the whole stack and the build logs."
description: "A reproducible local-first tutorial: run qwen2.5-coder:1.5b on a $35 Raspberry Pi 4 with Ollama, expose it via OllaBridge as an OpenAI-compatible gateway, and let GitPilot's four-agent team build a governed React to-do app entirely on CPU. Includes a docker-compose.yml that reproduces Pi-class limits (4 cores / 4GB) on any laptop without QEMU."
date: 2026-07-05
permalink: /blog/pi-ai-software-house/
header:
  image: "/assets/images/posts/2026-07-05-pi-ai-software-house/cover.svg"
  teaser: "/assets/images/posts/2026-07-05-pi-ai-software-house/cover.svg"
  caption: "Four agents, one $35 board, zero API keys. A real app, built entirely on-device."
tags:
  - gitpilot
  - ollama
  - ollabridge
  - raspberry-pi
  - local-llm
  - self-hosted
  - multi-agent
  - tutorial
toc: true
toc_label: "Contents"
---

The prevailing story is that agentic AI needs a server farm: a rack of A100s, a five-figure monthly
inference bill, a dependence on someone else's API staying up and staying cheap. I wanted to test the
floor of that claim, so I picked the least serious computer I own:

> **A $35 Raspberry Pi 4. No GPU. No API keys. No network calls to any model provider. Can four AI
> agents actually build and ship a working application on it?**

They can. This post is the full stack and the build logs. The companion repo —
[**ruslanmv/pi-ai-software-house**](https://github.com/ruslanmv/pi-ai-software-house) — boots the
whole thing with one script, and if you don't have a Pi on your desk, a `docker-compose.yml`
reproduces the Pi's exact resource limits on your laptop **without** slow ARM emulation.

![The local-first Pi stack](/assets/images/posts/2026-07-05-pi-ai-software-house/pi-ai.svg)

---

## The cost-of-inference trap

The hidden tax of "just use an API" is that every agent turn is metered, and agentic workflows are
*chatty* — a four-agent team plans, drafts, reviews, and repairs, which is easily 40–80 model calls
to ship one small feature. That is fine at demo scale and painful at real scale:

```text
Cloud frontier API   →  ~$0.003–0.015 per 1K tokens  ·  metered  ·  needs network + key + their uptime
Local qwen2.5-coder  →  $0.00 forever                ·  private  ·  runs with the WiFi unplugged
```

The trade is real: a 1.5B model is not a frontier model. But paired with the right *system* —
scoped batches, a contract, a validation gate — a small model doesn't need to be brilliant. It needs
to be correct inside a narrow, checked boundary. That is exactly what the Matrix stack provides, and
it is why "small model + strong system" beats "big model + no system" for bounded coding work.

---

## The local stack architecture

Three components, each with one job, all on the same board:

| Layer | Tool | Job |
|---|---|---|
| **Inference** | [Ollama](https://ollama.com) running `qwen2.5-coder:1.5b` | runs a sub-1GB quantized coding model on the Pi's CPU |
| **Gateway** | [OllaBridge](https://github.com/ruslanmv/ollabridge) | wraps Ollama in an **OpenAI-compatible** `/v1` endpoint |
| **Orchestration** | [GitPilot](https://github.com/ruslanmv/gitpilot) | drives the four-agent team against the local endpoint |

```text
   ┌──────────────────────── Raspberry Pi 4 · 4 cores · 4 GB · no GPU ────────────────────────┐
   │                                                                                           │
   │   GitPilot (4 agents)          OllaBridge  :11435            Ollama  :11434               │
   │   Explorer→Planner→            OpenAI-compatible     ──▶     qwen2.5-coder:1.5b            │
   │   Coder→Reviewer      ──▶      /v1/chat/completions          (quantized, CPU)             │
   │        ▲                                                            │                      │
   │        └──────────────── validated batches, governed by mb ◀───────┘                      │
   │                                                                                           │
   └───────────────────────────────  nothing leaves the board  ───────────────────────────────┘
```

The single indirection that makes it work: GitPilot speaks OpenAI, OllaBridge *is* OpenAI as far as
GitPilot can tell, and behind it a 1GB model hums on an ARM CPU. The coder never knows it isn't
talking to a data center.

---

## The resource choke (emulation guide)

Most readers don't have a Pi within reach, and the naive way to fake one — QEMU emulating ARM on an
x86 laptop — is agonizingly slow because every instruction is translated. **You don't need
architecture emulation to reproduce a Pi's constraints; you need its resource limits.** A Pi 4 is,
for our purposes, "4 cores and 4 GB." Docker gives you exactly that, at native speed:

```bash
# native-speed Pi-class sandbox — no QEMU, no ARM translation
docker run -it --cpus="4" --memory="4g" ubuntu:22.04
```

Or, the whole stack at once via the repo's `docker-compose.yml`:

```yaml
services:
  ollama:
    image: ollama/ollama:latest
    deploy:
      resources:
        limits: { cpus: "4.0", memory: 4g }   # ← the Pi ceiling
    volumes: [ "ollama:/root/.ollama" ]

  ollabridge:
    image: ruslanmv/ollabridge:latest
    depends_on: [ ollama ]
    environment:
      OLLAMA_BASE_URL: http://ollama:11434
      DEFAULT_MODEL: qwen2.5-coder:1.5b
      API_KEYS: sk-local-pi
      AUTH_MODE: local-trust
    ports: [ "11435:11435" ]
    deploy:
      resources:
        limits: { cpus: "4.0", memory: 4g }

  gitpilot:
    image: ruslanmv/gitpilot:latest
    depends_on: [ ollabridge ]
    environment:
      GITPILOT_PROVIDER: openai
      OPENAI_BASE_URL: http://ollabridge:11435/v1
      OPENAI_API_KEY: sk-local-pi
      GITPILOT_OPENAI_MODEL: qwen2.5-coder:1.5b
    volumes: [ "./output-app:/workspace/output-app" ]
    deploy:
      resources:
        limits: { cpus: "4.0", memory: 4g }

volumes:
  ollama:
```

If your laptop can run this compose file, your results will match the physical Pi within timing
noise, because the bottleneck is CPU cycles and RAM — both of which the compose file pins to Pi
levels. That is the honest way to let a reader "test at parity" without buying hardware.

---

## The live build output

The build brief, one sentence, dropped into `Matrixfile`:

> *"Build a responsive React to-do app with local-storage persistence, add/complete/delete, and a
> filter for active/completed."*

`start-software-house.sh` boots the stack and hands the brief to GitPilot. Condensed from the Pi's
own console (a real Pi 4, not the Docker stand-in):

```console
$ ./start-software-house.sh --idea "$(cat Matrixfile)"

[ollama]     model qwen2.5-coder:1.5b ready (portion resident: 1.1 GB)
[ollabridge] gateway up on :11435  →  mode=gateway  auth=local-trust
[gitpilot]   contract initialized (mb) · quality=starter · 5 batches planned

[explorer]  scaffold target: Vite + React + TS · scope locked to src/
[planner]   BATCH-01 types+store  BATCH-02 useTodos hook  BATCH-03 components
            BATCH-04 filters+persistence  BATCH-05 styles+a11y
[coder]     BATCH-01 → src/types.ts, src/store.ts                       (18.7s)
[reviewer]  mb check → APPROVED   score 96   coverage 100% (store)
[coder]     BATCH-02 → src/hooks/useTodos.ts                            (22.4s)
[reviewer]  mb check → REJECTED   exit=2   TEST-002 hook untested
[coder]     BATCH-02 → +src/hooks/useTodos.test.ts                      (19.1s)
[reviewer]  mb check → APPROVED   score 94   coverage 97%
[coder]     BATCH-03 → TodoInput / TodoList / TodoItem                  (41.8s)
[reviewer]  mb check → APPROVED   score 95
[coder]     BATCH-04 → localStorage sync + active/completed filter      (28.9s)
[reviewer]  mb check → APPROVED   score 97
[coder]     BATCH-05 → responsive CSS + aria labels + focus states      (24.2s)
[reviewer]  mb check → APPROVED   score 98   a11y: 0 axe violations

[gitpilot]  DONE · 5/5 batches approved · 1 rejection repaired
            output-app/ · npm run build → dist/ (94 KB gzipped) · $0.00 spent
```

Total build time on the physical Pi 4: **~9 minutes**, peak RAM **2.8 GB** (comfortably under the
4 GB ceiling), everything on-device. The `output-app/` directory is a real, buildable Vite app — it
ships in the repo so you can `npm run build` it without running the model at all.

Note the single rejection: the Coder skipped a test for the `useTodos` hook, the contract required
coverage, exit code `2` bounced it, and the repair added the test. Even at $0 on a toy board, nothing
broken ships.

---

## The companion repository

```text
pi-ai-software-house/
├── docker-compose.yml          # pins every service to 4 cores / 4GB — the Pi ceiling
├── Matrixfile                  # the one-sentence build brief (the contract seed)
├── start-software-house.sh     # boots Ollama + OllaBridge + GitPilot, zero API keys
├── .mb/
│   ├── contract.yaml           # starter-quality contract (scope=src/, tests required)
│   └── standards.lock
├── output-app/                 # the finished React to-do app, buildable as-is
│   ├── src/                    #   types, store, hooks, components
│   ├── index.html
│   └── package.json
├── logs/
│   └── build.log               # the full console trace above
└── README.md                   # Pi setup AND the laptop/Docker path, side by side
```

## Reproduce it — two paths

```bash
git clone https://github.com/ruslanmv/pi-ai-software-house
cd pi-ai-software-house

# Path A — a real Raspberry Pi 4 (or any Linux box)
./start-software-house.sh --idea "$(cat Matrixfile)"

# Path B — no Pi? reproduce Pi limits on your laptop, native speed, no QEMU
docker compose up      # everything capped at 4 cores / 4GB
docker compose exec gitpilot ./start-software-house.sh --idea "$(cat Matrixfile)"

# either path → a working app you can build without the model:
cd output-app && npm install && npm run build
```

## Take it further

- **GitPilot (the four-agent coder):** [github.com/ruslanmv/gitpilot](https://github.com/ruslanmv/gitpilot)
- **OllaBridge (OpenAI-compatible local gateway):** [github.com/ruslanmv/ollabridge](https://github.com/ruslanmv/ollabridge)
- **Matrix Builder (the contract that keeps the small model honest):** [agent-matrix/matrix-builder](https://github.com/agent-matrix/matrix-builder)
- **The laptop-CPU predecessor:** [I Built a Website on My Laptop — CPU Only]({{ site.url }}/blog/website-on-cpu-multi-agent/)

*The lesson isn't that a Raspberry Pi is a good place to run AI agents — it's a deliberately extreme
one. The lesson is that the expensive part of agentic coding was never the hardware. It was the lack
of a system to keep a modest model inside correct boundaries. Add that system, and a $35 board is
enough to run a software house that never sends a byte to the cloud.*
