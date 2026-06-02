---
title: "GitPilot — The First Open-Source Multi-Agent AI Coding Assistant"
layout: single
permalink: /gitpilot/
sitemap: true
canonical_url: https://ruslanmv.com/gitpilot/
last_modified_at: 2026-06-02
excerpt: "GitPilot is an open-source multi-agent AI coding assistant — Explorer, Planner, Coder, and Reviewer agents that collaborate to read, plan, write, and review code."
description: "GitPilot is the first open-source multi-agent AI coding assistant. A team of agents — Explorer, Planner, Coder, Reviewer — reads your repo, plans changes, writes and tests code, and opens pull requests, in VS Code, the web, and the CLI, with any LLM."
author_profile: true
toc: true
toc_label: "On this page"
toc_icon: "robot"
header:
  image: /assets/images/posts/projects/gitpilot/webapp.png
  teaser: /assets/images/posts/projects/gitpilot/webapp.png
  og_image: /assets/images/posts/projects/gitpilot/webapp.png
---

**GitPilot** is the first **open-source, multi-agent AI coding assistant**. Instead of one model behind a chat box, GitPilot deploys a team of specialized agents that collaborate like a real engineering team — and you stay in control the whole time.

<p>
  <a class="btn btn--primary" href="https://github.com/ruslanmv/gitpilot" target="_blank" rel="noopener">⭐ View on GitHub</a>
  <a class="btn btn--inverse" href="https://github.com/ruslanmv/gitpilot#readme" target="_blank" rel="noopener">Read the docs</a>
</p>

## How it works: four agents, one workflow

GitPilot mirrors how real teams ship software. Each agent has one job:

- **Explorer** — reads the repository, its structure, and dependencies to build context.
- **Planner** — drafts a clear, step-by-step execution plan with the diffs it intends to make.
- **Coder** — writes and tests the changes, self-correcting when something fails.
- **Reviewer** — validates the result and drafts the commit message and pull request.

![GitPilot multi-agent topologies — Explorer, Planner, Coder, Reviewer]({{ '/assets/images/posts/projects/gitpilot/topologies.png' | relative_url }})

## You stay in control: Ask, Auto, Plan

GitPilot offers three execution modes so you decide how much autonomy to give it:

- **Ask** — every potentially dangerous action needs your approval first.
- **Auto** — changes are applied immediately for fast, hands-off runs.
- **Plan** — read-only: GitPilot generates a plan and diffs without touching anything.

![GitPilot applying changes with a diff preview]({{ '/assets/images/posts/projects/gitpilot/operations.png' | relative_url }})

## Works where you work

GitPilot runs as a **VS Code extension**, a **web app**, and a **CLI** — with unified login and history across all three.

![GitPilot VS Code extension]({{ '/assets/images/posts/projects/gitpilot/vscode.png' | relative_url }})

![GitPilot demo on Hugging Face Spaces]({{ '/assets/images/posts/projects/gitpilot/demo.png' | relative_url }})

## Any LLM, zero lock-in — private by default

GitPilot works with **OpenAI, Anthropic Claude, IBM watsonx, Ollama, and OllaBridge**. Run it locally with Ollama and it's **private by default** — no telemetry, your code stays on your machine.

## Features

- Multi-agent pipeline (Explorer → Planner → Coder → Reviewer)
- Structured code generation with **diff preview** before patches are applied
- **GitHub integration** with pull-request creation
- Works with **any programming language**
- Open source under the **Apache 2.0** license, with 850+ passing tests

## Tech stack

- **Backend:** FastAPI (Python 3.11+), the CrewAI multi-agent framework
- **Frontend:** React web app + a VS Code extension (TypeScript)
- **LLMs:** OpenAI · Claude · IBM watsonx · Ollama · OllaBridge
- **Data/search:** PostgreSQL and Milvus (via the MCP stack)
- **Deploy:** Docker, Hugging Face Spaces, Vercel — cloud-agnostic

## Get started

GitPilot is free and open source. Clone it, star it, and try it on your own repositories:

<p>
  <a class="btn btn--primary" href="https://github.com/ruslanmv/gitpilot" target="_blank" rel="noopener">Get GitPilot on GitHub →</a>
</p>

Built by [Ruslan Magana Vsevolodovna](/about/). Explore more in [Projects](/projects/) and the [AI Agents guide](/ai-agents/).
