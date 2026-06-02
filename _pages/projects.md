---
title: "Projects"
layout: projects
excerpt: "Open-source platforms, live AI products, and research I've built and shipped."
sitemap: true
permalink: /projects/
redirect_from:
  - /projects
canonical_url: https://ruslanmv.com/projects/
hero_image: /assets/images/all-header-projects.jpg
header:
  og_image: /assets/images/all-header-projects.jpg

pj_intro: "Applied AI systems, infrastructure tools, and research prototypes."

# ---- Premium project cards (with quick-details modal) ----
featured:
  - title: "GitPilot"
    desc: "The first open-source multi-agent AI coding assistant."
    long: "GitPilot orchestrates a team of specialized agents — Explorer, Planner, Coder, and Reviewer — that collaborate like a real engineering team: read the repository, draft a step-by-step plan, write and test the changes, then review and open a pull request. It runs in VS Code, the web, and the CLI, works with any LLM (OpenAI, Claude, IBM watsonx, Ollama, OllaBridge), and is private by default."
    status: "Live"
    category: "AI Coding Assistant"
    role: "Founder & Builder"
    stack: "Python, FastAPI, CrewAI, React, TypeScript, Docker"
    launched: "2026"
    chips: ["Agents", "Developer Tools", "Open Source"]
    group: "AI Assistants"
    image: /assets/images/posts/projects/gitpilot/webapp.png
    gallery:
      - "/assets/images/posts/projects/gitpilot/vscode.png"
      - "/assets/images/posts/projects/gitpilot/demo.png"
      - "/assets/images/posts/projects/gitpilot/topologies.png"
    glance_title: "GitPilot at a glance"
    glance: "A multi-agent AI coding assistant. Explorer reads repository context, Planner drafts step-by-step plans with diffs, Coder writes and tests changes with self-correction, and Reviewer validates the output and writes the commit message — all under Ask, Auto, or Plan execution modes."
    capabilities:
      - "Multi-agent pipeline: Explorer → Planner → Coder → Reviewer."
      - "Ask / Auto / Plan modes with per-action approval in Ask mode."
      - "Any LLM, zero lock-in; private by default when run locally with Ollama."
      - "Works across a VS Code extension, web app, and CLI with unified history."
    url: "/gitpilot/"
    github: "https://github.com/ruslanmv/gitpilot"
    docs: "/gitpilot/"
  - title: "MatrixHub"
    desc: "Unified access to data, models, and agents across projects."
    long: "MatrixHub is the catalog and installer at the heart of Agent-Matrix — a package manager for agents and tools, with hybrid search, lockfile-style reproducibility, and native MCP registration, backed by a live cloud instance."
    status: "Live"
    category: "Agent Infrastructure"
    role: "Founder & Builder"
    stack: "Python, FastAPI, LangGraph, Docker, PostgreSQL"
    launched: "2024"
    chips: ["Infrastructure", "Agents", "API"]
    group: "Infrastructure"
    image: /assets/images/posts/projects/matrixhub.png
    gallery:
      - "/assets/images/posts/projects/gallery/matrixcloud-live.png"
    glance_title: "MatrixHub at a glance"
    glance: "The control layer for Agent-Matrix. MatrixHub provides unified access to a catalog, installer, and registry for AI agents, MCP servers, and tools across teams and projects."
    capabilities:
      - "Discover and search agents, MCP servers, and tools."
      - "Install and manage packages with lockfile reproducibility."
      - "Register and expose MCP servers across projects."
    url: "https://www.matrixhub.io"
    github: "https://github.com/agent-matrix/matrix-hub"
    docs: "https://cloud.matrixhub.io"
    essay: "/alive-system/"
  - title: "OllaBridge"
    desc: "A unified gateway for local and cloud LLMs behind one OpenAI-compatible API."
    long: "OllaBridge unifies every language model you run — local Ollama, spare GPUs, free clouds, paid APIs — behind one OpenAI-compatible endpoint, dialing outward over WebSockets so it works behind NAT and firewalls."
    status: "Production"
    category: "LLM Infrastructure"
    role: "Creator"
    stack: "Python, WebSockets, MCP, Apache 2.0"
    launched: "2024"
    chips: ["Infrastructure", "LLM Gateway"]
    group: "Infrastructure"
    image: /assets/images/posts/projects/ollabridge.png
    gallery:
      - "/assets/images/posts/projects/gallery/ollabridge-live.png"
      - "/assets/images/posts/projects/gallery/ollabridge-2.png"
      - "/assets/images/posts/projects/gallery/ollabridge-3.png"
      - "/assets/images/posts/projects/gallery/ollabridge-4.png"
    glance_title: "OllaBridge at a glance"
    glance: "One OpenAI-compatible gateway for every model you run — local Ollama, spare GPUs, free clouds, and paid APIs — with nothing to port-forward."
    capabilities:
      - "Unify local and cloud LLMs behind one endpoint."
      - "Outbound WebSocket nodes work behind NAT and firewalls."
      - "Smart router, admin dashboard, and model catalog."
    url: "https://ollabridge.com"
    github: "https://github.com/ruslanmv"
    docs: "https://pypi.org/project/ollabridge/"
  - title: "MedOS"
    desc: "A private, multilingual medical assistant aligned with WHO, CDC, and NHS guidance."
    long: "MedOS is a worldwide medical assistant offering private, multilingual health guidance aligned with published WHO, CDC, and NHS recommendations — built to inform and orient rather than diagnose, grounding every answer in recognised public-health guidance."
    status: "Live"
    category: "Health AI"
    role: "Creator"
    stack: "Python, RAG, LLMs"
    launched: "2024"
    chips: ["AI Assistants", "Health AI"]
    group: "AI Assistants"
    image: /assets/images/posts/projects/medos.png
    gallery:
      - "/assets/images/posts/projects/gallery/medos-live.png"
    glance_title: "MedOS at a glance"
    glance: "A private, multilingual medical assistant that gives source-aligned health guidance, available any hour and designed to inform rather than diagnose."
    capabilities:
      - "Guidance aligned with WHO, CDC, and NHS sources."
      - "Multilingual — ask in your own language."
      - "Privacy-first; built to orient, not to diagnose."
    url: "https://ai-medical-chabot.com"
  - title: "LearnAI"
    desc: "An adaptive AI tutor that tunes every lesson to the learner's age, pace, and goals."
    long: "LearnAI is a personal AI tutor built on a single pedagogical loop that adapts across distinct learning worlds — from a child learning to count to a professional studying cloud and AI — and extends into an organizations tier with dashboards and learning paths."
    status: "Live"
    category: "Education AI"
    role: "Creator"
    stack: "Python, LLMs, React"
    launched: "2024"
    chips: ["AI Assistants", "Education AI"]
    group: "AI Assistants"
    image: /assets/images/posts/projects/learnai.png
    gallery:
      - "/assets/images/posts/projects/gallery/learnai-live.png"
      - "/assets/images/posts/projects/gallery/learnai-2.png"
      - "/assets/images/posts/projects/gallery/learnai-3.png"
      - "/assets/images/posts/projects/gallery/learnai-4.png"
    glance_title: "LearnAI at a glance"
    glance: "One adaptive teacher for every learner — a single pedagogical engine that reshapes itself for a child, a student, or a working professional."
    capabilities:
      - "Adapts lessons to age, pace, and goals."
      - "Distinct learning worlds on one engine."
      - "Organizations tier with dashboards and learning paths."
    url: "https://learnskillsai.com"
    docs: "https://learnskillsai.com/organizations"
  - title: "HomePilot Avatar"
    desc: "An embodied 3D and VR avatar you can speak to, for home and personal AI."
    long: "HomePilot is a modular application for building AI personas with a live, embodied browser-based 3D and VR avatar, multi-provider language-model support, and real-time speech — where I explore what a genuinely useful personal AI system looks like with real tools and a face."
    status: "Active"
    category: "Personal AI"
    role: "Creator"
    stack: "FastAPI, React, CrewAI, LangGraph"
    launched: "2023"
    chips: ["AI Assistants", "Personal AI"]
    group: "AI Assistants"
    image: /assets/images/posts/projects/homepilot.png
    gallery:
      - "/assets/images/posts/projects/gallery/homepilot-live.png"
    glance_title: "HomePilot at a glance"
    glance: "A modular platform for AI personas with a live, embodied 3D/VR avatar you can speak to — exploring what a useful personal AI looks like with real tools and a face."
    capabilities:
      - "Browser-based 3D and VR avatar with real-time speech."
      - "Multi-provider language-model support."
      - "Persona engine with persistent memory."
    url: "https://yourfriend.online"
    github: "https://github.com/ruslanmv/HomePilot"
    essay: "/personas-and-memory/"
  - title: "Best of the Best"
    desc: "An autonomous AI digest that surfaces the best papers, code, and tools every day."
    long: "Best of the Best is an autonomous system that curates the AI ecosystem daily — monitoring where new work appears, ranking what it finds, and surfacing the most important models, repositories, and papers without my involvement. A self-contained demonstration of the alive-system philosophy."
    status: "Curated"
    category: "AI Digest"
    role: "Creator"
    stack: "Python, autonomous agents"
    launched: "2024"
    chips: ["Agents", "AI Digest"]
    group: "Infrastructure"
    image: /assets/images/posts/projects/best-of-the-best.png
    gallery:
      - "/assets/images/posts/projects/gallery/bestofthebest-live.png"
      - "/assets/images/posts/projects/gallery/bestofthebest-2.png"
      - "/assets/images/posts/projects/gallery/bestofthebest-3.png"
      - "/assets/images/posts/projects/gallery/bestofthebest-4.png"
    glance_title: "Best of the Best at a glance"
    glance: "An autonomous curator that scans the AI ecosystem every day and surfaces the most important models, repositories, and papers — a working demonstration of the alive-system idea."
    capabilities:
      - "Monitors GitHub, Hugging Face, and new papers daily."
      - "Ranks and surfaces what matters most."
      - "Runs autonomously on a schedule."
    url: "https://ruslanmv.com/Best-of-the-Best/"

# ---- Compact index (right sidebar) ----
index_items:
  - { title: "Agent-Matrix", url: "https://agent-matrix.github.io" }
  - { title: "MatrixHub", url: "https://www.matrixhub.io" }
  - { title: "OllaBridge", url: "https://ollabridge.com" }
  - { title: "MedOS", url: "https://ai-medical-chabot.com" }
  - { title: "LearnAI", url: "https://learnskillsai.com" }
  - { title: "HomePilot Avatar", url: "https://yourfriend.online" }
  - { title: "Best of the Best", url: "https://ruslanmv.com/Best-of-the-Best/" }
---
