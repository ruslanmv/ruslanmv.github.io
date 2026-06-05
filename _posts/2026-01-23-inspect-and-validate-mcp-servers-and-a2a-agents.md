---
title: "Inspecting MCP and A2A Agents"
excerpt: "Exposing an MCP server or A2A agent to the network means CORS walls, leaked tokens, and silent failures. Here are two free browser tools — and the hardened Worker — that make remote agent debugging human again."
description: "A developer's tour of the MCP Server Inspector and A2A Agent Validator: why a static site needs a Cloudflare Worker proxy, the MCP initialize → tools/list → tools/call loop against DeepWiki and Context7, and agent-card validation against live A2A agents like Silas and PostalForm."

header:
  image: "/assets/images/posts/2026-01-23-inspect-and-validate-mcp-servers-and-a2a-agents/hero.svg"
  teaser: "/assets/images/posts/2026-01-23-inspect-and-validate-mcp-servers-and-a2a-agents/hero.svg"
  caption: "Two static browser tools, one secure Cloudflare Worker"

tags: [MCP, A2A, Agents, Cloudflare Workers, Developer Tools, JSON-RPC]
essay: false   # developer how-to — lives in /blog, not the /essays section
read_time: "9 min read"
---

Building with agentic AI right now feels a bit like the early days of the web. The protocols are shifting beneath our feet, and two in particular are quietly becoming the underlying plumbing: **Model Context Protocol (MCP)**, which links AI models to data sources and tools, and **Agent-to-Agent (A2A)**, which lets separate AI entities discover and talk to one another.

On `localhost`, everything is easy. But the moment you expose an MCP server or an A2A agent to the open network, reality hits. Your browser starts blocking requests over CORS. You realize you can't embed a bearer token in static HTML without leaking it. And debugging a half-formed agent card becomes an exercise in tracing silent failures.

To make that process more human, I built two lightweight, free browser tools to handle the heavy lifting: the **MCP Server Inspector** and the **A2A Agent Validator**. Here's how they work under the hood — and how they solve the security headaches unique to remote agent testing.

## The architecture: why a static page needs a thin backend

Host a debugging tool on a static platform like GitHub Pages and you hit an immediate wall. A browser simply cannot bypass cross-origin restrictions, safely hold an API secret, or shell out to a command line.

Rather than build a heavy, stateful app to get around that, I paired the static frontend with a hardened **Cloudflare Worker** acting as an intelligent intermediary. The division of labor is clean: the browser owns the UI, the Worker owns the network.

<figure class="post-figure">
  <img src="/assets/images/posts/2026-01-23-inspect-and-validate-mcp-servers-and-a2a-agents/architecture.svg" alt="Three tiers. The browser on GitHub Pages builds the request and holds the token in memory only, but cannot bypass CORS or run npx. The Cloudflare Worker in the middle adds CORS, enforces HTTPS, guards against SSRF, attaches auth, and parses JSON or SSE. The remote MCP server or A2A agent on the right does the real work." loading="lazy">
  <figcaption>The browser builds the payload; the Worker attaches auth headers, enforces timeouts, cleans up CORS, and returns a predictable JSON shape.</figcaption>
</figure>

The Worker is deliberately an *ephemeral pipe*. It operates in memory, blocks localhost and private IP ranges to prevent server-side request forgery, rejects unencrypted targets, caps sizes and timeouts, and never logs or retains your tokens. One job, done safely.

## Inside the MCP Server Inspector

The Inspector splits along a simple line: where your server actually lives.

**Local servers.** If your MCP server runs as a local `stdio` process, a remote web app has no business reaching into it — so the tool doesn't try. Instead it becomes a configuration builder: you supply your paths and environment variables, and it hands back a clean command to run yourself.

```bash
npx @modelcontextprotocol/inspector \
  -e MCP_AUTH="Bearer YOUR_TOKEN" \
  -e MCP_SERVER_URL="https://your-mcp-server.example.com/mcp" \
  /path/to/venv/bin/python -m mcpgateway.wrapper
```

**Remote servers.** For live HTTP endpoints, the Worker drives the standard MCP lifecycle — **Initialize → List Tools → Call Tool** — and shows you the status, latency, and transport for each step.

<figure class="post-figure">
  <img src="/assets/images/posts/2026-01-23-inspect-and-validate-mcp-servers-and-a2a-agents/mcp-flow.svg" alt="Three stages. Initialize returns serverInfo and an optional Mcp-Session-Id. tools/list replays the session id and returns the tools array. tools/call sends a name and arguments and returns result.content. A band below contrasts DeepWiki (stateless) with Context7 (stateful, issues a session id that must be replayed)." loading="lazy">
  <figcaption>Three round-trips. The session id from <code>initialize</code> is captured and replayed automatically on the calls that follow.</figcaption>
</figure>

That session handling is the part that quietly saves you. Testing against **DeepWiki** (a public MCP server that reads GitHub repos), the calls are stateless — each stands alone. But production servers like **Context7** issue a unique `Mcp-Session-Id` during initialization and reject anything that doesn't carry it. The Inspector catches that header and threads it through every later call, so a stateful server never locks you out. And when something upstream misbehaves, a built-in **Diagnostics** panel grades the failure against a plain checklist: TLS, reachability, and response validity.

## Spotting faults with the A2A Agent Validator

A2A is all about discovery. An agent announces itself with a public file at `/.well-known/agent.json` (or the emerging `agent-card.json`), and the Validator's job is to fetch that file and check it against the protocol's structural rules — the foundational fields, the skills array — while staying flexible about custom extensions.

<figure class="post-figure">
  <img src="/assets/images/posts/2026-01-23-inspect-and-validate-mcp-servers-and-a2a-agents/a2a-flow.svg" alt="A four-step pipeline — normalize the base URL to /.well-known/agent.json, fetch over HTTPS, parse JSON, then shape-check the required fields and skills. Below, three verdicts: Silas valid with 3 skills, PostalForm valid with capability extensions, and example.com invalid with HTTP 404." loading="lazy">
  <figcaption>Normalize, fetch, parse, shape-check — then a verdict you can trust in both directions.</figcaption>
</figure>

Pointed at the public A2A registry, the tool surfaces how differently real agents present themselves. A minimal agent like **Silas** returns a clean schema — explicit skills, zero warnings, open auth. An enterprise agent like **PostalForm** shows how far the protocol stretches, advertising machine-payment extensions and very specific commerce capabilities.

But a validator earns trust by how it *rejects*, not just how it accepts. Feed it a non-agent URL like `example.com` and it returns an immediate, explicit `404` — not a vague timeout — proving the parsing layer can tell an unconfigured endpoint from a broken one.

## Try it on the wire

These protocols make a lot more sense once you watch the raw data move. Both tools are live:

1. Point the **[MCP Server Inspector](https://ruslanmv.com/assets/tools/mcp-inspector.html)** at `https://mcp.deepwiki.com/mcp` and watch the three-step initialize-and-call loop happen in real time.
2. Drop `https://silas.sylex.ai` into the **[A2A Agent Validator](https://ruslanmv.com/assets/tools/a2a-validator.html)** to see how an agent presents its skills — then try `example.com` to see a clean failure.

By letting an ephemeral backend handle the proxying and token isolation, you get the speed of a web tool without giving up basic network security. The workflow stays clean, visual, and simple — which, when you're debugging agents at 2 a.m., is the entire point.

> Protecting one of these endpoints yourself? The companion piece on [authentication for MCP and A2A](https://ruslanmv.com/blog/authentication-for-mcp-servers-and-a2a-agents) covers Bearer, Basic, and OAuth patterns — and the [Auth Generator](https://ruslanmv.com/assets/tools/auth-generator.html) mints secure tokens and ready-to-paste configs.

### Sources & references

- [Model Context Protocol — specification](https://modelcontextprotocol.io/)
- [A2A protocol — a2aproject/A2A](https://github.com/a2aproject/A2A)
- [A2A Registry — a2aregistry.org](https://www.a2aregistry.org/)
- Live test targets: [DeepWiki](https://mcp.deepwiki.com/mcp), [Context7](https://mcp.context7.com/mcp), [Silas](https://silas.sylex.ai/.well-known/agent.json), [PostalForm](https://postalform.com/.well-known/agent.json)
