---
title: "Securing the AI Web: Authentication for MCP Servers and A2A Agents"
excerpt: "MCP and A2A are powerful — and wide open by default. Here's how authentication actually works for both, the practices production teams rely on, and two runnable Python demos protected by Bearer tokens."
description: "A practical, human guide to securing MCP servers and A2A agents over HTTP: where auth fits in each protocol, how the industry handles it, and two runnable FastAPI demos — a Bearer-protected MCP server and an OpenAI-powered A2A agent."

header:
  image: "/assets/images/posts/2026-01-23-authentication-for-mcp-servers-and-a2a-agents/hero.svg"
  teaser: "/assets/images/posts/2026-01-23-authentication-for-mcp-servers-and-a2a-agents/hero.svg"
  caption: "Public discovery, protected execution — starting with a Bearer token"

tags: [MCP, A2A, Agents, Authentication, Security, OAuth, FastAPI, OpenAI]
essay: true
essay_date: "2026-01-23"
read_time: "12 min read"
toc: true
toc_label: "Contents"
card_image: /assets/images/posts/2026-01-23-authentication-for-mcp-servers-and-a2a-agents/hero.svg
card_excerpt: "MCP and A2A are wide open by default. Here's how to lock them down — with two runnable Bearer-token demos."
---

As AI systems become more agentic, we are rapidly moving away from isolated chatbots and toward interconnected networks of tools and models. Two protocols are driving this shift: **MCP** (Model Context Protocol), which connects AI clients to external tools and data, and **A2A** (Agent-to-Agent Protocol), which lets agents discover and converse with one another.

Both are a joy to work with locally. You spin up a server, point a client at it, and everything just talks. But that frictionlessness hides a sharp edge: the moment you expose an MCP server or an A2A agent over HTTP, you have published an endpoint that will execute instructions and reach into your tools — and by default, it will do that for *anyone who finds the URL*. At that point authentication stops being a "nice-to-have" and becomes the difference between a service and a liability.

This article breaks down the authentication patterns MCP and A2A actually use, the practices production teams rely on today, and two working Python demos — a minimal MCP server and an A2A agent — both protected by Bearer tokens.

## The anatomy of agentic authentication

MCP and A2A solve different problems, but over HTTP their security takes the same fundamental shape: **a public discovery document tells the client how to authenticate, and a protected execution endpoint strictly enforces it.** Get that separation right and most of the design falls into place.

### Model Context Protocol (MCP)

For locally hosted MCP servers, authentication is handled by the host machine — the client launches the server over `stdio` and credentials ride along in environment variables. There is no network surface to defend.

Remote MCP servers are a different story. They use the **Streamable HTTP** transport — a single endpoint such as `https://example.com/mcp` that speaks JSON-RPC over `POST` (and Server-Sent Events) — and authentication happens entirely at the network layer. The official MCP authorization specification explicitly builds on **OAuth 2.1** concepts. Authorization is technically optional in the protocol, but the documentation is blunt about it: the second your server touches user data, sensitive APIs, or an enterprise environment, you are expected to turn it on.

### Agent-to-Agent Protocol (A2A)

A2A leans on discovery. A client begins with a `GET` request to a public `/.well-known/agent-card.json`, and that **Agent Card** advertises the agent's capabilities, the protocols it speaks, and exactly which authentication scheme it expects at the execution endpoint.

> **The cardinal rule of Agent Cards:** the card is public, so it must never leak private keys, real tokens, or internal infrastructure details. It describes *how* to authenticate — never *the secret itself*.

## How the industry handles auth

Out in the wild, you'll see one of four schemes — **None, Bearer tokens, Basic Auth, or Custom Headers** — sometimes fronted by OAuth for the serious deployments. The scheme matters less than the discipline around it. A handful of baselines separate systems that hold up from systems that get owned:

- **HTTPS is non-negotiable.** Never send a token over plain HTTP. Require TLS for every remote MCP and A2A endpoint, full stop.
- **Prefer short-lived tokens.** Whoever holds a Bearer token holds the keys to the kingdom — there is no "who are you really" check beyond possession. Keep expiry windows short and rotate often.
- **Enforce least privilege.** Resist the "god token" that unlocks everything. Scope access by environment, user, or capability — `mcp:tools.read` is a very different grant from `mcp:tools.call`.
- **Validate the `Origin`.** The MCP Streamable HTTP transport requires servers to check the `Origin` header. It's a small line of code that shuts down DNS-rebinding attacks from malicious web pages.
- **Separate discovery from execution.** Keep `GET /.well-known/agent-card.json` open to the world, but gate `POST /a2a` and `POST /mcp` ruthlessly. Return **`401 Unauthorized`** when a credential is missing, and **`403 Forbidden`** when one is present but wrong.

That last point is the whole game in one picture — an open front door for discovery, a locked one for action:

<figure class="post-figure">
  <img src="/assets/images/posts/2026-01-23-authentication-for-mcp-servers-and-a2a-agents/bearer-sequence.svg" alt="A sequence diagram between a client and an MCP server / A2A agent. First, GET /.well-known/agent-card.json (public) returns 200 with securitySchemes bearerAuth. Then POST /mcp with no Authorization header returns 401 Missing Authorization header. POST /mcp with a wrong Bearer token returns 403 Invalid token. Finally POST /mcp with the correct Bearer token returns 200 with the result." loading="lazy">
  <figcaption>Discovery is open; execution is gated. <code>401</code> means "you brought no key," <code>403</code> means "your key doesn't fit."</figcaption>
</figure>

## Demo 1: a minimal MCP server (Bearer auth)

Enough theory. Let's build a remote MCP-style endpoint — `POST /mcp` — that handles initialization and a tool call, protected entirely by a Bearer token. The whole server is small enough to read in one sitting.

**Prerequisites:** `pip install fastapi uvicorn`

The pattern is simple: one guard function runs before any work happens, and it draws a hard line between "no credential" (`401`) and "wrong credential" (`403`).

```python
import os
import time
from typing import Any, Dict

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

app = FastAPI(title="Demo MCP Server with Bearer Auth")
MCP_BEARER_TOKEN = os.getenv("MCP_BEARER_TOKEN", "dev-mcp-token")


def require_bearer_token(authorization: str | None) -> None:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header format")

    token = authorization[len("Bearer "):].strip()
    if token != MCP_BEARER_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")


def json_rpc_result(request_id: Any, result: Dict[str, Any]) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


@app.post("/mcp")
async def mcp_endpoint(
    request: Request,
    authorization: str | None = Header(default=None),
    mcp_protocol_version: str | None = Header(default=None),
):
    # 1. Enforce authentication before anything else happens
    require_bearer_token(authorization)

    body = await request.json()
    request_id = body.get("id")
    method = body.get("method")
    params = body.get("params", {})

    # 2. The MCP handshake
    if method == "initialize":
        return json_rpc_result(request_id, {
            "protocolVersion": mcp_protocol_version or "2025-06-18",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "demo-mcp-server", "version": "1.0.0"},
        })

    # 3. Tool discovery + execution
    if method == "tools/list":
        return json_rpc_result(request_id, {"tools": [
            {"name": "time", "description": "Return the current server time.",
             "inputSchema": {"type": "object", "properties": {}}},
        ]})

    if method == "tools/call" and params.get("name") == "time":
        now = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        return json_rpc_result(request_id, {"content": [{"type": "text", "text": now}]})

    return JSONResponse(status_code=404, content={"error": "Method or tool not found"})
```

Run it with `uvicorn mcp_server:app --port 8001`. Hit the endpoint without the `dev-mcp-token` in your headers and the server rejects you before a single line of business logic executes — which is exactly where the rejection belongs.

> Want to watch this happen without writing a client? Paste the endpoint into the [MCP Server Inspector](https://ruslanmv.com/blog/inspect-and-validate-mcp-servers-and-a2a-agents), pick **Bearer**, and fire `initialize` → `tools/list` → `tools/call` live.

## Demo 2: a minimal A2A agent (Bearer auth + OpenAI)

Now the other half of the web. An A2A agent needs two endpoints with opposite postures: an **open** one that hands out the Agent Card, and a **protected** one that actually runs the model.

**Prerequisites:** `pip install fastapi uvicorn openai`

Notice how the Agent Card *advertises* `bearerAuth` as its security scheme without ever containing a token. A client reads the card, learns the rules of engagement, and only then — if it holds a valid credential — can it reach the execution endpoint.

```python
import os
from fastapi import FastAPI, Header, HTTPException, Request
from openai import OpenAI

app = FastAPI(title="Demo A2A Agent with Bearer Auth")
A2A_BEARER_TOKEN = os.getenv("A2A_BEARER_TOKEN", "dev-a2a-token")
client = OpenAI()  # reads OPENAI_API_KEY from the environment


def require_bearer_token(authorization: str | None) -> None:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    if authorization[len("Bearer "):].strip() != A2A_BEARER_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")


# The public discovery endpoint — describes auth, never contains it
@app.get("/.well-known/agent-card.json")
def agent_card():
    return {
        "name": "Demo OpenAI A2A Agent",
        "description": "Minimal A2A agent protected by Bearer token.",
        "url": "http://localhost:8002/a2a",
        "securitySchemes": {"bearerAuth": {"type": "http", "scheme": "bearer"}},
        "security": [{"bearerAuth": []}],
        "skills": [{"id": "chat", "name": "Chat", "inputModes": ["text/plain"]}],
    }


# The protected execution endpoint
@app.post("/a2a")
async def a2a_endpoint(request: Request, authorization: str | None = Header(default=None)):
    require_bearer_token(authorization)

    payload = await request.json()
    user_text = payload.get("message", "")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful A2A demo agent."},
            {"role": "user", "content": user_text},
        ],
    )
    answer = response.choices[0].message.content

    return {"role": "agent", "text": answer}
```

By splitting the public card from the protected endpoint, any client can dynamically learn *how* to talk to your agent — without ever holding the credentials needed to abuse it. That is the entire philosophy of agentic auth, expressed in two route handlers.

## Choosing the right authentication strategy

The most common security mistake isn't using weak auth — it's reaching for enterprise-grade machinery on day one and stalling, or shipping a prototype's throwaway token straight into production. The healthier instinct is to **match the scheme to the stage** and climb deliberately as the stakes rise.

<figure class="post-figure">
  <img src="/assets/images/posts/2026-01-23-authentication-for-mcp-servers-and-a2a-agents/auth-ladder.svg" alt="An authentication ladder rising diagonally from 'simple / demo' at the bottom-left to 'enterprise / multi-user' at the top-right. The rungs are None (open endpoints, local tests), Bearer token (demos and service-to-service, highlighted), Basic / Custom Headers (behind a gateway, X-API-Key), and OAuth 2.1 / OIDC (users, consent, scopes, refresh, audit, scoped tokens)." loading="lazy">
  <figcaption>Climb only as high as the use case demands. Each rung adds control — and operational weight.</figcaption>
</figure>

- **Local development (`stdio`).** Lean on environment variables and local config. There's no network attacker to defend against yet.
- **Public demos and prototypes.** A hardcoded — but non-sensitive — Bearer token is perfect for proving out the architecture without ceremony.
- **Internal service-to-service.** Keep Bearer tokens, but route them through an API gateway with IP restrictions and automated rotation.
- **Multi-tenant or enterprise production.** This is where you graduate to **OAuth 2.1 or OIDC**: scoped access tokens, real consent flows, and the audit trails your compliance team will eventually ask for.

> If you're prototyping right now, you don't need any of the upper rungs — you need a *good* token and somewhere safe to keep it. Our [MCP / A2A Auth Generator](https://ruslanmv.com/tools/auth-generator.html) mints cryptographically secure Bearer, Basic, and Custom-Header configs in the browser and hands you ready-to-paste snippets for exactly this.

## The pre-flight checklist

Before any MCP server or A2A agent touches the public web, walk this list end to end. None of it is exotic; all of it is the stuff that gets skipped under deadline pressure.

- Is traffic forced over **HTTPS**?
- Are secrets **completely absent** from public discovery documents (Agent Cards)?
- Are tokens loaded from **environment variables or a secrets manager** — never hardcoded in a committed file?
- Are you logging *metadata* (`user_id`, `scope`, `request_id`) while explicitly **filtering out** `Authorization` headers, passwords, and API keys?
- Are rejections handled correctly — **`401`** for missing credentials, **`403`** for invalid ones?
- Is **`Origin` validation** enabled for anything reachable from a browser?

## Conclusion

MCP and A2A are two sides of the same coin. MCP protects *how models reach out to the real world*; A2A governs *how agents discover and collaborate with one another*. They start from different problems, but the instant they hit the network their security needs converge on the same shape: open discovery, gated execution, and a credential that proves you belong.

Bearer authentication is brilliant precisely because it's simple — which makes it the ideal foundation. Treat it as your starting layer while you prototype, and let it pave the way toward stronger identity frameworks like OAuth 2.1 as you take on real users and sensitive data. The AI web is moving fast. Make sure your security moves with it.

> **Try it on the wire.** Generate a credential with the [Auth Generator](https://ruslanmv.com/tools/auth-generator.html), then exercise it against a real server with the [MCP Inspector and A2A Validator](https://ruslanmv.com/blog/inspect-and-validate-mcp-servers-and-a2a-agents) — both send all four auth types and show you the raw response.

### Sources & references

- [Understanding Authorization in MCP](https://modelcontextprotocol.io/docs/tutorials/security/authorization)
- [MCP Transports specification (2025-06-18)](https://modelcontextprotocol.io/specification/2025-06-18/basic/transports)
- [A2A specification — a2aproject/A2A](https://github.com/a2aproject/A2A/blob/main/docs/specification.md)
- [OpenAI API Platform](https://openai.com/api/)
