# MCP Server Inspector — Cloudflare Worker backend

Secure backend for the **MCP Server Inspector** frontend tool
(`assets/tools/mcp-inspector.html`).

```text
GitHub Pages frontend (ruslanmv.com)        Cloudflare Worker backend
┌───────────────────────────────┐           ┌──────────────────────────────┐
│ /tools/mcp-inspector.html      │  POST     │ mcp-inspector-api.workers.dev │
│  • Command Builder (local npx) │ ───────▶  │  POST /api/mcp                │
│  • Remote HTTP Test            │           │  • CORS + HTTPS-only          │
│  • Diagnostics                 │ ◀───────  │  • SSRF guard + timeout       │  ──▶ remote MCP HTTP/SSE server
└───────────────────────────────┘  JSON      └──────────────────────────────┘
```

GitHub Pages is static hosting — it cannot run backend code, `npx`, or Python.
The Worker provides the one piece that needs a server: a **secure JSON-RPC proxy**
that adds CORS, hides the user's token from the browser's cross-origin rules,
blocks unsafe targets, and applies timeouts.

> The website **generates** the local `npx @modelcontextprotocol/inspector …`
> command for stdio/local servers — it never executes it. Local stdio MCP
> servers must always be inspected on the user's own machine.

## What the Worker does

| Endpoint            | Method | Purpose                                            |
|---------------------|--------|----------------------------------------------------|
| `/api/health`       | GET    | Liveness probe + metadata (no secrets)             |
| `/api/mcp`          | OPTIONS| CORS preflight                                     |
| `/api/mcp`          | POST   | Proxy one JSON-RPC request to a remote MCP server  |

### Request body (`POST /api/mcp`)

```json
{
  "endpoint": "https://example.com/mcp",
  "action": "initialize",
  "payload": {},
  "auth": { "type": "none" },
  "sessionId": "optional-mcp-session-id",
  "protocolVersion": "2025-06-18",
  "timeoutMs": 30000
}
```

`action` is one of:

```text
initialize    -> builds the MCP initialize handshake
tools/list    -> lists the server's tools
tools/call    -> calls a tool   (payload: { "name": "...", "arguments": {...} })
raw           -> forwards a verbatim JSON-RPC body (payload = the JSON-RPC object)
```

### Authentication

Supply an `auth` object (used only for the forwarded request — never logged or
stored). The result echoes the `authType` used.

| Type             | `auth` object                                                   |
|------------------|-----------------------------------------------------------------|
| `none`           | `{ "type": "none" }`                                            |
| `bearer`         | `{ "type": "bearer", "token": "…" }`                            |
| `basic`          | `{ "type": "basic", "username": "…", "password": "…" }`         |
| `custom_headers` | `{ "type": "custom_headers", "headers": { "X-API-Key": "…" } }` |

`custom_headers` rejects hop-by-hop / spoofable names (`host`, `connection`,
`x-forwarded-for`, `cf-*`, …). The legacy top-level `"token": "…"` field is still
accepted and treated as a bearer token.

```bash
# bearer
curl -X POST ".../api/mcp" -H "Content-Type: application/json" \
  -d '{"endpoint":"https://your-mcp-server.com/mcp","action":"initialize","auth":{"type":"bearer","token":"YOUR_TOKEN"}}'

# basic
curl -X POST ".../api/mcp" -H "Content-Type: application/json" \
  -d '{"endpoint":"https://your-mcp-server.com/mcp","action":"initialize","auth":{"type":"basic","username":"u","password":"p"}}'

# custom headers
curl -X POST ".../api/mcp" -H "Content-Type: application/json" \
  -d '{"endpoint":"https://your-mcp-server.com/mcp","action":"initialize","auth":{"type":"custom_headers","headers":{"X-API-Key":"K","X-Tenant-ID":"t"}}}'
```

### Response

```json
{
  "ok": true,
  "action": "tools/list",
  "httpStatus": 200,
  "statusText": "OK",
  "latencyMs": 184,
  "headers": { "content-type": "application/json", "mcp-session-id": "…" },
  "sessionId": "…",
  "transport": "http",        // or "sse"
  "json": { "jsonrpc": "2.0", "id": 1, "result": { "tools": [ … ] } },
  "raw": "…",
  "authType": "bearer",       // none | bearer | basic | custom_headers
  "via": "cloudflare-worker"
}
```

Both `application/json` and `text/event-stream` (SSE / Streamable HTTP)
responses are parsed; the JSON-RPC message is returned in `json`.

## Security model

This is **not an open proxy**. The Worker:

- accepts **HTTPS targets only** (rejects `http://`, `ws://`, `file://`, …)
- blocks `localhost`, `127.0.0.0/8`, `0.0.0.0`, `::1`, `*.local`, `*.internal`
- blocks private ranges `10/8`, `172.16/12`, `192.168/16`, CGNAT `100.64/10`
- blocks link-local `169.254/16` (incl. the cloud metadata IP `169.254.169.254`)
- blocks private/link-local IPv6 (`fc00::/7`, `fe80::/10`, `fd00:ec2::254`)
- caps request body (256 KB) and response (500 KB), with a hard timeout (≤120 s)
- applies best-effort per-IP rate limiting (60 req/min)
- restricts CORS to an allowlist (ruslanmv.com + local dev)
- supports **none / bearer / basic / custom_headers** auth; `custom_headers`
  rejects hop-by-hop / spoofable header names
- **never logs, stores, or echoes the visitor's credentials** — they are used
  only to set the request headers for that single forwarded request

Cloudflare **secrets** should be used only for *your own* service keys (never for
visitor-entered MCP tokens). Secret values are write-only after definition:
<https://developers.cloudflare.com/workers/configuration/environment-variables/>

## Deploy

```bash
cd cloudflare-worker-mcp-inspector
npm install -g wrangler   # if you don't have it
wrangler login
wrangler deploy
```

This publishes to:

```text
https://mcp-inspector-api.<your-subdomain>.workers.dev
```

### Point the frontend at your Worker

The frontend defaults to `https://mcp-inspector-api.cloud-data.workers.dev`.
To use a different Worker URL without editing the bundle, set a global before the
module loads (e.g. in the page `<head>` or a shared include):

```html
<script>
  window.RMV_MCP_API_BASE = "https://mcp-inspector-api.your-subdomain.workers.dev";
</script>
```

(`window.RMV_API_BASE` is also honored as a fallback, shared with the other
network tools.)

## Test after deploying

```bash
# Health check
curl https://mcp-inspector-api.<your-subdomain>.workers.dev/api/health

# Initialize a remote MCP server
curl -X POST "https://mcp-inspector-api.<your-subdomain>.workers.dev/api/mcp" \
  -H "Content-Type: application/json" \
  --data '{
    "endpoint": "https://your-mcp-server.example.com/mcp",
    "token": "YOUR_TOKEN",
    "action": "initialize"
  }'
```

You should see a JSON-RPC `initialize` result with `serverInfo` and `capabilities`.

## Files

```text
cloudflare-worker-mcp-inspector/
├─ worker.js        # the Worker (proxy + SSRF guard + SSE parsing)
├─ wrangler.toml    # deploy config
├─ README.md        # this file
└─ INSTALL.md       # step-by-step install / wiring notes
```
