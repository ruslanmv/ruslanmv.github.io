# A2A Agent Validator — Cloudflare Worker backend

Secure backend for the **A2A Agent Validator** frontend tool
(`assets/tools/a2a-validator.html`).

```text
GitHub Pages frontend (ruslanmv.com)        Cloudflare Worker backend
┌──────────────────────────────┐            ┌────────────────────────────┐
│ /tools/a2a-validator.html     │  POST      │ a2a-validator.workers.dev   │
│  • enter agent base URL       │ ────────▶  │  POST /validate { url }     │
│  • verdict / errors / warnings│            │  • CORS allowlist           │
│  • required-field checklist   │ ◀────────  │  • SSRF guard (no localhost) │  ──▶ <url>/.well-known/agent.json
│  • agent-card JSON viewer     │  JSON      │  • shape validation          │
└──────────────────────────────┘            └────────────────────────────┘
```

GitHub Pages is static — it cannot fetch an arbitrary agent card from the
browser (CORS) or safely probe networks. The Worker fetches the agent's
`/.well-known/agent.json`, validates its shape, and returns a structured result.

## Live deployment

```text
Production:  https://a2a-validator.cloud-data.workers.dev
Preview:     *-a2a-validator.cloud-data.workers.dev
```

## Endpoints

| Endpoint                 | Method      | Purpose                                  |
|--------------------------|-------------|------------------------------------------|
| `/`                      | GET         | Service info + endpoint list             |
| `/health`                | GET         | Liveness probe                           |
| `/validate?url=<agent>`  | GET         | Validate an agent card                   |
| `/validate`              | POST        | JSON body `{ "url": "https://…" }`       |
| `/validate`              | OPTIONS     | CORS preflight                           |

### Authentication

The validator can authenticate to protected agent-card endpoints. Use **POST**
so secrets never appear in the URL.

| Type             | POST `auth` object                                              |
|------------------|-----------------------------------------------------------------|
| `none`           | `{ "type": "none" }`                                            |
| `bearer`         | `{ "type": "bearer", "token": "…" }`                            |
| `basic`          | `{ "type": "basic", "username": "…", "password": "…" }`         |
| `custom_headers` | `{ "type": "custom_headers", "headers": { "X-API-Key": "…" } }` |

Hop-by-hop / spoofable headers (`host`, `content-length`, `connection`,
`x-forwarded-for`, `cf-*`, …) are rejected from `custom_headers`. The Worker
also maps upstream `401`/`403` to clear auth errors. Secrets are never logged
or stored. `GET /validate?url=…&bearer_token=…` is supported for quick checks
but the token then appears in the URL — prefer POST.

```bash
# bearer
curl -X POST ".../validate" -H "Content-Type: application/json" \
  -d '{"url":"https://your-a2a-server.com","auth":{"type":"bearer","token":"YOUR_TOKEN"}}'

# basic
curl -X POST ".../validate" -H "Content-Type: application/json" \
  -d '{"url":"https://your-a2a-server.com","auth":{"type":"basic","username":"u","password":"p"}}'

# custom headers
curl -X POST ".../validate" -H "Content-Type: application/json" \
  -d '{"url":"https://your-a2a-server.com","auth":{"type":"custom_headers","headers":{"X-API-Key":"K","X-Tenant-ID":"t"}}}'
```

### What it validates

The agent base URL is normalized to `<url>/.well-known/agent.json`, fetched, and
checked for the A2A agent-card shape:

- **Required fields:** `name`, `description`, `url`, `version` (strings),
  `capabilities` (object), `skills` (array)
- **Skills:** each entry must be an object; missing `id`/`name` → *warning*,
  wrong types → *error*
- **Content-Type:** non-`application/json` → *warning*

### Response shape

```json
{
  "valid": true,
  "agent_card_url": "https://your-agent.example.com/.well-known/agent.json",
  "auth_type": "bearer",
  "status_code": 200,
  "errors": [],
  "warnings": [],
  "agent_card": { "name": "…", "skills": [ … ] },
  "message": "A2A agent card looks valid based on basic checks."
}
```

The Worker replies `200` when valid, `422` when invalid, `400` on bad input.

## Security model

- **HTTP/HTTPS only** targets
- Blocks `localhost`, `127.0.0.1`, `0.0.0.0`, `::1`
- Blocks private IPv4 ranges (`10/8`, `192.168/16`, `172.16/12`, `169.254/16`)
- CORS restricted to an allowlist (ruslanmv.com + local dev origins)
- Read-only `GET` to the agent card — no tokens stored or logged

## Deploy / update (code for future updates)

The full Worker source lives in [`worker.js`](./worker.js) in this folder so it
can be re-deployed or extended at any time:

```bash
cd cloudflare-worker-a2a-validator
npm install -g wrangler   # if needed
wrangler login
wrangler deploy
```

To extend validation in the future, edit `worker.js` (e.g. add fields to
`REQUIRED_FIELDS`, tighten `validateAgentCardShape`, or support
`/.well-known/agent-card.json` as an alternate path) and run `wrangler deploy`
again. No secrets are required.

### Point the frontend at your Worker

Defaults to `https://a2a-validator.cloud-data.workers.dev`. Override without
editing the bundle:

```html
<script>
  window.RMV_A2A_API_BASE = "https://a2a-validator.your-subdomain.workers.dev";
</script>
```

(`window.RMV_API_BASE` is honored as a shared fallback.)

## Test after deploying

```bash
curl https://a2a-validator.cloud-data.workers.dev/health
# {"ok":true,"service":"a2a-validator"}

curl -X POST "https://a2a-validator.cloud-data.workers.dev/validate" \
  -H "Content-Type: application/json" \
  --data '{"url":"https://your-agent.example.com"}'
```

## Files

```text
cloudflare-worker-a2a-validator/
├─ worker.js        # the Worker (fetch + shape validation + SSRF guard)
├─ wrangler.toml    # deploy config
├─ README.md        # this file
└─ INSTALL.md       # step-by-step install / wiring notes
```
