# MCP Server Inspector — install notes

This add-on ships a premium-style frontend tool plus an optional secure backend.

```text
Frontend (GitHub Pages / Jekyll)
  assets/tools/mcp-inspector.html
  assets/tools/assets/mcp-inspector/
      mcp-inspector.css
      mcp-inspector.js          # controller (tabs, wiring)
      command-builder.js        # generates the local npx command + .sh script
      mcp-client.js             # talks to the Worker (JSON-RPC over HTTPS)
      diagnostics.js            # the diagnostics checklist logic
  (shared) assets/tools/assets/shared/tool-shell.css | tool-shell.js | export.js | storage.js

Cloudflare Worker backend
  cloudflare-worker-mcp-inspector/worker.js
  cloudflare-worker-mcp-inspector/wrangler.toml
```

## 1. Frontend (already wired)

The page is registered in the tools hub (`assets/tools/index.html`) as:

```js
{cat:'ai', icon:'mcp', name:'MCP Server Inspector', href:'mcp-inspector.html', … }
```

It reuses the shared `tool-shell` design system, so it inherits the
ruslanmv.com look (dark-green hero header, warm-white cards, green accent,
rounded corners, mobile responsive) with no extra setup.

### Run it locally

```bash
make serve        # Jekyll
# then open whichever address Jekyll prints, e.g.
#   http://localhost:4000/assets/tools/mcp-inspector.html
```

Or any static server:

```bash
python3 -m http.server 8000
# http://localhost:8000/assets/tools/mcp-inspector.html
```

The **Command Builder** and **Diagnostics** tabs work fully offline/static.
The **Remote HTTP Test** tab needs the Worker (below).

## 2. Deploy the Worker

```bash
cd cloudflare-worker-mcp-inspector
npm install -g wrangler
wrangler login
wrangler deploy
```

Result:

```text
https://mcp-inspector-api.<your-subdomain>.workers.dev
```

## 3. Point the frontend at your Worker

The default base URL is in `assets/tools/assets/mcp-inspector/mcp-client.js`:

```js
const DEFAULT_WORKER_BASE = 'https://mcp-inspector-api.cloud-data.workers.dev';
```

Either edit that constant, or override it at runtime without touching the bundle:

```html
<script>
  window.RMV_MCP_API_BASE = "https://mcp-inspector-api.your-subdomain.workers.dev";
</script>
```

The header badge on the page shows **“Worker online”** once `/api/health`
responds, or **“Worker offline”** otherwise.

## 4. Allowed CORS origins

Set in `worker.js` → `ALLOWED_ORIGINS`:

```text
http://localhost:8000      http://127.0.0.1:8000
http://localhost:4000      http://127.0.0.1:4000
http://localhost:4001      http://127.0.0.1:4001
https://ruslanmv.com       https://www.ruslanmv.com
https://ruslanmv.github.io
```

For a page served from `https://ruslanmv.com/tools/mcp-inspector.html`, the CORS
origin is exactly `https://ruslanmv.com`. Add/remove origins to match your hosts.

## 5. Quick verification

```bash
curl https://mcp-inspector-api.<your-subdomain>.workers.dev/api/health
# {"ok":true,"service":"mcp-inspector-api", … }

curl -X POST "https://mcp-inspector-api.<your-subdomain>.workers.dev/api/mcp" \
  -H "Content-Type: application/json" \
  --data '{"endpoint":"https://your-mcp-server.example.com/mcp","action":"initialize","token":"YOUR_TOKEN"}'
```

## Security reminders (enforced by the Worker)

- HTTPS-only targets; localhost / private / metadata IPs are rejected.
- Visitor tokens are **never** logged, stored, or echoed back.
- Request/response sizes are capped and a hard timeout is applied.
- Best-effort per-IP rate limiting (60 req/min). For strict limits, use a
  Durable Object or the Cloudflare Rate Limiting product.
- Frontend keeps tokens **in memory only** — never in `localStorage` or the URL,
  and masks them in the UI.
- Local **stdio** MCP servers cannot be inspected online; the Command Builder
  generates the `npx @modelcontextprotocol/inspector …` command to run locally.
```
