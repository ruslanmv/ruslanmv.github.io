# A2A Agent Validator — install notes

```text
Frontend (GitHub Pages / Jekyll)
  assets/tools/a2a-validator.html
  assets/tools/assets/a2a-validator/
      a2a-validator.css
      a2a-validator.js          # controller (input, verdict, render)
      validator-client.js       # talks to the Worker (POST /validate)
      checks.js                 # required-field + lifecycle checklist logic
  (shared) assets/tools/assets/shared/tool-shell.css | tool-shell.js | export.js | storage.js

Cloudflare Worker backend
  cloudflare-worker-a2a-validator/worker.js
  cloudflare-worker-a2a-validator/wrangler.toml
```

## 1. Frontend (already wired)

Registered in the tools hub (`assets/tools/index.html`) as:

```js
{cat:'ai', icon:'a2a', name:'A2A Agent Validator', href:'a2a-validator.html', … }
```

It reuses the shared `tool-shell` design system, so it inherits the ruslanmv.com
look (dark-green hero header, warm-white cards, green accent, rounded corners,
mobile responsive) with no extra setup.

### Run locally

```bash
make serve        # Jekyll, then open the address it prints, e.g.
#   http://localhost:4000/assets/tools/a2a-validator.html
```

or any static server:

```bash
python3 -m http.server 8000
#   http://localhost:8000/assets/tools/a2a-validator.html
```

## 2. The Worker is already deployed

```text
https://a2a-validator.cloud-data.workers.dev
```

The frontend defaults to that URL. The page header shows **“Worker online”**
once `/health` responds.

## 3. Future updates

The full Worker source is vendored in `cloudflare-worker-a2a-validator/worker.js`.
To ship an update:

```bash
cd cloudflare-worker-a2a-validator
wrangler login        # first time only
wrangler deploy
```

Common extension points in `worker.js`:

- `REQUIRED_FIELDS` — add/remove required agent-card fields
- `validateAgentCardShape()` — stricter type/format checks
- `normalizeAgentCardUrl()` — also accept `/.well-known/agent-card.json`
- `ALLOWED_ORIGINS` — add hosts that may call the Worker from the browser

## 4. Point the frontend elsewhere (optional)

Edit the constant in `assets/tools/assets/a2a-validator/validator-client.js`:

```js
const DEFAULT_WORKER_BASE = 'https://a2a-validator.cloud-data.workers.dev';
```

or override at runtime:

```html
<script>
  window.RMV_A2A_API_BASE = "https://a2a-validator.your-subdomain.workers.dev";
</script>
```

## 5. Verify

```bash
curl https://a2a-validator.cloud-data.workers.dev/health
curl -X POST "https://a2a-validator.cloud-data.workers.dev/validate" \
  -H "Content-Type: application/json" \
  --data '{"url":"https://your-agent.example.com"}'
```

## Security reminders (enforced by the Worker)

- HTTP/HTTPS only; localhost & private IPv4 ranges are rejected.
- Read-only `GET` to the agent card; nothing is stored or logged.
- CORS limited to ruslanmv.com + local dev origins.
- Local agents on localhost/private IPs must be validated from your own machine.
