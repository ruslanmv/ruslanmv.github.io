# Network & API Tools Add-on

This package adds two premium-style frontend tools to your existing static tools project:

1. **IP & Network Inspector** — `ip-inspector.html`
2. **HTTP Request & cURL Builder** — `http-curl-builder.html`

It also includes shared UI assets and an optional Cloudflare Worker backend.

## Folder structure

```text
assets/
├─ shared/
│  ├─ tool-shell.css
│  ├─ tool-shell.js
│  ├─ storage.js
│  ├─ export.js
│  └─ icons.js
├─ ip-inspector/
│  ├─ ip-inspector.css
│  ├─ ip-inspector.js
│  └─ ip-services.js
└─ http-curl-builder/
   ├─ http-curl-builder.css
   ├─ http-curl-builder.js
   ├─ curl-generator.js
   ├─ request-runner.js
   └─ code-generators.js

ip-inspector.html
http-curl-builder.html
cloudflare-worker/
├─ worker.js
└─ wrangler.toml
```

## How to add to your project

Copy these files/folders into the root of your tools website:

```text
ip-inspector.html
http-curl-builder.html
assets/shared/
assets/ip-inspector/
assets/http-curl-builder/
```

Then add two cards to your homepage/tool hub:

```html
<a class="tool-card" href="ip-inspector.html">
  <h3>IP & Network Inspector</h3>
  <p>View your public IP, browser details, timezone, and network diagnostics.</p>
</a>

<a class="tool-card" href="http-curl-builder.html">
  <h3>HTTP Request & cURL Builder</h3>
  <p>Build API requests, generate cURL/fetch/Python code, and test endpoints.</p>
</a>
```

## Run locally

Use a local server because ES modules and some browser APIs work best over HTTP.

Simple static server:

```bash
python3 -m http.server 8000
```

Open:

```text
http://localhost:8000/ip-inspector.html
http://localhost:8000/http-curl-builder.html
```

For your Jekyll project using `make serve`, open whichever server address Jekyll prints, for example:

```text
http://localhost:4000/tools/ip-inspector.html
http://localhost:4000/tools/http-curl-builder.html
http://localhost:4001/tools/ip-inspector.html
http://localhost:4001/tools/http-curl-builder.html
```

## Static mode

Both tools work on GitHub Pages without a backend.

### IP tool

The IP tool tries:

1. `window.RMV_API_BASE + /api/ip`, if configured
2. `https://ipapi.co/json/`
3. `https://api.ipify.org?format=json`

### HTTP/cURL tool

The HTTP tool can:

- generate cURL
- generate JavaScript fetch
- generate Python requests
- generate Axios code
- send browser `fetch()` requests when the target API allows CORS

If a target API blocks browser CORS, use the generated cURL in the terminal or configure the optional Worker proxy.

## Optional Cloudflare Worker backend

The Worker adds:

```text
GET  /api/ip
POST /api/proxy
```

Deploy:

```bash
cd cloudflare-worker
npm install -g wrangler
wrangler login
wrangler deploy
```

Then add this before the page scripts in both HTML files, or in a shared config file:

```html
<script>
  window.RMV_API_BASE = "https://network-api-tools.cloud-data.workers.dev";
</script>
```

For production, protect `/api/proxy` with stricter rules: rate limits, allowed domains or authentication, stronger private-network blocking, and response-size limits.

## Security notes

- Browser-only requests can be blocked by CORS.
- Do not expose real private tokens in screenshots or shared reports.
- The Worker proxy should not be deployed as an unrestricted public open proxy.
- The tools store local state in the browser with `localStorage`.



## RuslanMV deployment configuration

This package has been updated for:

```text
Frontend production path:
https://ruslanmv.com/assets/tools/

GitHub Pages origin/fallback:
https://ruslanmv.github.io

Worker base URL:
https://network-api-tools.cloud-data.workers.dev

Local development origins:
http://localhost:8000
http://127.0.0.1:8000
http://localhost:4000
http://127.0.0.1:4000
http://localhost:4001
http://127.0.0.1:4001
```

The Worker CORS origins in `cloudflare-worker/worker.js` are:

```js
const ALLOWED_ORIGINS = [
  'http://localhost:8000',
  'http://127.0.0.1:8000',
  'http://localhost:4000',
  'http://127.0.0.1:4000',
  'http://localhost:4001',
  'http://127.0.0.1:4001',
  'https://ruslanmv.com',
  'https://www.ruslanmv.com',
  'https://ruslanmv.github.io'
];
```

The frontend modules already default to:

```js
const DEFAULT_WORKER_BASE = 'https://network-api-tools.cloud-data.workers.dev';
```

Test after deploying the Worker:

```text
https://network-api-tools.cloud-data.workers.dev/api/ip
```

```bash
curl -X POST "https://network-api-tools.cloud-data.workers.dev/api/proxy"   -H "Content-Type: application/json"   --data '{
    "method": "GET",
    "url": "https://api.github.com/repos/octocat/Hello-World/issues?state=open&per_page=1",
    "headers": {
      "Accept": "application/vnd.github+json",
      "User-Agent": "Network-API-Tools"
    },
    "body": ""
  }'
```
