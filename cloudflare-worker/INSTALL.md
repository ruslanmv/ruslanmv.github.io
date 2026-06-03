# RuslanMV install notes

This package is configured for your current setup:

```text
Frontend tools:
https://ruslanmv.com/assets/tools/
https://ruslanmv.github.io/          # GitHub Pages origin/fallback

Cloudflare Worker API:
https://network-api-tools.cloud-data.workers.dev

Local Jekyll development:
http://localhost:4000/
http://localhost:4001/               # observed server address in `make serve`
http://127.0.0.1:4000/
http://127.0.0.1:4001/
```

## Files to copy into your `/tools/` project

```text
ip-inspector.html
http-curl-builder.html
assets/shared/
assets/ip-inspector/
assets/http-curl-builder/
```

## Worker file to deploy

Deploy this file into Cloudflare Workers:

```text
cloudflare-worker/worker.js
```

The Worker exposes:

```text
GET  /api/ip
POST /api/proxy
```

## Allowed CORS origins

The Worker has been updated to allow these frontend origins:

```text
http://localhost:8000
http://127.0.0.1:8000
http://localhost:4000
http://127.0.0.1:4000
http://localhost:4001
http://127.0.0.1:4001
https://ruslanmv.com
https://www.ruslanmv.com
https://ruslanmv.github.io
```

Important: for a page like `https://ruslanmv.com/assets/tools/ip-inspector.html`, the CORS origin is only:

```text
https://ruslanmv.com
```

For local Jekyll pages such as `http://localhost:4000/tools/ip-inspector.html`, the CORS origin is only:

```text
http://localhost:4000
```

## Test after deploying the Worker

Open:

```text
https://network-api-tools.cloud-data.workers.dev/api/ip
```

You should see JSON with your IP and Cloudflare metadata.

Test the proxy endpoint:

```bash
curl -X POST "https://network-api-tools.cloud-data.workers.dev/api/proxy" \
  -H "Content-Type: application/json" \
  --data '{
    "method": "GET",
    "url": "https://api.github.com/repos/octocat/Hello-World/issues?state=open&per_page=1",
    "headers": {
      "Accept": "application/vnd.github+json",
      "User-Agent": "Network-API-Tools"
    },
    "body": ""
  }'
```

## Test the frontend locally

With your Jekyll site running:

```bash
make serve
```

Open one of these depending on what Jekyll prints:

```text
http://localhost:4000/tools/ip-inspector.html
http://localhost:4000/tools/http-curl-builder.html
http://localhost:4001/tools/ip-inspector.html
http://localhost:4001/tools/http-curl-builder.html
```

## Frontend Worker URL

The frontend modules already default to:

```js
const DEFAULT_WORKER_BASE = 'https://network-api-tools.cloud-data.workers.dev';
```

Updated files:

```text
assets/ip-inspector/ip-services.js
assets/http-curl-builder/request-runner.js
```

You can override the Worker URL before loading the module if needed:

```html
<script>
  window.RMV_API_BASE = 'https://network-api-tools.cloud-data.workers.dev';
</script>
```
