// Turns an auth config into ready-to-use, professional code snippets for reuse
// with the MCP Inspector and A2A Validator (and any HTTP client). Every snippet
// embeds the real secret so it works as-is — the UI masks on screen, but copy
// and download always emit the real value.

import { authToHttpHeaders } from './auth-core.js';

export const DEFAULTS = {
  mcpWorker: 'https://mcp-inspector-api.cloud-data.workers.dev/api/mcp',
  a2aWorker: 'https://a2a-validator.cloud-data.workers.dev/validate',
  mcpEndpoint: 'https://your-mcp-server.example.com/mcp',
  a2aUrl: 'https://your-a2a-agent.example.com'
};

const j = (v) => JSON.stringify(v, null, 2);

// auth config object as a JS literal (auth: { ... })
export function snippetAuthConfig(auth) {
  return `// Auth config — drop straight into the MCP Inspector / A2A Validator\nconst auth = ${j(auth)};`;
}

export function snippetHeaders(auth) {
  const headers = authToHttpHeaders(auth);
  return `// Equivalent HTTP request headers\n${j(headers)}`;
}

// JavaScript fetch() for both the MCP worker and the A2A validator
export function snippetJavaScript(auth, ctx = DEFAULTS) {
  return `// Reusable client — MCP Inspector + A2A Validator (browser or Node 18+)
const auth = ${j(auth)};

// 1) Inspect a remote MCP server
async function inspectMcp(endpoint = ${q(ctx.mcpEndpoint)}) {
  const res = await fetch(${q(ctx.mcpWorker)}, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ endpoint, action: "initialize", auth }),
  });
  return res.json();
}

// 2) Validate an A2A agent card
async function validateA2A(url = ${q(ctx.a2aUrl)}) {
  const res = await fetch(${q(ctx.a2aWorker)}, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url, auth }),
  });
  return res.json();
}

console.log(await inspectMcp());
console.log(await validateA2A());`;
}

// curl for both workers
export function snippetCurl(auth, ctx = DEFAULTS) {
  const authStr = JSON.stringify(auth);
  const mcpBody = `{"endpoint":"${ctx.mcpEndpoint}","action":"initialize","auth":${authStr}}`;
  const a2aBody = `{"url":"${ctx.a2aUrl}","auth":${authStr}}`;
  return `# Inspect a remote MCP server
curl -X POST "${ctx.mcpWorker}" \\
  -H "Content-Type: application/json" \\
  -d '${mcpBody}'

# Validate an A2A agent card
curl -X POST "${ctx.a2aWorker}" \\
  -H "Content-Type: application/json" \\
  -d '${a2aBody}'`;
}

// Python (requests)
export function snippetPython(auth, ctx = DEFAULTS) {
  return `# pip install requests
import requests

auth = ${pyDict(auth)}

# 1) Inspect a remote MCP server
mcp = requests.post(
    "${ctx.mcpWorker}",
    json={"endpoint": "${ctx.mcpEndpoint}", "action": "initialize", "auth": auth},
    timeout=30,
)
print(mcp.json())

# 2) Validate an A2A agent card
a2a = requests.post(
    "${ctx.a2aWorker}",
    json={"url": "${ctx.a2aUrl}", "auth": auth},
    timeout=30,
)
print(a2a.json())`;
}

// .env exports (only meaningful for secret-bearing types)
export function snippetEnv(auth) {
  const lines = ['# Save the SAME secret on your MCP server / A2A agent', '# Never commit this file — add it to .gitignore'];
  if (auth.type === 'bearer') {
    lines.push(`MCP_BEARER_TOKEN=${auth.token}`, `A2A_BEARER_TOKEN=${auth.token}`);
  } else if (auth.type === 'basic') {
    lines.push(`AUTH_USERNAME=${auth.username}`, `AUTH_PASSWORD=${auth.password}`);
  } else if (auth.type === 'custom_headers') {
    for (const [k, v] of Object.entries(auth.headers || {})) {
      lines.push(`${envKey(k)}=${v}`);
    }
  } else {
    lines.push('# Auth type "none" needs no secret.');
  }
  return lines.join('\n');
}

// Map of snippet id -> { label, lang (for filename ext), build(auth, ctx) }
export const SNIPPETS = {
  auth:    { label: 'Auth config',  ext: 'js',  build: (a) => snippetAuthConfig(a) },
  headers: { label: 'HTTP headers', ext: 'json', build: (a) => snippetHeaders(a) },
  js:      { label: 'JavaScript',   ext: 'js',  build: snippetJavaScript },
  curl:    { label: 'curl',         ext: 'sh',  build: snippetCurl },
  python:  { label: 'Python',       ext: 'py',  build: snippetPython },
  env:     { label: '.env',         ext: 'env', build: (a) => snippetEnv(a) }
};

// helpers ------------------------------------------------------------------
function q(s) { return JSON.stringify(s); }

function pyDict(auth) {
  // JSON is valid Python for these shapes except true/false/null — none used here.
  return JSON.stringify(auth, null, 4);
}

function envKey(headerName) {
  return String(headerName).toUpperCase().replace(/[^A-Z0-9]+/g, '_').replace(/^_+|_+$/g, '');
}
