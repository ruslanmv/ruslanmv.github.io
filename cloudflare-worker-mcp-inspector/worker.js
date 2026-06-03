/**
 * Cloudflare Worker backend for the RuslanMV MCP Server Inspector.
 *
 * Frontend tool lives at:
 *   https://ruslanmv.com/assets/tools/mcp-inspector.html
 *   https://ruslanmv.github.io/assets/tools/mcp-inspector.html   (GitHub Pages origin/fallback)
 *   http://localhost:4000/ or http://127.0.0.1:4001/ during Jekyll development
 *
 * Suggested Worker URL:
 *   https://mcp-inspector-api.<your-subdomain>.workers.dev
 *
 * Endpoints:
 *   GET     /api/health   - liveness probe + worker metadata (no secrets)
 *   OPTIONS /api/mcp      - CORS preflight
 *   POST    /api/mcp      - secure JSON-RPC proxy to a remote MCP HTTP/SSE server
 *
 * The single POST endpoint accepts:
 *   {
 *     "endpoint":  "https://example.com/mcp",   // required, https only
 *     "action":    "initialize|tools/list|tools/call|raw",
 *     "payload":   { ... },                      // tool arguments / raw JSON-RPC body
 *     "auth":      { "type": "none|bearer|basic|custom_headers", ... },
 *     "token":     "optional-user-token",       // legacy: equivalent to { type: bearer }
 *     "sessionId": "optional-mcp-session-id",    // returned by initialize, replayed by client
 *     "protocolVersion": "2025-06-18"            // optional MCP protocol version
 *   }
 *
 * Authentication (used ONLY for this request — never logged or stored):
 *   - none            : no auth
 *   - bearer          : { "token": "…" }            -> Authorization: Bearer …
 *   - basic           : { "username": "…", "password": "…" } -> Authorization: Basic …
 *   - custom_headers  : { "headers": { "X-API-Key": "…" } }
 *
 * Safety: this is NOT an open proxy.
 *   - HTTPS-only targets (rejects http://, ws://, file://, etc.)
 *   - Blocks localhost, 0.0.0.0, ::1, link-local and private IPv4/IPv6 ranges
 *   - Blocks cloud metadata IPs (169.254.169.254, fd00:ec2::254, ...)
 *   - Rejects raw-IP targets that resolve into private space
 *   - Caps request + response body size and applies a hard timeout
 *   - Best-effort per-IP rate limiting
 *   - NEVER logs, stores, echoes back, or persists the visitor's token
 */

const ALLOWED_ORIGINS = [
  // Simple static server testing
  'http://localhost:8000',
  'http://127.0.0.1:8000',

  // Jekyll local development (`make serve`)
  'http://localhost:4000',
  'http://127.0.0.1:4000',
  'http://localhost:4001',
  'http://127.0.0.1:4001',

  // Production / custom domain
  'https://ruslanmv.com',
  'https://www.ruslanmv.com',

  // GitHub Pages origin/fallback
  'https://ruslanmv.github.io'
];

const MAX_BODY_BYTES = 256_000;       // inbound payload limit
const MAX_RESPONSE_CHARS = 500_000;   // outbound (remote -> client) limit
const TIMEOUT_MS = 30_000;            // hard cap; client may request a shorter one
const MAX_TIMEOUT_MS = 120_000;
const RATE_LIMIT_MAX = 60;            // requests
const RATE_LIMIT_WINDOW_MS = 60_000;  // per minute, per IP (best-effort, per-isolate)
const ALLOWED_ACTIONS = new Set(['initialize', 'tools/list', 'tools/call', 'raw']);
const AUTH_TYPES = new Set(['none', 'bearer', 'basic', 'custom_headers']);
const BLOCKED_CUSTOM_HEADERS = new Set([
  'host',
  'content-length',
  'connection',
  'transfer-encoding',
  'upgrade',
  'cf-connecting-ip',
  'cf-ipcountry',
  'cf-ray',
  'x-forwarded-for',
  'x-real-ip'
]);
const DEFAULT_PROTOCOL_VERSION = '2025-06-18';

// Best-effort, in-memory rate limiter. Cloudflare may run several isolates, so
// this is a soft guard, not a hard quota. For strict limits use a Durable Object
// or the Cloudflare Rate Limiting product.
const rateBuckets = new Map();

export default {
  async fetch(request) {
    const url = new URL(request.url);

    if (request.method === 'OPTIONS') {
      return corsResponse(request);
    }

    if (url.pathname === '/api/health' && request.method === 'GET') {
      return json(request, {
        ok: true,
        service: 'mcp-inspector-api',
        time: new Date().toISOString(),
        // Surface only the existence of a configured service key, never the value.
        hasServiceKey: typeof globalThis.MCP_INSPECTOR_KEY === 'string',
        supportedAuthentication: ['none', 'bearer', 'basic', 'custom_headers']
      });
    }

    if (url.pathname === '/api/mcp') {
      return handleMcp(request);
    }

    return json(request, { error: 'Not found' }, 404);
  }
};

async function handleMcp(request) {
  if (request.method !== 'POST') {
    return json(request, { error: 'Use POST' }, 405);
  }

  // ----- best-effort rate limiting ------------------------------------------
  const clientIp =
    request.headers.get('CF-Connecting-IP') ||
    request.headers.get('X-Forwarded-For') ||
    'unknown';
  if (isRateLimited(clientIp)) {
    return json(request, { error: 'Rate limit exceeded. Try again shortly.' }, 429);
  }

  // ----- size guard before parsing ------------------------------------------
  const declared = Number(request.headers.get('Content-Length') || 0);
  if (declared && declared > MAX_BODY_BYTES) {
    return json(request, { error: 'Request body too large' }, 413);
  }

  const rawBody = await request.text();
  if (new TextEncoder().encode(rawBody).byteLength > MAX_BODY_BYTES) {
    return json(request, { error: 'Request body too large' }, 413);
  }

  let payload;
  try {
    payload = JSON.parse(rawBody || '{}');
  } catch {
    return json(request, { error: 'Invalid JSON body' }, 400);
  }

  const action = String(payload.action || 'raw');
  if (!ALLOWED_ACTIONS.has(action)) {
    return json(request, { error: `Unsupported action: ${action}` }, 400);
  }

  // ----- validate target endpoint -------------------------------------------
  let target;
  try {
    target = new URL(String(payload.endpoint || ''));
  } catch {
    return json(request, { error: 'Invalid endpoint URL' }, 400);
  }

  const validation = validateTarget(target);
  if (!validation.ok) {
    return json(request, { error: validation.error }, 400);
  }

  // ----- build the JSON-RPC body --------------------------------------------
  const protocolVersion = sanitizeProtocol(payload.protocolVersion);
  let rpcBody;
  try {
    rpcBody = buildJsonRpc(action, payload.payload, protocolVersion);
  } catch (err) {
    return json(request, { error: String(err?.message || err) }, 400);
  }

  // ----- assemble outbound headers (auth used here, never logged) -----------
  const authResult = buildAuthHeaders(payload.auth, payload.token);
  if (!authResult.ok) {
    return json(request, { error: authResult.error }, 400);
  }

  const headers = {
    ...authResult.headers,
    'Content-Type': 'application/json',
    // MCP Streamable HTTP servers may answer with JSON or an SSE stream.
    'Accept': 'application/json, text/event-stream',
    'MCP-Protocol-Version': protocolVersion,
    'User-Agent': 'ruslanmv-mcp-inspector/1.0'
  };
  if (payload.sessionId) {
    headers['Mcp-Session-Id'] = String(payload.sessionId);
  }

  // ----- timeout ------------------------------------------------------------
  const requested = Number(payload.timeoutMs) || TIMEOUT_MS;
  const timeoutMs = Math.min(Math.max(requested, 1000), MAX_TIMEOUT_MS);
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort('timeout'), timeoutMs);
  const started = Date.now();

  try {
    const upstream = await fetch(target.toString(), {
      method: 'POST',
      headers,
      body: JSON.stringify(rpcBody),
      signal: controller.signal,
      redirect: 'manual'
    });
    clearTimeout(timer);

    const latencyMs = Date.now() - started;
    const parsed = await readMcpResponse(upstream);
    const truncated = parsed.rawText.length > MAX_RESPONSE_CHARS;

    return json(request, {
      ok: upstream.ok,
      action,
      httpStatus: upstream.status,
      statusText: upstream.statusText,
      latencyMs,
      // A safe, non-sensitive subset of upstream headers.
      headers: headerSubset(upstream.headers),
      sessionId: upstream.headers.get('mcp-session-id') || null,
      contentType: upstream.headers.get('content-type') || '',
      transport: parsed.transport,
      json: parsed.json,
      messages: parsed.messages || undefined,
      raw: truncated ? parsed.rawText.slice(0, MAX_RESPONSE_CHARS) : parsed.rawText,
      truncated,
      requestId: rpcBody.id ?? null,
      authType: authResult.authType,
      via: 'cloudflare-worker'
    });
  } catch (error) {
    clearTimeout(timer);
    const aborted = error?.name === 'AbortError' || String(error).includes('timeout');
    return json(
      request,
      {
        ok: false,
        action,
        error: aborted ? 'Upstream request timed out' : 'Proxy request failed',
        message: String(error?.message || error),
        latencyMs: Date.now() - started
      },
      aborted ? 504 : 502
    );
  }
}

// ---------------------------------------------------------------------------
// Authentication header builder
// Supports none / bearer / basic / custom_headers, plus the legacy { token }
// payload (treated as a bearer token). Returns { ok, headers, authType } or
// { ok:false, error }. Secrets are used only for the forwarded request.
// ---------------------------------------------------------------------------
function buildAuthHeaders(auth = {}, legacyToken = '') {
  const headers = {};

  let normalizedAuth = auth && typeof auth === 'object' && !Array.isArray(auth) ? auth : {};

  // Backward compatibility with the previous inspector payload: { token: "..." }
  if ((!normalizedAuth.type || normalizedAuth.type === 'none') && legacyToken) {
    normalizedAuth = { type: 'bearer', token: legacyToken };
  }

  const authType = String(normalizedAuth.type || 'none').toLowerCase();

  if (!AUTH_TYPES.has(authType)) {
    return { ok: false, error: `Unsupported authentication type: ${authType}` };
  }

  if (authType === 'none') {
    return { ok: true, headers, authType: 'none' };
  }

  if (authType === 'bearer') {
    const token = String(normalizedAuth.token || '').trim();
    if (!token) return { ok: false, error: 'Bearer authentication requires auth.token.' };
    headers['Authorization'] = /^bearer\s+/i.test(token) ? token : `Bearer ${token}`;
    return { ok: true, headers, authType: 'bearer' };
  }

  if (authType === 'basic') {
    const username = String(normalizedAuth.username || '');
    const password = String(normalizedAuth.password || '');
    if (!username || !password) {
      return { ok: false, error: 'Basic authentication requires auth.username and auth.password.' };
    }
    headers['Authorization'] = `Basic ${btoa(`${username}:${password}`)}`;
    return { ok: true, headers, authType: 'basic' };
  }

  // custom_headers
  const customHeaders = normalizedAuth.headers;
  if (!customHeaders || typeof customHeaders !== 'object' || Array.isArray(customHeaders)) {
    return { ok: false, error: 'Custom Headers authentication requires auth.headers as an object.' };
  }
  for (const [key, value] of Object.entries(customHeaders)) {
    const headerName = String(key).trim();
    const lowerName = headerName.toLowerCase();
    if (!headerName) return { ok: false, error: 'Custom header names cannot be empty.' };
    if (BLOCKED_CUSTOM_HEADERS.has(lowerName)) {
      return { ok: false, error: `Header "${headerName}" is not allowed.` };
    }
    if (typeof value !== 'string') {
      return { ok: false, error: `Header "${headerName}" must have a string value.` };
    }
    headers[headerName] = value;
  }
  return { ok: true, headers, authType: 'custom_headers' };
}

// ---------------------------------------------------------------------------
// JSON-RPC builders
// ---------------------------------------------------------------------------
function buildJsonRpc(action, payload = {}, protocolVersion = DEFAULT_PROTOCOL_VERSION) {
  const id = Date.now() % 100000;

  if (action === 'initialize') {
    return {
      jsonrpc: '2.0',
      id,
      method: 'initialize',
      params: {
        protocolVersion,
        capabilities: (payload && payload.capabilities) || {},
        clientInfo: {
          name: 'ruslanmv-mcp-inspector',
          version: '1.0.0'
        }
      }
    };
  }

  if (action === 'tools/list') {
    return { jsonrpc: '2.0', id, method: 'tools/list', params: (payload && payload.params) || {} };
  }

  if (action === 'tools/call') {
    const name = payload && payload.name;
    if (!name) throw new Error('tools/call requires a "name"');
    return {
      jsonrpc: '2.0',
      id,
      method: 'tools/call',
      params: {
        name: String(name),
        arguments: (payload && payload.arguments) || {}
      }
    };
  }

  // action === 'raw' -> forward the client-supplied JSON-RPC body verbatim,
  // but guarantee it is shaped like a JSON-RPC 2.0 message.
  if (!payload || typeof payload !== 'object') {
    throw new Error('raw action requires a JSON-RPC object payload');
  }
  const body = { jsonrpc: '2.0', ...payload };
  if (body.id === undefined && body.method && !String(body.method).startsWith('notifications/')) {
    body.id = id;
  }
  return body;
}

// ---------------------------------------------------------------------------
// Response reader: handles both application/json and text/event-stream (SSE)
// ---------------------------------------------------------------------------
async function readMcpResponse(upstream) {
  const contentType = (upstream.headers.get('content-type') || '').toLowerCase();
  const text = await upstream.text();

  if (contentType.includes('text/event-stream')) {
    const messages = [];
    for (const block of text.split(/\r?\n\r?\n/)) {
      const dataLines = block
        .split(/\r?\n/)
        .filter((line) => line.startsWith('data:'))
        .map((line) => line.slice(5).trim());
      if (!dataLines.length) continue;
      try {
        messages.push(JSON.parse(dataLines.join('\n')));
      } catch {
        /* ignore non-JSON SSE frames (comments, keep-alives) */
      }
    }
    // The actual JSON-RPC response is the message carrying an id.
    const response =
      messages.find((m) => m && m.id !== undefined && (m.result !== undefined || m.error !== undefined)) ||
      messages[messages.length - 1] ||
      null;
    return { json: response, messages, rawText: text, transport: 'sse' };
  }

  let parsed = null;
  try {
    parsed = JSON.parse(text);
  } catch {
    /* leave parsed null; raw text still returned */
  }
  return { json: parsed, rawText: text, transport: 'http' };
}

// ---------------------------------------------------------------------------
// Target validation (SSRF guard)
// ---------------------------------------------------------------------------
function validateTarget(parsed) {
  if (parsed.protocol !== 'https:') {
    return { ok: false, error: 'Only HTTPS endpoints are allowed' };
  }

  const host = parsed.hostname.toLowerCase().replace(/^\[|\]$/g, '');

  const blockedNames = new Set(['localhost', '0.0.0.0', '::', '::1', 'ip6-localhost']);
  if (blockedNames.has(host)) {
    return { ok: false, error: 'Localhost targets are blocked' };
  }
  if (host.endsWith('.local') || host.endsWith('.internal') || host.endsWith('.localhost')) {
    return { ok: false, error: 'Local/internal hostnames are blocked' };
  }

  // IPv4 literal checks
  if (/^\d{1,3}(\.\d{1,3}){3}$/.test(host)) {
    if (isPrivateOrReservedIPv4(host)) {
      return { ok: false, error: 'Private, loopback, link-local or metadata IPs are blocked' };
    }
  }

  // IPv6 literal checks (loopback, link-local, unique-local, metadata)
  if (host.includes(':')) {
    if (
      host === '::1' ||
      host.startsWith('fe80:') ||   // link-local
      host.startsWith('fc') ||      // unique local fc00::/7
      host.startsWith('fd') ||      // unique local
      host.startsWith('fd00:ec2')   // AWS IMDS over IPv6
    ) {
      return { ok: false, error: 'Private or link-local IPv6 targets are blocked' };
    }
  }

  return { ok: true };
}

function isPrivateOrReservedIPv4(ip) {
  const parts = ip.split('.').map(Number);
  if (parts.some((n) => Number.isNaN(n) || n < 0 || n > 255)) return true; // malformed -> block
  const [a, b] = parts;
  if (a === 10) return true;                              // 10.0.0.0/8
  if (a === 127) return true;                             // loopback
  if (a === 0) return true;                               // "this" network
  if (a === 192 && b === 168) return true;                // 192.168.0.0/16
  if (a === 172 && b >= 16 && b <= 31) return true;       // 172.16.0.0/12
  if (a === 169 && b === 254) return true;                // link-local incl. 169.254.169.254 metadata
  if (a === 100 && b >= 64 && b <= 127) return true;      // CGNAT 100.64.0.0/10
  if (a >= 224) return true;                              // multicast / reserved
  return false;
}

function sanitizeProtocol(value) {
  const v = String(value || DEFAULT_PROTOCOL_VERSION);
  return /^\d{4}-\d{2}-\d{2}$/.test(v) ? v : DEFAULT_PROTOCOL_VERSION;
}

// ---------------------------------------------------------------------------
// Header helpers
// ---------------------------------------------------------------------------
function headerSubset(h) {
  const keep = ['content-type', 'mcp-session-id', 'mcp-protocol-version', 'server', 'date', 'cache-control'];
  const out = {};
  for (const key of keep) {
    const value = h.get(key);
    if (value) out[key] = value;
  }
  return out;
}

// ---------------------------------------------------------------------------
// Rate limiting
// ---------------------------------------------------------------------------
function isRateLimited(ip) {
  const now = Date.now();
  const bucket = rateBuckets.get(ip) || [];
  const recent = bucket.filter((t) => now - t < RATE_LIMIT_WINDOW_MS);
  recent.push(now);
  rateBuckets.set(ip, recent);
  // Opportunistic cleanup to avoid unbounded growth.
  if (rateBuckets.size > 5000) rateBuckets.clear();
  return recent.length > RATE_LIMIT_MAX;
}

// ---------------------------------------------------------------------------
// CORS + JSON helpers
// ---------------------------------------------------------------------------
function corsHeaders(request) {
  const origin = request.headers.get('Origin') || '';
  const allowOrigin = ALLOWED_ORIGINS.includes(origin) ? origin : ALLOWED_ORIGINS[0];
  return {
    'Access-Control-Allow-Origin': allowOrigin,
    'Access-Control-Allow-Methods': 'GET,POST,OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type, Authorization',
    'Access-Control-Max-Age': '86400',
    'Vary': 'Origin'
  };
}

function corsResponse(request) {
  return new Response(null, { status: 204, headers: corsHeaders(request) });
}

function json(request, data, status = 200) {
  return new Response(JSON.stringify(data, null, 2), {
    status,
    headers: {
      'Content-Type': 'application/json; charset=utf-8',
      ...corsHeaders(request)
    }
  });
}
