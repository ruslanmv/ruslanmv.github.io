// Talks to the Cloudflare Worker, which securely proxies JSON-RPC to a remote
// MCP HTTP/SSE server. The browser never calls the MCP server directly (that
// would leak tokens and be blocked by CORS); the Worker handles CORS, SSRF
// protection, and timeouts.

const DEFAULT_WORKER_BASE = 'https://mcp-inspector-api.cloud-data.workers.dev';

export function workerBase() {
  // Allow overriding the Worker URL without editing the bundle.
  return (window.RMV_MCP_API_BASE || window.RMV_API_BASE || DEFAULT_WORKER_BASE).replace(/\/$/, '');
}

// Builds a JSON-RPC 2.0 body purely for *preview* in the UI. The Worker builds
// the authoritative body it actually sends, but showing the user the shape is
// useful and matches what the Worker will transmit.
export function previewJsonRpc(action, { protocolVersion, toolName, toolArgs, rawBody } = {}) {
  const id = 1;
  switch (action) {
    case 'initialize':
      return {
        jsonrpc: '2.0', id, method: 'initialize',
        params: {
          protocolVersion: protocolVersion || '2025-06-18',
          capabilities: {},
          clientInfo: { name: 'ruslanmv-mcp-inspector', version: '1.0.0' }
        }
      };
    case 'tools/list':
      return { jsonrpc: '2.0', id, method: 'tools/list', params: {} };
    case 'tools/call':
      return {
        jsonrpc: '2.0', id, method: 'tools/call',
        params: { name: toolName || 'tool_name', arguments: toolArgs || {} }
      };
    case 'raw':
    default:
      return rawBody || { jsonrpc: '2.0', id, method: 'tools/list', params: {} };
  }
}

export class McpClient {
  constructor() {
    this.sessionId = null;
    this.protocolVersion = '2025-06-18';
  }

  async send(action, { endpoint, token, auth, payload, timeoutMs } = {}) {
    const body = {
      endpoint,
      action,
      payload: payload || {},
      protocolVersion: this.protocolVersion
    };
    // Auth is forwarded once, in the POST body, over HTTPS — never persisted.
    //   auth = { type: 'none' | 'bearer' | 'basic' | 'custom_headers', … }
    // `token` is kept for backward compatibility (treated as a bearer token).
    if (auth && auth.type && auth.type !== 'none') body.auth = auth;
    else if (token) body.token = token;
    if (this.sessionId) body.sessionId = this.sessionId;
    if (timeoutMs) body.timeoutMs = timeoutMs;

    const started = performance.now();
    let res;
    try {
      res = await fetch(`${workerBase()}/api/mcp`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        cache: 'no-store'
      });
    } catch (err) {
      return {
        ok: false,
        transportError: true,
        error: 'Could not reach the Inspector Worker',
        message: String(err?.message || err),
        latencyMs: Math.round(performance.now() - started)
      };
    }

    const clientLatency = Math.round(performance.now() - started);
    let data;
    try {
      data = await res.json();
    } catch {
      return { ok: false, error: 'Worker returned a non-JSON response', httpStatus: res.status, latencyMs: clientLatency };
    }

    // Persist the MCP session id (if the server issued one during initialize)
    // so subsequent tools/list and tools/call calls stay in the same session.
    if (data.sessionId) this.sessionId = data.sessionId;

    if (!('latencyMs' in data)) data.latencyMs = clientLatency;
    return data;
  }

  // Convenience wrappers ----------------------------------------------------
  initialize(ctx) {
    this.sessionId = null; // fresh handshake
    return this.send('initialize', ctx);
  }

  listTools(ctx) {
    return this.send('tools/list', ctx);
  }

  callTool(ctx, name, args) {
    return this.send('tools/call', { ...ctx, payload: { name, arguments: args || {} } });
  }

  raw(ctx, rawBody) {
    return this.send('raw', { ...ctx, payload: rawBody });
  }

  setProtocolVersion(v) {
    if (/^\d{4}-\d{2}-\d{2}$/.test(v)) this.protocolVersion = v;
  }
}
