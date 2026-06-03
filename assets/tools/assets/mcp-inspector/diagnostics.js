// Pure diagnostics logic — given the current state of a remote test session,
// produce a list of checks. No network calls here; the controller feeds in the
// results of the Worker round-trips.

const PRIVATE_HOST = /(^localhost$)|(^127\.)|(^0\.0\.0\.0$)|(^10\.)|(^192\.168\.)|(^172\.(1[6-9]|2\d|3[01])\.)|(^169\.254\.)|(\.local$)|(\.internal$)/i;

export function urlChecks(rawUrl) {
  const value = String(rawUrl || '').trim();
  let parsed = null;
  try { parsed = new URL(value); } catch { /* invalid */ }

  return {
    provided: value.length > 0,
    valid: !!parsed,
    https: !!parsed && parsed.protocol === 'https:',
    notPrivate: !!parsed && !PRIVATE_HOST.test(parsed.hostname.toLowerCase().replace(/^\[|\]$/g, ''))
  };
}

// state = {
//   url, token, timeout,
//   results: { initialize, tools, call }  // each: { status: 'ok'|'fail'|'pending', detail }
// }
export function buildDiagnostics(state = {}) {
  const u = urlChecks(state.url);
  const r = state.results || {};
  const status = (res) => (res ? (res.ok ? 'ok' : 'fail') : 'idle');

  return [
    {
      key: 'url-format',
      label: 'URL format valid',
      status: !u.provided ? 'idle' : (u.valid ? 'ok' : 'fail'),
      detail: !u.provided ? 'Enter an endpoint URL' : (u.valid ? 'Parses as a valid URL' : 'Not a valid URL')
    },
    {
      key: 'https',
      label: 'HTTPS endpoint',
      status: !u.valid ? 'idle' : (u.https ? 'ok' : 'fail'),
      detail: u.https ? 'Secure transport' : 'The Worker rejects non-HTTPS endpoints'
    },
    {
      key: 'not-private',
      label: 'Public target (no localhost / private IP)',
      status: !u.valid ? 'idle' : (u.notPrivate ? 'ok' : 'fail'),
      detail: u.notPrivate ? 'Reachable from the Worker' : 'Localhost & private ranges are blocked — use the local command instead'
    },
    {
      key: 'token',
      label: 'Authentication provided',
      status: state.token ? 'ok' : 'warn',
      detail: state.token ? 'Auth configured (forwarded once)' : 'Optional — some servers are open'
    },
    {
      key: 'timeout',
      label: 'Timeout configured',
      status: Number(state.timeout) > 0 ? 'ok' : 'warn',
      detail: Number(state.timeout) > 0 ? `${state.timeout}s tool-call timeout` : 'Using default timeout'
    },
    {
      key: 'cors',
      label: 'CORS handled by Worker',
      status: 'ok',
      detail: 'All requests proxied through the Cloudflare Worker'
    },
    {
      key: 'reachable',
      label: 'Endpoint reachable',
      status: r.initialize ? (r.initialize.transportError ? 'fail' : 'ok') : 'idle',
      detail: !r.initialize ? 'Run Initialize' :
        (r.initialize.transportError ? 'Worker could not reach the server' :
          `HTTP ${r.initialize.httpStatus ?? '—'} · ${r.initialize.latencyMs ?? '—'} ms`)
    },
    {
      key: 'initialize',
      label: 'Initialize response valid',
      status: status(r.initialize),
      detail: validityDetail(r.initialize, 'serverInfo')
    },
    {
      key: 'tools-list',
      label: 'tools/list response valid',
      status: status(r.tools),
      detail: r.tools && r.tools.json && r.tools.json.result
        ? `${(r.tools.json.result.tools || []).length} tool(s) discovered`
        : validityDetail(r.tools)
    },
    {
      key: 'tool-call',
      label: 'Tool call response valid',
      status: status(r.call),
      detail: validityDetail(r.call, 'content')
    }
  ];
}

function validityDetail(res, expectKey) {
  if (!res) return 'Not run yet';
  if (res.transportError) return 'Transport error';
  if (res.json && res.json.error) {
    return `JSON-RPC error ${res.json.error.code ?? ''}: ${res.json.error.message || ''}`.trim();
  }
  if (res.json && res.json.result) {
    if (expectKey === 'serverInfo' && res.json.result.serverInfo) {
      const s = res.json.result.serverInfo;
      return `${s.name || 'server'} ${s.version || ''}`.trim();
    }
    return 'Valid JSON-RPC result';
  }
  return res.ok ? 'Responded, but no result field' : `HTTP ${res.httpStatus ?? '—'}`;
}
