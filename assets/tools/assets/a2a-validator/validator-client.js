// Talks to the A2A Validator Cloudflare Worker. The Worker fetches the agent's
// `/.well-known/agent.json` card, validates its shape, and returns a structured
// result. The browser never fetches the agent card directly (CORS + SSRF), so
// all requests go through the Worker.

const DEFAULT_WORKER_BASE = 'https://a2a-validator.cloud-data.workers.dev';

export function workerBase() {
  // Override without editing the bundle, e.g. window.RMV_A2A_API_BASE = '...'.
  return (window.RMV_A2A_API_BASE || window.RMV_API_BASE || DEFAULT_WORKER_BASE).replace(/\/$/, '');
}

export async function checkHealth() {
  try {
    const res = await fetch(`${workerBase()}/health`, { cache: 'no-store' });
    return res.ok;
  } catch {
    return false;
  }
}

// Returns the Worker's structured result, augmented with client-side timing and
// the Worker's own HTTP status. Network failures are returned as a soft object
// rather than thrown so the UI can render a friendly message.
//
// `auth` is one of:
//   { type: 'none' }
//   { type: 'bearer', token: '…' }
//   { type: 'basic', username: '…', password: '…' }
//   { type: 'custom_headers', headers: { 'X-API-Key': '…' } }
// Always sent via POST so secrets never appear in the URL.
export async function validateAgent(url, auth = { type: 'none' }) {
  const started = performance.now();
  let res;
  try {
    res = await fetch(`${workerBase()}/validate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url, auth }),
      cache: 'no-store'
    });
  } catch (err) {
    return {
      valid: false,
      transportError: true,
      error: 'Could not reach the A2A Validator Worker.',
      message: String(err?.message || err),
      latencyMs: Math.round(performance.now() - started)
    };
  }

  const latencyMs = Math.round(performance.now() - started);
  let data;
  try {
    data = await res.json();
  } catch {
    return { valid: false, error: 'Worker returned a non-JSON response.', workerStatus: res.status, latencyMs };
  }

  return { workerStatus: res.status, latencyMs, ...data };
}
