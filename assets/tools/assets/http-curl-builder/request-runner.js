import { buildEffectiveHeaders, buildUrlWithParams } from './curl-generator.js';

const DEFAULT_WORKER_BASE = 'https://network-api-tools.cloud-data.workers.dev';

export async function runRequest(request, { useProxy = false } = {}) {
  const started = performance.now();
  const method = (request.method || 'GET').toUpperCase();
  const targetUrl = buildUrlWithParams(request.url, request.params);
  const headers = buildEffectiveHeaders(request);

  try {
    if (useProxy && (window.RMV_API_BASE || DEFAULT_WORKER_BASE)) {
      return await runViaProxy({ ...request, url: targetUrl, headers }, started);
    }

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 15000);
    const res = await fetch(targetUrl, {
      method,
      headers,
      body: ['GET','HEAD'].includes(method) ? undefined : request.body,
      signal: controller.signal
    });
    clearTimeout(timer);
    const text = await res.text();
    return {
      ok: true,
      status: res.status,
      statusText: res.statusText,
      headers: Object.fromEntries(res.headers.entries()),
      body: text,
      timeMs: Math.round(performance.now() - started),
      sizeBytes: new Blob([text]).size,
      via: 'browser'
    };
  } catch (error) {
    return {
      ok: false,
      error: error.name === 'AbortError' ? 'Request timed out after 15 seconds' : error.message,
      corsHint: true,
      timeMs: Math.round(performance.now() - started),
      via: 'browser'
    };
  }
}

async function runViaProxy(request, started) {
  const apiBase = (window.RMV_API_BASE || DEFAULT_WORKER_BASE).replace(/\/$/, '');
  const res = await fetch(`${apiBase}/api/proxy`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(request)
  });
  const data = await res.json();
  return { ...data, ok: res.ok && !data.error, timeMs: data.timeMs ?? Math.round(performance.now() - started), via: 'proxy' };
}
