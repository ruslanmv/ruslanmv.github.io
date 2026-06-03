/**
 * Cloudflare Worker backend for RuslanMV Network & API tools.
 *
 * Frontend tools live at:
 *   https://ruslanmv.com/assets/tools/
 *   https://ruslanmv.github.io/
 *   http://localhost:4000/ or http://127.0.0.1:4001/ during Jekyll development
 *
 * Worker URL:
 *   https://network-api-tools.cloud-data.workers.dev
 *
 * Endpoints:
 *   GET  /api/ip     - returns public IP + Cloudflare request metadata
 *   POST /api/proxy  - controlled request proxy for HTTP Request & cURL Builder
 *
 * Safety: this is not a fully open proxy. It blocks localhost/private ranges,
 * limits methods, limits body/response size, and times out long requests.
 */
const ALLOWED_ORIGINS = [
  // Simple static server testing
  'http://localhost:8000',
  'http://127.0.0.1:8000',

  // Jekyll local development from `make serve`
  'http://localhost:4000',
  'http://127.0.0.1:4000',
  'http://localhost:4001',
  'http://127.0.0.1:4001',

  // Production/custom domain
  'https://ruslanmv.com',
  'https://www.ruslanmv.com',

  // GitHub Pages fallback/origin
  'https://ruslanmv.github.io'
];

const MAX_BODY_BYTES = 512_000;
const MAX_RESPONSE_CHARS = 500_000;
const TIMEOUT_MS = 15_000;
const ALLOWED_METHODS = new Set(['GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'HEAD']);
const BLOCKED_HOSTS = new Set([
  'localhost',
  '127.0.0.1',
  '0.0.0.0',
  '::1',
  '169.254.169.254'
]);

export default {
  async fetch(request) {
    const url = new URL(request.url);

    if (request.method === 'OPTIONS') {
      return corsResponse(request);
    }

    if (url.pathname === '/api/ip') {
      return handleIp(request);
    }

    if (url.pathname === '/api/proxy') {
      return handleProxy(request);
    }

    return json(request, { error: 'Not found' }, 404);
  }
};

function handleIp(request) {
  const cf = request.cf || {};
  const ip = request.headers.get('CF-Connecting-IP') || request.headers.get('X-Forwarded-For') || '';

  return json(request, {
    ip,
    country: cf.country || null,
    countryCode: cf.country || null,
    city: cf.city || null,
    region: cf.region || null,
    timezone: cf.timezone || null,
    asn: cf.asn || null,
    asOrganization: cf.asOrganization || null,
    colo: cf.colo || null,
    latitude: cf.latitude || null,
    longitude: cf.longitude || null,
    version: guessIpVersion(ip),
    userAgent: request.headers.get('User-Agent') || null,
    language: request.headers.get('Accept-Language') || null,
    detectedAt: new Date().toISOString(),
    source: 'Cloudflare Worker'
  });
}

async function handleProxy(request) {
  if (request.method !== 'POST') {
    return json(request, { error: 'Use POST' }, 405);
  }

  let payload;
  try {
    payload = await request.json();
  } catch {
    return json(request, { error: 'Invalid JSON body' }, 400);
  }

  const method = String(payload.method || 'GET').toUpperCase();
  const targetUrl = String(payload.url || '');
  const headers = sanitizeHeaders(payload.headers || {});
  const body = typeof payload.body === 'string'
    ? payload.body
    : payload.body
      ? JSON.stringify(payload.body)
      : '';

  if (!ALLOWED_METHODS.has(method)) {
    return json(request, { error: 'Method not allowed' }, 400);
  }

  let parsed;
  try {
    parsed = new URL(targetUrl);
  } catch {
    return json(request, { error: 'Invalid target URL' }, 400);
  }

  const validation = validateTarget(parsed);
  if (!validation.ok) {
    return json(request, { error: validation.error }, 400);
  }

  if (new TextEncoder().encode(body).byteLength > MAX_BODY_BYTES) {
    return json(request, { error: 'Request body too large' }, 413);
  }

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort('timeout'), TIMEOUT_MS);
  const started = Date.now();

  try {
    const upstream = await fetch(parsed.toString(), {
      method,
      headers,
      body: ['GET', 'HEAD'].includes(method) ? undefined : body,
      signal: controller.signal
    });

    clearTimeout(timer);

    const contentType = upstream.headers.get('content-type') || '';
    const text = await upstream.text();
    const truncated = text.length > MAX_RESPONSE_CHARS;

    return json(request, {
      ok: upstream.ok,
      status: upstream.status,
      statusText: upstream.statusText,
      headers: Object.fromEntries(upstream.headers.entries()),
      contentType,
      body: truncated ? text.slice(0, MAX_RESPONSE_CHARS) : text,
      truncated,
      timeMs: Date.now() - started,
      sizeBytes: new TextEncoder().encode(text).byteLength,
      via: 'cloudflare-worker'
    });
  } catch (error) {
    clearTimeout(timer);

    return json(
      request,
      {
        error: 'Proxy request failed',
        message: String(error?.message || error)
      },
      502
    );
  }
}

function validateTarget(parsed) {
  if (!['http:', 'https:'].includes(parsed.protocol)) {
    return { ok: false, error: 'Only http/https URLs are allowed' };
  }

  const host = parsed.hostname.toLowerCase();

  if (BLOCKED_HOSTS.has(host)) {
    return { ok: false, error: 'Target host blocked' };
  }

  // Block common private IPv4 ranges and link-local ranges.
  if (
    /^10\./.test(host) ||
    /^192\.168\./.test(host) ||
    /^172\.(1[6-9]|2\d|3[0-1])\./.test(host) ||
    /^169\.254\./.test(host)
  ) {
    return { ok: false, error: 'Private IP ranges are blocked' };
  }

  if (host.endsWith('.local') || host.endsWith('.internal')) {
    return { ok: false, error: 'Local/internal hostnames are blocked' };
  }

  return { ok: true };
}

function sanitizeHeaders(input) {
  const blocked = new Set([
    'host',
    'connection',
    'content-length',
    'cf-connecting-ip',
    'cf-ray',
    'x-forwarded-for',
    'x-real-ip'
  ]);

  const output = {};

  for (const [key, value] of Object.entries(input || {})) {
    const cleanKey = String(key).trim();
    const lower = cleanKey.toLowerCase();

    if (!cleanKey || blocked.has(lower)) continue;
    output[cleanKey] = String(value ?? '');
  }

  return output;
}

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
  return new Response(null, {
    status: 204,
    headers: corsHeaders(request)
  });
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

function guessIpVersion(ip) {
  return String(ip || '').includes(':') ? 'IPv6' : 'IPv4';
}
