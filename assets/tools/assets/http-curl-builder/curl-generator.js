export function buildUrlWithParams(url, params = []) {
  if (!url) return '';
  let parsed;
  try { parsed = new URL(url); }
  catch { return url; }
  params.filter(p => p.enabled && p.key).forEach(p => parsed.searchParams.set(p.key, p.value ?? ''));
  return parsed.toString();
}

export function headersArrayToObject(headers = []) {
  const out = {};
  headers.filter(h => h.enabled && h.key).forEach(h => out[h.key] = h.value ?? '');
  return out;
}

export function buildEffectiveHeaders(request) {
  const headers = headersArrayToObject(request.headers);
  if (request.auth?.type === 'bearer' && request.auth.token) headers.Authorization = `Bearer ${request.auth.token}`;
  if (request.auth?.type === 'basic' && (request.auth.username || request.auth.password)) {
    headers.Authorization = `Basic ${btoa(`${request.auth.username || ''}:${request.auth.password || ''}`)}`;
  }
  if (request.bodyMode === 'json' && request.body && !headers['Content-Type'] && !headers['content-type']) headers['Content-Type'] = 'application/json';
  return headers;
}

function shellQuote(str) {
  const s = String(str ?? '');
  return `'${s.replace(/'/g, "'\\''")}'`;
}

export function maskSecrets(text) {
  return String(text || '').replace(/(Bearer\s+)[A-Za-z0-9._\-]+/g, '$1****************').replace(/(gh[pousr]_[A-Za-z0-9_]+)/g, 'ghp_****************');
}

export function buildCurl(request, { mask = true } = {}) {
  const method = (request.method || 'GET').toUpperCase();
  const url = buildUrlWithParams(request.url, request.params);
  const headers = buildEffectiveHeaders(request);
  const lines = [`curl -X ${method}`];

  Object.entries(headers).forEach(([key, value]) => {
    lines.push(`  -H ${shellQuote(`${key}: ${value}`)}`);
  });

  if (request.body && !['GET','HEAD'].includes(method)) {
    lines.push(`  --data ${shellQuote(request.body)}`);
  }

  lines.push(`  ${shellQuote(url)}`);
  const curl = lines.join(' \\\n');
  return mask ? maskSecrets(curl) : curl;
}
