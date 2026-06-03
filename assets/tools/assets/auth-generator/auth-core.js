// MCP / A2A Authentication core — the reusable generator logic, as an ES module.
// Generates auth configs (none / bearer / basic / custom_headers) with
// cryptographically secure random secrets, converts them to HTTP headers, and
// masks secrets for safe display. Pure + framework-free; runs in any modern
// browser (uses the Web Crypto API).

export const DEFAULT_TOKEN_BYTES = 32;

function arrayBufferToBase64Url(buffer) {
  const bytes = new Uint8Array(buffer);
  let binary = '';
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/g, '');
}

export function generateSecureToken(byteLength = DEFAULT_TOKEN_BYTES) {
  const bytes = new Uint8Array(byteLength);
  crypto.getRandomValues(bytes);
  return arrayBufferToBase64Url(bytes);
}

export function generateNoAuth() {
  return { type: 'none' };
}

export function generateBearerAuth(options = {}) {
  const token = options.token || generateSecureToken(options.byteLength || 32);
  return { type: 'bearer', token };
}

export function generateBasicAuth(options = {}) {
  const username = options.username || 'demo-user';
  const password = options.password || generateSecureToken(options.byteLength || 24);
  return { type: 'basic', username, password };
}

export function generateCustomHeadersAuth(options = {}) {
  const headerName = options.headerName || 'X-API-Key';
  const headerValue = options.headerValue || generateSecureToken(options.byteLength || 32);
  return { type: 'custom_headers', headers: { [headerName]: headerValue } };
}

export function generateAuthConfig(authType, options = {}) {
  switch (authType) {
    case 'none': return generateNoAuth();
    case 'bearer': return generateBearerAuth(options);
    case 'basic': return generateBasicAuth(options);
    case 'custom_headers': return generateCustomHeadersAuth(options);
    default: throw new Error(`Unsupported authentication type: ${authType}`);
  }
}

export function authToHttpHeaders(auth) {
  if (!auth || auth.type === 'none') return {};
  if (auth.type === 'bearer') return { Authorization: `Bearer ${auth.token}` };
  if (auth.type === 'basic') {
    const encoded = btoa(`${auth.username}:${auth.password}`);
    return { Authorization: `Basic ${encoded}` };
  }
  if (auth.type === 'custom_headers') return { ...auth.headers };
  throw new Error(`Unsupported authentication type: ${auth.type}`);
}

export function maskSecret(value) {
  if (!value) return '';
  const text = String(value);
  if (text.length <= 8) return '••••••••';
  return `${text.slice(0, 4)}••••••••${text.slice(-4)}`;
}

export function maskAuthConfig(auth) {
  if (!auth || auth.type === 'none') return { type: 'none' };
  if (auth.type === 'bearer') return { type: 'bearer', token: maskSecret(auth.token) };
  if (auth.type === 'basic') {
    return { type: 'basic', username: auth.username, password: maskSecret(auth.password) };
  }
  if (auth.type === 'custom_headers') {
    const maskedHeaders = {};
    for (const [key, value] of Object.entries(auth.headers || {})) maskedHeaders[key] = maskSecret(value);
    return { type: 'custom_headers', headers: maskedHeaders };
  }
  return { type: auth.type };
}

export function maskHttpHeaders(headers) {
  const out = {};
  for (const [k, v] of Object.entries(headers || {})) {
    // keep the scheme word ("Bearer"/"Basic") visible, mask the credential
    const m = /^(Bearer|Basic)\s+(.+)$/.exec(v);
    out[k] = m ? `${m[1]} ${maskSecret(m[2])}` : maskSecret(v);
  }
  return out;
}
