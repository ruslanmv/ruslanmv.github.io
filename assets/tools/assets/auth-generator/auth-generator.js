import { $, $$, copyText, showToast } from '../shared/tool-shell.js';
import { downloadFile, timestampSlug } from '../shared/export.js';
import { saveState, loadState } from '../shared/storage.js';
import { generateAuthConfig, authToHttpHeaders, maskAuthConfig, maskHttpHeaders } from './auth-core.js';
import { SNIPPETS, DEFAULTS } from './snippets.js';

// Non-secret preferences only — generated secrets are NEVER persisted.
const PREFS_KEY = 'rmv-auth-generator-prefs';

let authType = 'none';
let snippetId = 'auth';
let revealed = false;
let currentAuth = { type: 'none' };

/* ----------------------------------------------------------------- *
 * Segmented controls
 * ----------------------------------------------------------------- */
function setAuthType(type) {
  authType = type;
  $$('#authSeg button').forEach((b) => b.classList.toggle('active', b.dataset.auth === type));
  $$('.auth-pane').forEach((p) => { p.hidden = p.dataset.pane !== type; });
}

function setSnippet(id) {
  snippetId = id;
  $$('#snippetSeg button').forEach((b) => b.classList.toggle('active', b.dataset.snip === id));
  renderSnippet();
}

/* ----------------------------------------------------------------- *
 * Generation
 * ----------------------------------------------------------------- */
function readOptions() {
  const byteLength = Number($('#byteLen').value) || 32;
  switch (authType) {
    case 'bearer':
      return { byteLength, token: $('#bearerToken').value.trim() || undefined };
    case 'basic':
      return {
        byteLength,
        username: $('#basicUser').value.trim() || undefined,
        password: $('#basicPass').value.trim() || undefined
      };
    case 'custom_headers':
      return {
        byteLength,
        headerName: $('#hdrName').value.trim() || undefined,
        headerValue: $('#hdrValue').value.trim() || undefined
      };
    default:
      return { byteLength };
  }
}

function snippetContext() {
  const kind = $('#endpointKind').value;
  // The snippet builders accept the full DEFAULTS; "kind" trims which calls we show.
  return { ...DEFAULTS, kind };
}

function generate() {
  try {
    currentAuth = generateAuthConfig(authType, readOptions());
  } catch (err) {
    showToast(String(err.message || err));
    return;
  }
  saveState(PREFS_KEY, { authType, snippetId, byteLen: $('#byteLen').value, endpointKind: $('#endpointKind').value });
  render();
  showToast(authType === 'none' ? 'Config ready' : 'Secret generated');
}

/* ----------------------------------------------------------------- *
 * Rendering
 * ----------------------------------------------------------------- */
function render() {
  const authView = revealed ? currentAuth : maskAuthConfig(currentAuth);
  $('#authOut').textContent = JSON.stringify(authView, null, 2);

  const headers = authToHttpHeaders(currentAuth);
  const headersView = revealed ? headers : maskHttpHeaders(headers);
  $('#headersOut').textContent = Object.keys(headersView).length ? JSON.stringify(headersView, null, 2) : '{}  // type "none" sends no auth header';

  $('#revealBtn').textContent = revealed ? '🙈 Hide' : '👁 Reveal';
  renderSnippet();
}

// The on-screen snippet respects the reveal toggle; copy/download use the real one.
function renderSnippet() {
  const def = SNIPPETS[snippetId];
  if (!def) return;
  const real = def.build(currentAuth, snippetContext());
  const shown = revealed ? real : maskSnippet(real);
  $('#snippetOut').textContent = shown;
}

// Mask any long base64url-ish secret runs for safe on-screen display.
function maskSnippet(text) {
  return text.replace(/[A-Za-z0-9._\-+/]{16,}/g, (m) => {
    // leave URLs / module names mostly alone: only mask high-entropy tokens
    if (/^https?:|workers\.dev|example\.com|modelcontextprotocol|mcpgateway|cloud-data/.test(m)) return m;
    return `${m.slice(0, 4)}••••••••${m.slice(-4)}`;
  });
}

/* ----------------------------------------------------------------- *
 * Actions
 * ----------------------------------------------------------------- */
function currentSnippetText() {
  const def = SNIPPETS[snippetId];
  return def ? def.build(currentAuth, snippetContext()) : '';
}

function init() {
  // restore non-secret prefs
  const prefs = loadState(PREFS_KEY, null);
  if (prefs) {
    if (prefs.byteLen) $('#byteLen').value = prefs.byteLen;
    if (prefs.endpointKind) $('#endpointKind').value = prefs.endpointKind;
  }

  $('#authSeg').addEventListener('click', (e) => {
    const b = e.target.closest('button[data-auth]');
    if (b) { setAuthType(b.dataset.auth); }
  });
  $('#snippetSeg').addEventListener('click', (e) => {
    const b = e.target.closest('button[data-snip]');
    if (b) setSnippet(b.dataset.snip);
  });
  $('#endpointKind').addEventListener('change', renderSnippet);

  $('#genBtn').addEventListener('click', generate);
  $('#regenBtn').addEventListener('click', () => {
    // clear manual fields so a fresh secret is generated
    ['#bearerToken', '#basicPass', '#hdrValue'].forEach((s) => { const el = $(s); if (el) el.value = ''; });
    generate();
  });

  $('#revealBtn').addEventListener('click', () => { revealed = !revealed; render(); });
  $('#copyAuthBtn').addEventListener('click', () => copyText(JSON.stringify(currentAuth, null, 2), 'Auth config copied (real secret)'));
  $('#copySnippetBtn').addEventListener('click', () => copyText(currentSnippetText(), 'Snippet copied (real secret)'));
  $('#downloadSnippetBtn').addEventListener('click', () => {
    const def = SNIPPETS[snippetId];
    if (!def) return;
    const name = def.ext === 'env' ? '.env' : `mcp-a2a-auth-${snippetId}-${timestampSlug()}.${def.ext}`;
    downloadFile(name, currentSnippetText(), 'text/plain;charset=utf-8');
    showToast('Snippet downloaded');
  });

  setAuthType('none');
  setSnippet('auth');
  generate();
}

document.addEventListener('DOMContentLoaded', init);
