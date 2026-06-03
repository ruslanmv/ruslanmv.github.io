import { $, $$, copyText, escapeHtml, showToast } from '../shared/tool-shell.js';
import { downloadFile, timestampSlug } from '../shared/export.js';
import { saveState, loadState } from '../shared/storage.js';
import { buildInspectorCommand, buildShellScript } from './command-builder.js';
import { McpClient, workerBase } from './mcp-client.js';
import { buildDiagnostics } from './diagnostics.js';

// SECURITY: tokens are NEVER written to localStorage, the URL, or the console.
// They live only in these input fields (and the in-flight POST body). Only the
// non-sensitive command-builder fields are persisted.
const CB_STATE_KEY = 'rmv-mcp-inspector-builder';

const client = new McpClient();
const remoteResults = { initialize: null, tools: null, call: null };

/* --------------------------------------------------------------------- *
 * Tabs
 * --------------------------------------------------------------------- */
function initTabs() {
  $$('.tab').forEach((tab) => {
    tab.addEventListener('click', () => {
      $$('.tab').forEach((t) => t.classList.remove('active'));
      tab.classList.add('active');
      const name = tab.dataset.tab;
      $$('.panel').forEach((p) => { p.hidden = p.dataset.panel !== name; });
      if (name === 'diagnostics') renderDiagnostics();
    });
  });
}

/* --------------------------------------------------------------------- *
 * Token masking (reveal toggle)
 * --------------------------------------------------------------------- */
function initRevealButtons() {
  $$('.reveal-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      const input = document.getElementById(btn.dataset.reveal);
      if (!input) return;
      input.type = input.type === 'password' ? 'text' : 'password';
      btn.classList.toggle('on', input.type === 'text');
    });
  });
}

/* --------------------------------------------------------------------- *
 * Tab 1 — Command Builder
 * --------------------------------------------------------------------- */
function readBuilderFields() {
  return {
    serverUrl: $('#cbUrl').value.trim(),
    token: $('#cbToken').value.trim(),
    timeout: $('#cbTimeout').value.trim(),
    pythonPath: $('#cbPython').value.trim(),
    wrapperModule: $('#cbWrapper').value.trim()
  };
}

function refreshCommand() {
  const fields = readBuilderFields();
  $('#cbOutput').textContent = buildInspectorCommand(fields);
  // Persist everything EXCEPT the token.
  const { token, ...safe } = fields;
  saveState(CB_STATE_KEY, safe);
}

function initCommandBuilder() {
  const saved = loadState(CB_STATE_KEY, null);
  if (saved) {
    if (saved.serverUrl) $('#cbUrl').value = saved.serverUrl;
    if (saved.timeout) $('#cbTimeout').value = saved.timeout;
    if (saved.pythonPath) $('#cbPython').value = saved.pythonPath;
    if (saved.wrapperModule) $('#cbWrapper').value = saved.wrapperModule;
  }
  ['#cbUrl', '#cbToken', '#cbTimeout', '#cbPython', '#cbWrapper'].forEach((sel) =>
    $(sel).addEventListener('input', refreshCommand)
  );

  $('#cbCopy').addEventListener('click', () => copyText(buildInspectorCommand(readBuilderFields()), 'Command copied'));
  $('#cbDownload').addEventListener('click', () => {
    downloadFile(`mcp-inspector-${timestampSlug()}.sh`, buildShellScript(readBuilderFields()), 'text/x-shellscript;charset=utf-8');
    showToast('Shell script downloaded');
  });
  $('#cbClear').addEventListener('click', () => {
    $('#cbUrl').value = '';
    $('#cbToken').value = '';
    $('#cbTimeout').value = '120';
    $('#cbPython').value = '/path/to/venv/bin/python';
    $('#cbWrapper').value = 'mcpgateway.wrapper';
    refreshCommand();
    showToast('Cleared');
  });

  refreshCommand();
}

/* --------------------------------------------------------------------- *
 * Tab 2 — Remote HTTP Test
 * --------------------------------------------------------------------- */
// Authentication UI state. Secrets live only in the DOM inputs (and the
// in-flight POST body) — they are never written to localStorage or the URL.
let authType = 'none';

function setAuthType(type) {
  authType = type;
  $$('#authSeg button').forEach((b) => b.classList.toggle('active', b.dataset.auth === type));
  $$('.auth-pane').forEach((p) => { p.hidden = p.dataset.pane !== type; });
}

function addHeaderRow(key = '', value = '') {
  const list = $('#hdrList');
  const row = document.createElement('div');
  row.className = 'hdr-row';
  row.innerHTML = `
    <input class="hdr-key" type="text" placeholder="X-API-Key" autocomplete="off" spellcheck="false" />
    <input class="hdr-val" type="text" placeholder="value" autocomplete="off" spellcheck="false" />
    <button type="button" class="btn small ghost hdr-del" aria-label="Remove header">✕</button>`;
  row.querySelector('.hdr-key').value = key;
  row.querySelector('.hdr-val').value = value;
  row.querySelector('.hdr-del').addEventListener('click', () => row.remove());
  list.appendChild(row);
}

// Reads the auth UI into the { type, … } object the Worker expects.
// Returns { ok, auth } or { ok:false, error } for client-side validation.
function buildRemoteAuth() {
  if (authType === 'none') return { ok: true, auth: { type: 'none' } };

  if (authType === 'bearer') {
    const token = $('#authToken').value.trim();
    if (!token) return { ok: false, error: 'Enter a bearer token (or choose None).' };
    return { ok: true, auth: { type: 'bearer', token } };
  }

  if (authType === 'basic') {
    const username = $('#authUser').value;
    const password = $('#authPass').value;
    if (!username || !password) return { ok: false, error: 'Basic auth needs a username and password.' };
    return { ok: true, auth: { type: 'basic', username, password } };
  }

  if (authType === 'custom_headers') {
    const headers = {};
    let count = 0;
    for (const row of $$('#hdrList .hdr-row')) {
      const key = row.querySelector('.hdr-key').value.trim();
      const value = row.querySelector('.hdr-val').value;
      if (!key) continue;
      headers[key] = value;
      count++;
    }
    if (!count) return { ok: false, error: 'Add at least one custom header (or choose None).' };
    return { ok: true, auth: { type: 'custom_headers', headers } };
  }

  return { ok: true, auth: { type: 'none' } };
}

function remoteContext() {
  return {
    endpoint: $('#rtUrl').value.trim(),
    auth: buildRemoteAuth().auth || { type: 'none' },
    timeoutMs: Math.max(1, Number($('#rtTimeout').value) || 30) * 1000
  };
}

function setMetrics(res) {
  $('#mStatus').textContent = res.transportError ? 'ERR' : (res.httpStatus ?? '—');
  $('#mLatency').textContent = res.latencyMs != null ? `${res.latencyMs} ms` : '—';
  $('#mTransport').textContent = res.transport || (res.transportError ? 'unreachable' : '—');
  $('#mSession').textContent = res.sessionId ? `${String(res.sessionId).slice(0, 8)}…` : '—';
  const authEl = $('#mAuth');
  if (authEl) authEl.textContent = res.authType || authType || '—';
}

function showResponse(res) {
  setMetrics(res);
  // Prefer the parsed JSON-RPC message; fall back to the raw payload / error.
  const view = res.json ?? res.messages ?? res;
  $('#rtOutput').textContent = JSON.stringify(view, null, 2);
  renderDiagnostics();
}

function validateBeforeSend() {
  const url = $('#rtUrl').value.trim();
  if (!url) { showToast('Enter an endpoint URL'); return false; }
  try {
    const u = new URL(url);
    if (u.protocol !== 'https:') { showToast('Endpoint must be HTTPS'); return false; }
  } catch {
    showToast('Invalid endpoint URL'); return false;
  }
  const a = buildRemoteAuth();
  if (!a.ok) { showToast(a.error); return false; }
  return true;
}

async function withBusy(btn, fn) {
  if (btn) { btn.disabled = true; btn.dataset.label = btn.textContent; btn.textContent = '…'; }
  try { return await fn(); }
  finally { if (btn) { btn.disabled = false; btn.textContent = btn.dataset.label; } }
}

function populateTools(res) {
  const select = $('#rtTool');
  const tools = res?.json?.result?.tools;
  if (!Array.isArray(tools) || !tools.length) {
    select.innerHTML = '<option value="">— no tools returned —</option>';
    return;
  }
  select.innerHTML = tools
    .map((t) => `<option value="${escapeHtml(t.name)}">${escapeHtml(t.name)}</option>`)
    .join('');
  // Pre-fill the arguments editor with the first tool's input schema skeleton.
  prefillArgs(tools[0]);
  select.onchange = () => {
    const tool = tools.find((x) => x.name === select.value);
    if (tool) prefillArgs(tool);
  };
}

function prefillArgs(tool) {
  const props = tool?.inputSchema?.properties;
  if (!props) { $('#rtArgs').value = '{}'; return; }
  const skeleton = {};
  for (const key of Object.keys(props)) skeleton[key] = props[key].default ?? '';
  $('#rtArgs').value = JSON.stringify(skeleton, null, 2);
}

function initRemoteTest() {
  $('#rtProtocol').addEventListener('change', (e) => client.setProtocolVersion(e.target.value));

  // auth controls
  $('#authSeg').addEventListener('click', (e) => {
    const b = e.target.closest('button[data-auth]');
    if (b) setAuthType(b.dataset.auth);
  });
  $('#addHdrBtn').addEventListener('click', () => addHeaderRow());
  addHeaderRow('X-API-Key', '');
  setAuthType('none');

  $('#actInit').addEventListener('click', (e) => withBusy(e.target, async () => {
    if (!validateBeforeSend()) return;
    client.setProtocolVersion($('#rtProtocol').value);
    const res = await client.initialize(remoteContext());
    remoteResults.initialize = res;
    remoteResults.tools = null;
    remoteResults.call = null;
    showResponse(res);
    showToast(res.ok ? 'Initialized' : 'Initialize failed');
  }));

  $('#actList').addEventListener('click', (e) => withBusy(e.target, async () => {
    if (!validateBeforeSend()) return;
    const res = await client.listTools(remoteContext());
    remoteResults.tools = res;
    populateTools(res);
    showResponse(res);
    showToast(res.ok ? 'Tools listed' : 'List failed');
  }));

  $('#actCall').addEventListener('click', (e) => withBusy(e.target, async () => {
    if (!validateBeforeSend()) return;
    const name = $('#rtTool').value;
    if (!name) { showToast('Pick a tool (run List tools first)'); return; }
    let args;
    try { args = JSON.parse($('#rtArgs').value || '{}'); }
    catch { showToast('Arguments must be valid JSON'); return; }
    const res = await client.callTool(remoteContext(), name, args);
    remoteResults.call = res;
    showResponse(res);
    showToast(res.ok ? 'Tool called' : 'Call failed');
  }));

  $('#actRaw').addEventListener('click', (e) => withBusy(e.target, async () => {
    if (!validateBeforeSend()) return;
    let body;
    try { body = JSON.parse($('#rtRaw').value || '{}'); }
    catch { showToast('Raw body must be valid JSON'); return; }
    const res = await client.raw(remoteContext(), body);
    showResponse(res);
    showToast(res.ok ? 'Sent' : 'Request failed');
  }));

  $('#rtCopy').addEventListener('click', () => copyText($('#rtOutput').textContent, 'Response copied'));
}

/* --------------------------------------------------------------------- *
 * Tab 3 — Diagnostics
 * --------------------------------------------------------------------- */
const STATUS_ICON = { ok: '✓', fail: '✕', warn: '!', idle: '○' };

function renderDiagnostics() {
  const list = $('#diagList');
  if (!list) return;
  const checks = buildDiagnostics({
    url: $('#rtUrl').value.trim() || $('#cbUrl').value.trim(),
    // truthy when remote auth is configured, else fall back to the builder token
    token: (authType !== 'none' ? authType : '') || $('#cbToken').value.trim(),
    timeout: $('#rtTimeout')?.value || $('#cbTimeout')?.value,
    results: remoteResults
  });
  list.innerHTML = checks.map((c) => `
    <div class="diag-row diag-${c.status}">
      <span class="diag-dot">${STATUS_ICON[c.status] || '○'}</span>
      <span class="diag-label">${escapeHtml(c.label)}</span>
      <span class="diag-detail">${escapeHtml(c.detail || '')}</span>
    </div>`).join('');
}

function initDiagnostics() {
  $('#diagRun').addEventListener('click', renderDiagnostics);
  renderDiagnostics();
}

/* --------------------------------------------------------------------- *
 * Worker health badge
 * --------------------------------------------------------------------- */
async function checkWorker() {
  const badge = $('#workerBadge');
  try {
    const res = await fetch(`${workerBase()}/api/health`, { cache: 'no-store' });
    if (res.ok) {
      badge.textContent = '◎ Worker online';
      badge.className = 'badge';
    } else {
      throw new Error(String(res.status));
    }
  } catch {
    badge.textContent = '◎ Worker offline — set RMV_MCP_API_BASE';
    badge.className = 'badge warn';
  }
}

/* --------------------------------------------------------------------- */
function init() {
  initTabs();
  initRevealButtons();
  initCommandBuilder();
  initRemoteTest();
  initDiagnostics();
  checkWorker();
}

document.addEventListener('DOMContentLoaded', init);
