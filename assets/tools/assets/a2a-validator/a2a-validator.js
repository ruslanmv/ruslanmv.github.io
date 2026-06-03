import { $, $$, copyText, escapeHtml, showToast } from '../shared/tool-shell.js';
import { downloadJson, timestampSlug } from '../shared/export.js';
import { saveState, loadState } from '../shared/storage.js';
import { validateAgent, checkHealth, workerBase } from './validator-client.js';
import { fieldChecks, lifecycleChecks } from './checks.js';

const URL_STATE_KEY = 'rmv-a2a-validator-url';
const AUTH_STATE_KEY = 'rmv-a2a-validator-auth-type';
const STATUS_ICON = { ok: '✓', fail: '✕', warn: '!', idle: '○' };

let lastResult = null;
let authType = 'none';

/* ----------------------------------------------------------------- *
 * Authentication UI (secrets stay in memory only — never persisted)
 * ----------------------------------------------------------------- */
function setAuthType(type) {
  authType = type;
  $$('#authSeg button').forEach((b) => b.classList.toggle('active', b.dataset.auth === type));
  $$('.auth-pane').forEach((p) => { p.hidden = p.dataset.pane !== type; });
  saveState(AUTH_STATE_KEY, type);
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

// Reads the current auth UI into the { type, … } object the Worker expects.
// Returns { ok, auth } or { ok:false, error } for client-side validation.
function buildAuth() {
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

/* ----------------------------------------------------------------- *
 * Rendering
 * ----------------------------------------------------------------- */
function renderChecklist(el, items, labelKey) {
  el.innerHTML = items.map((c) => `
    <div class="diag-row diag-${c.status}">
      <span class="diag-dot">${STATUS_ICON[c.status] || '○'}</span>
      <span class="diag-label">${escapeHtml(c[labelKey])}</span>
      <span class="diag-detail">${escapeHtml(c.detail || '')}</span>
    </div>`).join('');
}

function renderMessages(result) {
  const list = $('#msgList');
  const errors = result.errors || [];
  const warnings = result.warnings || [];
  const blocks = [];

  if (result.transportError || (result.error && !errors.length && !warnings.length)) {
    blocks.push(`<div class="msg msg-error"><span class="msg-ico">✕</span><div>${escapeHtml(result.error || 'Validation failed')}${result.message ? `<br><small>${escapeHtml(result.message)}</small>` : ''}</div></div>`);
  }
  if (result.valid && result.message) {
    blocks.push(`<div class="msg msg-ok"><span class="msg-ico">✓</span><div>${escapeHtml(result.message)}</div></div>`);
  }
  errors.forEach((e) => blocks.push(`<div class="msg msg-error"><span class="msg-ico">✕</span><div>${escapeHtml(e)}</div></div>`));
  warnings.forEach((w) => blocks.push(`<div class="msg msg-warn"><span class="msg-ico">!</span><div>${escapeHtml(w)}</div></div>`));

  list.innerHTML = blocks.length ? blocks.join('') : '<p class="placeholder">No messages.</p>';
}

function renderSkills(card) {
  const wrap = $('#skillsPreview');
  const skills = Array.isArray(card?.skills) ? card.skills : null;
  if (!skills || !skills.length) { wrap.hidden = true; wrap.innerHTML = ''; return; }
  wrap.hidden = false;
  wrap.innerHTML = `<div class="skills-head">Skills · ${skills.length}</div>` + skills.map((s) => {
    const name = typeof s?.name === 'string' ? s.name : (typeof s?.id === 'string' ? s.id : 'unnamed');
    const desc = typeof s?.description === 'string' ? s.description : '';
    return `<div class="skill-chip"><strong>${escapeHtml(name)}</strong>${desc ? `<span>${escapeHtml(desc)}</span>` : ''}</div>`;
  }).join('');
}

function renderResult(result) {
  lastResult = result;

  // verdict badge + metrics
  const badge = $('#verdictBadge');
  if (result.transportError) {
    badge.textContent = 'Unreachable'; badge.className = 'badge error';
    $('#mResult').textContent = 'ERR';
  } else if (result.valid) {
    badge.textContent = 'Valid'; badge.className = 'badge';
    $('#mResult').textContent = 'PASS';
  } else {
    badge.textContent = 'Invalid'; badge.className = 'badge error';
    $('#mResult').textContent = 'FAIL';
  }
  $('#mHttp').textContent = result.status_code ?? (result.transportError ? '—' : (result.workerStatus ?? '—'));
  $('#mErrors').textContent = (result.errors || []).length || (result.error ? 1 : 0);
  $('#mWarnings').textContent = (result.warnings || []).length;

  // resolved url + auth type used
  if (result.agent_card_url) {
    $('#resolvedRow').hidden = false;
    $('#resolvedUrl').textContent = result.agent_card_url;
    const authEl = $('#resolvedAuth');
    if (authEl) {
      const at = result.auth_type || authType;
      authEl.textContent = `auth: ${at}`;
      authEl.hidden = !at;
    }
  }

  renderMessages(result);
  renderChecklist($('#fieldList'), fieldChecks(result.agent_card), 'field');
  renderChecklist($('#lifecycleList'), lifecycleChecks(result), 'label');
  renderSkills(result.agent_card);

  $('#cardOutput').textContent = result.agent_card
    ? JSON.stringify(result.agent_card, null, 2)
    : 'No agent card returned.';
}

function renderIdle() {
  renderChecklist($('#fieldList'), fieldChecks(null), 'field');
  renderChecklist($('#lifecycleList'), lifecycleChecks(null), 'label');
}

/* ----------------------------------------------------------------- *
 * Actions
 * ----------------------------------------------------------------- */
async function runValidation() {
  const url = $('#agentUrl').value.trim();
  if (!url) { showToast('Enter an agent URL'); return; }
  try { new URL(url); } catch { showToast('That is not a valid URL'); return; }

  const built = buildAuth();
  if (!built.ok) { showToast(built.error); return; }

  saveState(URL_STATE_KEY, url); // URL only — secrets are never persisted
  const btn = $('#validateBtn');
  btn.disabled = true; const label = btn.textContent; btn.textContent = 'Validating…';
  try {
    const result = await validateAgent(url, built.auth);
    renderResult(result);
    showToast(result.valid ? 'Agent card is valid' : (result.transportError ? 'Worker unreachable' : 'Validation found issues'));
  } finally {
    btn.disabled = false; btn.textContent = label;
  }
}

function clearAll() {
  $('#agentUrl').value = '';
  // wipe in-memory secrets from the DOM
  ['#authToken', '#authUser', '#authPass'].forEach((sel) => { const el = $(sel); if (el) el.value = ''; });
  $('#hdrList').innerHTML = '';
  addHeaderRow('X-API-Key', '');
  setAuthType('none');
  $('#resolvedRow').hidden = true;
  $('#verdictBadge').textContent = 'Idle'; $('#verdictBadge').className = 'badge neutral';
  ['mResult', 'mHttp', 'mErrors', 'mWarnings'].forEach((id) => { $('#' + id).textContent = '—'; });
  $('#msgList').innerHTML = '<p class="placeholder">Enter an agent URL and run <strong>Validate agent</strong> to see the result.</p>';
  $('#cardOutput').textContent = 'No agent card yet…';
  $('#skillsPreview').hidden = true;
  lastResult = null;
  renderIdle();
  showToast('Cleared');
}

/* ----------------------------------------------------------------- *
 * Worker health badge
 * ----------------------------------------------------------------- */
async function refreshWorkerBadge() {
  const badge = $('#workerBadge');
  const ok = await checkHealth();
  if (ok) { badge.textContent = '◎ Worker online'; badge.className = 'badge'; }
  else { badge.textContent = '◎ Worker offline — set RMV_A2A_API_BASE'; badge.className = 'badge warn'; }
}

/* ----------------------------------------------------------------- */
function init() {
  const saved = loadState(URL_STATE_KEY, '');
  if (saved) $('#agentUrl').value = saved;

  // auth controls
  initRevealButtons();
  $('#authSeg').addEventListener('click', (e) => {
    const b = e.target.closest('button[data-auth]');
    if (b) setAuthType(b.dataset.auth);
  });
  $('#addHdrBtn').addEventListener('click', () => addHeaderRow());
  addHeaderRow('X-API-Key', ''); // one empty row to start
  setAuthType(loadState(AUTH_STATE_KEY, 'none'));

  $('#validateBtn').addEventListener('click', runValidation);
  $('#agentUrl').addEventListener('keydown', (e) => { if (e.key === 'Enter') runValidation(); });
  $('#clearBtn').addEventListener('click', clearAll);
  $$('.chip-btn').forEach((b) => b.addEventListener('click', () => { $('#agentUrl').value = b.dataset.ex; runValidation(); }));

  $('#copyCardBtn').addEventListener('click', () => {
    if (!lastResult?.agent_card) return showToast('No agent card to copy');
    copyText(JSON.stringify(lastResult.agent_card, null, 2), 'Agent card copied');
  });
  $('#downloadCardBtn').addEventListener('click', () => {
    if (!lastResult?.agent_card) return showToast('No agent card to download');
    downloadJson(`a2a-agent-card-${timestampSlug()}.json`, lastResult.agent_card);
    showToast('Agent card downloaded');
  });

  renderIdle();
  refreshWorkerBadge();
}

document.addEventListener('DOMContentLoaded', init);
