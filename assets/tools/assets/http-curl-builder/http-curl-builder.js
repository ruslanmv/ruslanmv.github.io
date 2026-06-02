import { $, $$, copyText, escapeHtml, formatBytes, formatDateTime, showToast, safeJsonParse } from '../shared/tool-shell.js';
import { downloadFile, downloadJson, timestampSlug } from '../shared/export.js';
import { saveState, loadState, pushRecent } from '../shared/storage.js';
import { buildCurl, buildEffectiveHeaders, buildUrlWithParams } from './curl-generator.js';
import { buildFetchCode, buildPythonRequestsCode, buildAxiosCode } from './code-generators.js';
import { runRequest } from './request-runner.js';

const LS_KEY = 'rmv-http-curl-builder-state';
let lastResult = null;
let activeCodeTab = 'curl';

const defaultState = {
  method: 'GET',
  url: 'https://api.github.com/repos/octocat/Hello-World/issues',
  params: [
    { enabled: true, key: 'state', value: 'open', description: 'Filter by state' },
    { enabled: true, key: 'per_page', value: '5', description: 'Results per page (max 100)' }
  ],
  headers: [{ enabled: true, key: 'Accept', value: 'application/vnd.github+json', description: '' }],
  auth: { type: 'none', token: '', username: '', password: '', apiKeyName: '', apiKeyValue: '', apiKeyLocation: 'header' },
  bodyMode: 'json',
  body: '{}',
  environment: 'dev',
  useProxy: false
};

let state = loadState(LS_KEY, defaultState);

function rowHtml(type, row, index) {
  return `<tr data-${type}-row="${index}">
    <td style="width:44px"><input type="checkbox" data-field="enabled" ${row.enabled ? 'checked' : ''}></td>
    <td><input data-field="key" value="${escapeHtml(row.key || '')}" placeholder="Enter key"></td>
    <td><input data-field="value" value="${escapeHtml(row.value || '')}" placeholder="Enter value"></td>
    <td><input data-field="description" value="${escapeHtml(row.description || '')}" placeholder="Enter description"></td>
    <td style="width:48px"><button class="btn small" data-remove-${type}="${index}" title="Remove">⌫</button></td>
  </tr>`;
}

function renderRows() {
  $('#paramsBody').innerHTML = state.params.map((r, i) => rowHtml('param', r, i)).join('');
  $('#headersBody').innerHTML = state.headers.map((r, i) => rowHtml('header', r, i)).join('');
}

function collectRows(type) {
  return $$(`[data-${type}-row]`).map(row => {
    const data = {};
    $$('[data-field]', row).forEach(input => data[input.dataset.field] = input.type === 'checkbox' ? input.checked : input.value);
    return data;
  });
}

function getStateFromForm() {
  state.method = $('#method').value;
  state.url = $('#url').value.trim();
  state.params = collectRows('param');
  state.headers = collectRows('header');
  state.auth = {
    type: $('#authType').value,
    token: $('#authToken').value,
    username: $('#basicUser').value,
    password: $('#basicPass').value,
    apiKeyName: $('#apiKeyName').value,
    apiKeyValue: $('#apiKeyValue').value,
    apiKeyLocation: $('#apiKeyLocation').value
  };
  state.bodyMode = $('.body-mode.active')?.dataset.mode || 'json';
  state.body = $('#bodyInput').value;
  state.environment = $('.env-btn.active')?.dataset.env || 'dev';
  state.useProxy = $('#proxyToggle').checked;
  return state;
}

function applyApiKeyIfNeeded(request) {
  const auth = request.auth || {};
  if (auth.type !== 'apikey' || !auth.apiKeyName || !auth.apiKeyValue) return request;
  if (auth.apiKeyLocation === 'query') {
    request.params = [...(request.params || []), { enabled: true, key: auth.apiKeyName, value: auth.apiKeyValue, description: 'API key' }];
  } else {
    request.headers = [...(request.headers || []), { enabled: true, key: auth.apiKeyName, value: auth.apiKeyValue, description: 'API key' }];
  }
  return request;
}

function requestForUse() {
  return applyApiKeyIfNeeded(structuredClone(getStateFromForm()));
}

function updateGenerated() {
  const request = requestForUse();
  let code;
  if (activeCodeTab === 'fetch') code = buildFetchCode(request);
  else if (activeCodeTab === 'python') code = buildPythonRequestsCode(request);
  else if (activeCodeTab === 'axios') code = buildAxiosCode(request);
  else code = buildCurl(request, { mask: true });
  $('#generatedCode').textContent = code;
  $('#fullUrlPreview').textContent = buildUrlWithParams(request.url, request.params);
  saveState(LS_KEY, state);
}

function fillForm() {
  $('#method').value = state.method || 'GET';
  $('#url').value = state.url || '';
  $('#authType').value = state.auth?.type || 'none';
  $('#authToken').value = state.auth?.token || '';
  $('#basicUser').value = state.auth?.username || '';
  $('#basicPass').value = state.auth?.password || '';
  $('#apiKeyName').value = state.auth?.apiKeyName || '';
  $('#apiKeyValue').value = state.auth?.apiKeyValue || '';
  $('#apiKeyLocation').value = state.auth?.apiKeyLocation || 'header';
  $('#bodyInput').value = state.body || '';
  $('#proxyToggle').checked = !!state.useProxy;
  $$('.env-btn').forEach(b => b.classList.toggle('active', b.dataset.env === (state.environment || 'dev')));
  $$('.body-mode').forEach(b => b.classList.toggle('active', b.dataset.mode === (state.bodyMode || 'json')));
  renderRows();
  updateAuthVisibility();
  updateGenerated();
}

function updateAuthVisibility() {
  const type = $('#authType').value;
  $('#bearerAuth').classList.toggle('hidden', type !== 'bearer');
  $('#basicAuth').classList.toggle('hidden', type !== 'basic');
  $('#apiKeyAuth').classList.toggle('hidden', type !== 'apikey');
}

function prettyBody(text) {
  const parsed = safeJsonParse(text, null);
  return parsed ? JSON.stringify(parsed, null, 2) : text;
}

function renderResponse(result) {
  lastResult = result;
  const ok = result.ok && result.status < 400;
  $('#statusBadge').className = ok ? 'status-ok' : 'status-error';
  $('#statusBadge').textContent = result.status ? `${result.status} ${result.statusText || ''}` : 'Failed';
  $('#resTime').textContent = `${result.timeMs || 0} ms`;
  $('#resSize').textContent = formatBytes(result.sizeBytes || 0);
  $('#lastRun').textContent = formatDateTime(new Date());
  $('#metricStatus').textContent = result.status ? `${result.status} ${result.statusText || ''}` : 'Failed';
  $('#metricLatency').textContent = `${result.timeMs || 0} ms`;
  $('#metricSize').textContent = formatBytes(result.sizeBytes || 0);
  $('#metricLastRun').textContent = formatDateTime(new Date());
  $('#requestId').textContent = `req_${Math.random().toString(16).slice(2, 18)}`;

  if (!result.ok) {
    $('#responseBody').textContent = `Request failed.\n\n${result.error || result.message || 'Unknown error'}\n\n${result.corsHint ? 'Possible reason: browser CORS restrictions. Generated cURL will still work in your terminal. For premium mode, configure a Cloudflare Worker proxy.' : ''}`;
  } else {
    $('#responseBody').textContent = prettyBody(result.body || '');
  }
  $('#responseHeaders').textContent = JSON.stringify(result.headers || {}, null, 2);
  pushRecent('rmv-http-curl-builder-recent', { ...state, lastRun: new Date().toISOString() }, 20);
}

async function send() {
  const request = requestForUse();
  if (!request.url) return showToast('Enter a URL');
  $('#responseBody').textContent = 'Sending request…';
  const result = await runRequest(request, { useProxy: state.useProxy });
  renderResponse(result);
  showToast(result.ok ? 'Request complete' : 'Request failed');
}

function bindEvents() {
  document.addEventListener('input', (e) => {
    if (e.target.matches('input, textarea, select')) {
      getStateFromForm();
      updateAuthVisibility();
      updateGenerated();
    }
  });

  document.addEventListener('click', (e) => {
    const pRemove = e.target.closest('[data-remove-param]');
    const hRemove = e.target.closest('[data-remove-header]');
    if (pRemove) { state.params.splice(Number(pRemove.dataset.removeParam), 1); renderRows(); updateGenerated(); }
    if (hRemove) { state.headers.splice(Number(hRemove.dataset.removeHeader), 1); renderRows(); updateGenerated(); }
  });

  $('#addParam').addEventListener('click', () => { state.params.push({ enabled:true, key:'', value:'', description:'' }); renderRows(); updateGenerated(); });
  $('#addHeader').addEventListener('click', () => { state.headers.push({ enabled:true, key:'', value:'', description:'' }); renderRows(); updateGenerated(); });
  $('#sendBtn').addEventListener('click', send);
  $('#copyCode').addEventListener('click', () => copyText($('#generatedCode').textContent, 'Code copied'));
  $('#downloadCode').addEventListener('click', () => downloadFile(`${activeCodeTab}-${timestampSlug()}.txt`, $('#generatedCode').textContent));
  $('#copyResponse').addEventListener('click', () => copyText($('#responseBody').textContent, 'Response copied'));
  $('#downloadResponse').addEventListener('click', () => lastResult && downloadJson(`response-${timestampSlug()}.json`, lastResult));
  $('#saveRequest').addEventListener('click', () => { saveState(LS_KEY, getStateFromForm()); showToast('Request saved locally'); });
  $('#resetBtn').addEventListener('click', () => { state = structuredClone(defaultState); fillForm(); showToast('Reset to example'); });
  $$('.env-btn').forEach(btn => btn.addEventListener('click', () => { $$('.env-btn').forEach(b => b.classList.remove('active')); btn.classList.add('active'); updateGenerated(); }));
  $$('.body-mode').forEach(btn => btn.addEventListener('click', () => { $$('.body-mode').forEach(b => b.classList.remove('active')); btn.classList.add('active'); updateGenerated(); }));
  $$('.code-tab').forEach(btn => btn.addEventListener('click', () => { activeCodeTab = btn.dataset.code; $$('.code-tab').forEach(b => b.classList.remove('active')); btn.classList.add('active'); updateGenerated(); }));
  $$('.response-tab').forEach(btn => btn.addEventListener('click', () => {
    $$('.response-tab').forEach(b => b.classList.remove('active')); btn.classList.add('active');
    $('#responseBody').classList.toggle('hidden', btn.dataset.response !== 'body');
    $('#responseHeaders').classList.toggle('hidden', btn.dataset.response !== 'headers');
  }));
}

function init() {
  fillForm();
  bindEvents();
  renderResponse({ ok: true, status: 200, statusText: 'Example', body: '{\n  "message": "Click Send to run the request"\n}', headers: {}, timeMs: 0, sizeBytes: 0 });
}

document.addEventListener('DOMContentLoaded', init);
