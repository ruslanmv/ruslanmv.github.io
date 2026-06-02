// Page 1 controller — search + ranked results table.
import { $, showToast, escapeHtml } from '../shared/tool-shell.js';
import { HARDWARE_PROFILES, RUNTIMES, USE_CASES, MODEL_SIZES, SORTS, profileById } from './hardware-profiles.js';
import { searchModels } from './api.js';
import { renderModelRow } from './render-model-row.js';
import { scoreLabel } from './formatters.js';
import { saveSelectedModel, saveQuery, saveFilters, loadFilters } from './storage.js';

const els = {
  search: $('#modelSearch'),
  runtime: $('#runtimeTarget'),
  profile: $('#hardwareProfile'),
  useCase: $('#useCase'),
  size: $('#modelSize'),
  sort: $('#sortBy'),
  pageSize: $('#pageSize'),
  searchBtn: $('#searchBtn'),
  compareBtn: $('#compareBtn'),
  body: $('#resultsBody'),
  loading: $('#loadingState'),
  empty: $('#emptyState'),
  banner: $('#sourceBanner'),
  showing: $('#showingCount'),
  pager: $('#pager'),
  sumFound: $('#sumFound'),
  sumBest: $('#sumBest'),
  sumBestPill: $('#sumBestPill'),
  sumProfile: $('#sumProfile'),
  sumProfileNote: $('#sumProfileNote')
};

let results = [];   // raw from API
let view = [];      // filtered + sorted
let page = 1;

function fill(sel, items, selected) {
  if (!sel) return;
  sel.innerHTML = items.map((i) => `<option value="${i.id}"${i.id === selected ? ' selected' : ''}>${escapeHtml(i.label)}</option>`).join('');
}

function initControls() {
  const f = loadFilters() || {};
  fill(els.runtime, RUNTIMES, f.runtime || 'ollama');
  fill(els.profile, HARDWARE_PROFILES, f.profile || 'mac-16gb');
  fill(els.useCase, USE_CASES, f.useCase || 'coding');
  fill(els.size, MODEL_SIZES, f.size || 'all');
  fill(els.sort, SORTS, f.sort || 'score');
  fill(els.pageSize, [{ id: '5', label: '5' }, { id: '10', label: '10' }, { id: '25', label: '25' }], String(f.pageSize || 5));
  if (els.search) els.search.value = f.q != null ? f.q : 'qwen coder';
}

function currentFilters() {
  const profile = profileById(els.profile.value);
  return {
    q: (els.search.value || '').trim(),
    runtime: els.runtime.value,
    profile: profile.id,
    profileLabel: profile.label,
    ramGb: profile.ramGb,
    vramGb: profile.vramGb,
    useCase: els.useCase.value,
    size: els.size.value,
    sort: els.sort.value,
    pageSize: Number(els.pageSize.value) || 5
  };
}

async function doSearch() {
  const f = currentFilters();
  saveFilters(f);
  els.loading.style.display = 'flex';
  els.empty.style.display = 'none';
  els.body.innerHTML = '';
  els.banner.style.display = 'none';
  els.searchBtn.disabled = true;

  const { source, data, error } = await searchModels({
    q: f.q, target: f.runtime, ramGb: f.ramGb, vramGb: f.vramGb, useCase: f.useCase, limit: 30
  });

  els.loading.style.display = 'none';
  els.searchBtn.disabled = false;

  if (source === 'none' || !data) {
    results = [];
    view = [];
    renderTable(f);
    els.empty.style.display = 'block';
    els.empty.textContent = 'Could not reach the LLM Compatibility Worker, and no sample data is available. Please try again.';
    return;
  }

  if (source === 'sample') {
    els.banner.style.display = 'flex';
    els.banner.innerHTML = `<span>⚠ Could not reach the Worker (${escapeHtml(error || 'network error')}). Showing bundled sample results.</span>`;
  }

  results = Array.isArray(data.results) ? data.results : [];
  applyAndRender(f);
}

function applyAndRender(f) {
  const sizeMax = (MODEL_SIZES.find((s) => s.id === f.size) || {}).max || Infinity;
  view = results.filter((m) => !m.estimatedParamsB || m.estimatedParamsB <= sizeMax);

  const by = f.sort;
  view.sort((a, b) => {
    if (by === 'downloads') return (b.downloads || 0) - (a.downloads || 0);
    if (by === 'updated') return Date.parse(b.lastModified || 0) - Date.parse(a.lastModified || 0);
    if (by === 'memory') return (a.estimatedMemoryGb || 1e9) - (b.estimatedMemoryGb || 1e9);
    return (b.compatibilityScore || 0) - (a.compatibilityScore || 0);
  });

  page = 1;
  renderTable(f);
  renderSummary(f);
}

function renderTable(f) {
  const ps = f.pageSize || 5;
  const pages = Math.max(1, Math.ceil(view.length / ps));
  if (page > pages) page = pages;
  const start = (page - 1) * ps;
  const slice = view.slice(start, start + ps);

  els.body.innerHTML = slice.map((m) => renderModelRow(m, results.indexOf(m))).join('');
  els.empty.style.display = view.length ? 'none' : 'block';
  if (!view.length) els.empty.textContent = 'No models matched your filters. Try a broader search or a larger model-size limit.';

  els.showing.textContent = view.length
    ? `Showing ${view.length} result${view.length === 1 ? '' : 's'}`
    : '';

  // pager
  if (pages <= 1) { els.pager.innerHTML = ''; return; }
  const btns = [];
  btns.push(`<button class="pg" data-pg="${page - 1}"${page === 1 ? ' disabled' : ''}>‹</button>`);
  for (let i = 1; i <= pages; i++) {
    if (i === 1 || i === pages || Math.abs(i - page) <= 1) {
      btns.push(`<button class="pg${i === page ? ' active' : ''}" data-pg="${i}">${i}</button>`);
    } else if (btns[btns.length - 1] !== '<span class="pg-dots">…</span>') {
      btns.push('<span class="pg-dots">…</span>');
    }
  }
  btns.push(`<button class="pg" data-pg="${page + 1}"${page === pages ? ' disabled' : ''}>›</button>`);
  const from = start + 1, to = Math.min(start + ps, view.length);
  els.pager.innerHTML = `<span class="pg-info">${from}-${to} of ${view.length}</span><div class="pg-btns">${btns.join('')}</div>`;
}

function renderSummary(f) {
  els.sumFound.textContent = String(view.length);
  const best = view[0];
  if (best) {
    els.sumBest.textContent = (best.name || best.repoId).replace(/\s*GGUF$/i, '');
    els.sumBestPill.textContent = `${best.compatibilityScore}% compatible`;
    els.sumBestPill.style.display = '';
  } else {
    els.sumBest.textContent = '—';
    els.sumBestPill.style.display = 'none';
  }
  els.sumProfile.textContent = f.profileLabel;
  els.sumProfileNote.textContent = `${(USE_CASES.find((u) => u.id === f.useCase) || {}).label || 'Chat'} use case`;
}

function wire() {
  els.searchBtn.addEventListener('click', doSearch);
  els.search.addEventListener('keydown', (e) => { if (e.key === 'Enter') doSearch(); });
  [els.size, els.sort, els.pageSize].forEach((s) => s && s.addEventListener('change', () => { if (results.length) applyAndRender(currentFilters()); }));
  // re-run a fresh search when runtime/profile/usecase change (affects scoring)
  [els.runtime, els.profile, els.useCase].forEach((s) => s && s.addEventListener('change', doSearch));

  els.body.addEventListener('click', (e) => {
    const btn = e.target.closest('[data-action="view-report"]');
    if (!btn) return;
    const idx = Number(btn.dataset.index);
    const model = results[idx];
    if (!model) return;
    const f = currentFilters();
    saveSelectedModel(model);
    saveQuery(f);
    window.location.href = 'llm-compatibility-report.html';
  });

  els.pager.addEventListener('click', (e) => {
    const b = e.target.closest('[data-pg]');
    if (!b || b.disabled) return;
    page = Number(b.dataset.pg);
    renderTable(currentFilters());
    window.scrollTo({ top: els.showing.offsetTop - 90, behavior: 'smooth' });
  });

  if (els.compareBtn) els.compareBtn.addEventListener('click', () =>
    showToast('Open a model report, then use “Compare another model” (coming soon).'));
}

initControls();
wire();
doSearch();
