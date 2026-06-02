// Renders the full Model Compatibility Report (page 2) — matches the mockup.
import { escapeHtml } from '../shared/tool-shell.js';
import { formatCount, formatMemory, relativeDate, scoreLabel, scoreLevel } from './formatters.js';
import { avatarFor } from './render-model-row.js';

const LEVEL_TEXT = { excellent: 'Excellent', good: 'Good', limited: 'Limited', low: 'Low' };

export function appCompat(model) {
  const f = model.formats || [];
  const gguf = f.includes('gguf'), st = f.includes('safetensors');
  const p = Number(model.estimatedParamsB) || 0;
  return [
    ['Ollama',            gguf ? 'excellent' : 'low'],
    ['llama.cpp',         gguf ? 'excellent' : 'low'],
    ['LM Studio',         gguf ? 'good' : 'low'],
    ['Jan',               gguf ? 'good' : 'low'],
    ['Colab Free',        st ? 'good' : (gguf ? 'limited' : 'low')],
    ['ZeroGPU',           p && p <= 3 ? 'good' : (p && p <= 7 ? 'limited' : 'low')],
    ['vLLM (NVIDIA GPU)', st ? 'excellent' : (gguf ? 'good' : 'low')]
  ];
}

function ring(score) {
  const r = 34, C = 2 * Math.PI * r;
  const dash = Math.max(0, Math.min(100, score)) / 100 * C;
  const lvl = scoreLevel(score);
  return `<svg class="ring ${lvl}" viewBox="0 0 80 80" width="84" height="84" aria-hidden="true">
    <circle class="ring-bg" cx="40" cy="40" r="${r}" fill="none" stroke-width="7"></circle>
    <circle class="ring-fg" cx="40" cy="40" r="${r}" fill="none" stroke-width="7" stroke-linecap="round"
      stroke-dasharray="${dash.toFixed(1)} ${(C - dash).toFixed(1)}" transform="rotate(-90 40 40)"></circle>
    <text x="40" y="38" class="ring-num">${score}%</text>
    <text x="40" y="53" class="ring-lab">${scoreLabel(score)}</text>
  </svg>`;
}

function cmdBlock(label, key, command) {
  return `<div class="cmd">
    <div class="cmd-head"><span class="cmd-label">${escapeHtml(label)}</span>
      <button class="btn small" data-copy-command="${escapeHtml(key)}">Copy</button></div>
    <pre class="cmd-body mono">${escapeHtml(command)}</pre>
  </div>`;
}

export function renderReport(model, query) {
  const score = Number(model.compatibilityScore) || 0;
  const q = query || {};
  const cmds = model.commands || {};
  const apps = Array.isArray(model.compatibleApps) ? model.compatibleApps : [];
  const bestTarget = (model.recommendedTarget || apps[0] || 'ollama');
  const bestLabel = bestTarget === 'llama.cpp' ? 'llama.cpp' : bestTarget.replace(/(^|[-\s])\w/g, (m) => m.toUpperCase()).replace('Lm-studio', 'LM Studio');
  const author = String(model.repoId || '').split('/')[0];
  const hfUrl = (model.links && model.links.huggingFace) || ('https://huggingface.co/' + model.repoId);

  // tags
  const tags = [];
  (model.formats || []).forEach((f) => tags.push(f.toUpperCase()));
  if (/instruct/i.test(model.repoId || model.name || '')) tags.push('Instruct');
  if (q.useCase) tags.push(q.useCase.charAt(0).toUpperCase() + q.useCase.slice(1));

  // command blocks (only those present)
  const blocks = [
    ['Ollama (Recommended)', 'ollama', cmds.ollama],
    ['llama.cpp (Server)', 'llamaCppServer', cmds.llamaCppServer],
    ['llama.cpp (CLI)', 'llamaCppCli', cmds.llamaCppCli],
    ['vLLM (GPU Server)', 'vllm', cmds.vllm]
  ].filter((b) => b[2]).map((b) => cmdBlock(b[0], b[1], b[2])).join('');

  // compatibility overview
  const compatRows = appCompat(model).map(([name, lvl]) =>
    `<div class="ov-row"><span>${escapeHtml(name)}</span><span class="lvl lvl-${lvl}">${LEVEL_TEXT[lvl]}</span></div>`
  ).join('');

  // why / warnings
  const why = (Array.isArray(model.why) && model.why.length ? model.why : deriveWhy(model, q));
  const warns = (Array.isArray(model.warnings) ? model.warnings.slice() : []);
  if (!warns.some((w) => /estimat/i.test(w))) warns.unshift('Compatibility is estimated, not guaranteed.');

  // hardware fit
  const avail = (q.vramGb && q.vramGb > 0) ? q.vramGb : (q.ramGb || 16);
  const est = Number(model.estimatedMemoryGb) || 0;
  const headroom = Math.max(0, avail - est);
  const pctUsed = avail ? Math.min(100, Math.round(est / avail * 100)) : 0;
  const headPct = avail ? Math.round(headroom / avail * 100) : 0;
  const fitTitle = q.profileLabel || 'Selected hardware';

  return `
  <div class="rep-top">
    <div class="rep-id">
      ${avatarFor(model.repoId)}
      <div>
        <h1 class="rep-name">${escapeHtml(model.name || model.repoId)}</h1>
        <div class="rep-sub">${escapeHtml(author)} · ${model.estimatedParamsB ? model.estimatedParamsB + 'B parameters · ' : ''}${escapeHtml(model.task || 'Text Generation')}</div>
        <div class="rep-tags">${tags.map((t) => `<span class="tag">${escapeHtml(t)}</span>`).join('')}</div>
      </div>
    </div>
    <div class="rep-actions">
      <a class="btn primary" href="${escapeHtml(hfUrl)}" target="_blank" rel="noopener">Open on Hugging Face ↗</a>
      <button class="btn" id="downloadReport">Download report ⇩</button>
      <button class="btn" id="copySummary">Copy summary ⧉</button>
    </div>
  </div>

  <div class="grid-4 rep-kpis">
    <div class="kpi rep-kpi"><div class="ring-wrap">${ring(score)}</div>
      <div><div class="kpi__label">Compatibility Score</div><div class="kpi__note">Highly compatible with your selected setup</div></div></div>
    <div class="kpi rep-kpi"><div><div class="kpi__label">Best Target</div><div class="kpi__value">${escapeHtml(bestLabel)}</div>
      <div class="kpi__note">${apps.length > 1 ? 'Also great with ' + apps.slice(0, 3).map((a) => a === 'lm-studio' ? 'LM Studio' : a).filter((a) => a.toLowerCase() !== bestTarget).slice(0, 2).join(', ') : 'Local runtime'}</div></div></div>
    <div class="kpi rep-kpi"><div><div class="kpi__label">Recommended Quant</div><div class="kpi__value">${escapeHtml(model.recommendedQuant || '—')}</div>
      <div class="kpi__note">Best balance of quality and performance</div></div></div>
    <div class="kpi rep-kpi"><div><div class="kpi__label">Est. Memory Usage</div><div class="kpi__value">${formatMemory(model.estimatedMemoryGb)}</div>
      <div class="kpi__note">Against your ${avail} GB profile</div></div></div>
  </div>

  <div class="grid-2 rep-mid">
    <div class="card">
      <div class="card__header"><h2 class="card__title">⌘ Runtime Commands</h2></div>
      <div class="card__body">
        <p class="muted-line">Copy and run in your preferred environment.</p>
        ${blocks || '<p class="muted-line">No ready-made commands for this model.</p>'}
      </div>
    </div>
    <div class="card">
      <div class="card__header"><h2 class="card__title">Compatibility Overview</h2></div>
      <div class="card__body ov-list">${compatRows}</div>
    </div>
  </div>

  <div class="grid-3 rep-low">
    <div class="card pad">
      <h3 class="mini-title">Why this model is compatible</h3>
      <ul class="why-list">${why.map((w) => `<li>${escapeHtml(w)}</li>`).join('')}</ul>
    </div>
    <div class="card pad">
      <h3 class="mini-title">Hardware Fit (${escapeHtml(fitTitle)})</h3>
      <div class="fit-bar"><span style="width:${pctUsed}%"></span></div>
      <div class="fit-rows">
        <div><span>Estimated usage</span><b>${formatMemory(est).replace('~', '')}</b></div>
        <div><span>Available</span><b>${avail} GB</b></div>
        <div><span>Headroom</span><b>${(Math.round(headroom * 10) / 10)} GB (${headPct}%)</b></div>
      </div>
    </div>
    <div class="card pad">
      <h3 class="mini-title">⚠ Warnings</h3>
      <ul class="warn-list">${warns.map((w) => `<li>${escapeHtml(w)}</li>`).join('')}</ul>
    </div>
  </div>

  <div class="card pad rep-details">
    <h3 class="mini-title">Model Details</h3>
    <div class="detail-grid">
      ${detail('Repository ID', model.repoId)}
      ${detail('License', model.license || '—')}
      ${detail('Downloads', formatCount(model.downloads))}
      ${detail('Likes', formatCount(model.likes))}
      ${detail('Last updated', relativeDate(model.lastModified))}
      ${detail('Model type', model.task || '—')}
      ${detail('Formats', (model.formats || []).map((f) => f.toUpperCase()).join(', ') || '—')}
      ${detail('Quantizations', (model.quants || []).join(', ') || '—')}
    </div>
  </div>`;
}

function detail(label, value) {
  return `<div class="detail"><div class="detail-k">${escapeHtml(label)}</div><div class="detail-v">${escapeHtml(String(value))}</div></div>`;
}

function deriveWhy(model, q) {
  const out = [];
  const f = model.formats || [];
  if (f.includes('gguf')) out.push('GGUF files detected in repository.');
  if (f.includes('safetensors')) out.push('safetensors files available for GPU serving.');
  if (model.recommendedQuant) out.push(`${model.recommendedQuant} quantization available.`);
  if (model.estimatedMemoryGb) out.push(`Estimated memory (${formatMemory(model.estimatedMemoryGb)}) fits your hardware profile.`);
  if (q.useCase) out.push(`Reasonable match for the ${q.useCase} use case.`);
  return out.length ? out : ['Model metadata indicates general compatibility.'];
}

export function buildSummaryText(model, query) {
  const q = query || {};
  const cmds = model.commands || {};
  return [
    `LLM Compatibility Report — ${model.name || model.repoId}`,
    `Repository: ${model.repoId}`,
    `Compatibility score: ${model.compatibilityScore}% (${scoreLabel(model.compatibilityScore)})`,
    `Best target: ${model.recommendedTarget || (model.compatibleApps || [])[0] || '—'}`,
    `Recommended quant: ${model.recommendedQuant || '—'}`,
    `Estimated memory: ${formatMemory(model.estimatedMemoryGb)}`,
    `License: ${model.license || '—'}`,
    q.profileLabel ? `Hardware profile: ${q.profileLabel}` : '',
    '',
    cmds.ollama ? `Ollama:    ${cmds.ollama}` : '',
    cmds.llamaCppServer ? `llama.cpp: ${cmds.llamaCppServer}` : '',
    cmds.vllm ? `vLLM:      ${cmds.vllm}` : '',
    '',
    'Compatibility is estimated and may vary with your real setup.'
  ].filter((l) => l !== '').join('\n');
}
