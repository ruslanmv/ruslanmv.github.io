// Renders one results-table row for the finder page (matches the mockup layout).
import { escapeHtml } from '../shared/tool-shell.js';
import { formatCount, formatMemory, relativeDate, scoreLabel, scoreLevel, appLabel } from './formatters.js';

const AV_COLORS = ['#0f7a43', '#175cd3', '#8a5cf6', '#b9770e', '#0e7490', '#be185d', '#4f46e5'];

export function avatarFor(repoId) {
  const author = String(repoId || '?').split('/')[0];
  const letter = (author[0] || '?').toUpperCase();
  let h = 0;
  for (let i = 0; i < author.length; i++) h = (h * 31 + author.charCodeAt(i)) >>> 0;
  const color = AV_COLORS[h % AV_COLORS.length];
  return `<span class="m-avatar" style="background:${color}1a;color:${color}">${escapeHtml(letter)}</span>`;
}

export function renderModelRow(model, index) {
  const score = Number(model.compatibilityScore) || 0;
  const apps = Array.isArray(model.compatibleApps) ? model.compatibleApps : [];
  const best = appLabel(model.recommendedTarget || apps[0] || '—');
  const second = apps.map(appLabel).filter((a) => a !== best)[0] || '';
  const author = String(model.repoId || '').split('/')[0];
  const sizeBit = model.estimatedParamsB ? `${model.estimatedParamsB}B` : '';
  const subline = [author, sizeBit, model.task || 'Text Generation'].filter(Boolean).join(' · ');
  const fmt = (model.formats || [])[0];
  const hfUrl = (model.links && model.links.huggingFace) || ('https://huggingface.co/' + model.repoId);

  return `<tr>
    <td class="m-cell">
      ${avatarFor(model.repoId)}
      <div class="m-id">
        <div class="m-name">${escapeHtml(model.name || model.repoId)}</div>
        <div class="m-sub">${escapeHtml(subline)}</div>
        ${fmt ? `<span class="fmt-tag">${escapeHtml(fmt.toUpperCase())}</span>` : ''}
      </div>
    </td>
    <td data-label="Compatibility"><div class="cmp"><span class="cmp-pct ${scoreLevel(score)}">${score}%</span><span class="cmp-lab">${scoreLabel(score)}</span></div></td>
    <td data-label="Best runtime"><div class="rt">${escapeHtml(best)}${second ? `<span class="rt-2">${escapeHtml(second)}</span>` : ''}</div></td>
    <td data-label="Quant">${model.recommendedQuant ? `<span class="quant">${escapeHtml(model.recommendedQuant)}</span>` : '—'}</td>
    <td data-label="Est. memory" class="nowrap">${formatMemory(model.estimatedMemoryGb)}</td>
    <td data-label="License" class="nowrap">${escapeHtml(model.license || '—')}</td>
    <td data-label="Updated" class="upd"><div>${escapeHtml(relativeDate(model.lastModified))}</div><div class="dl">↓ ${formatCount(model.downloads)}</div></td>
    <td class="m-actions">
      <button class="btn small primary" data-action="view-report" data-index="${index}">View Report</button>
      <a class="btn small icon" href="${escapeHtml(hfUrl)}" target="_blank" rel="noopener" title="Open on Hugging Face" aria-label="Open on Hugging Face">↗</a>
    </td>
  </tr>`;
}
