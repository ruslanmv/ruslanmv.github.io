export function $(selector, root = document) { return root.querySelector(selector); }
export function $$(selector, root = document) { return Array.from(root.querySelectorAll(selector)); }

export function showToast(message, ms = 1800) {
  let el = document.querySelector('[data-toast]');
  if (!el) {
    el = document.createElement('div');
    el.className = 'toast';
    el.setAttribute('data-toast', '');
    document.body.appendChild(el);
  }
  el.textContent = message;
  el.classList.add('show');
  clearTimeout(showToast._timer);
  showToast._timer = setTimeout(() => el.classList.remove('show'), ms);
}

export async function copyText(text, label = 'Copied') {
  if (!text) { showToast('Nothing to copy'); return; }
  try {
    await navigator.clipboard.writeText(text);
    showToast(label);
  } catch {
    const ta = document.createElement('textarea');
    ta.value = text;
    ta.style.cssText = 'position:fixed;opacity:0;left:-9999px;top:0';
    document.body.appendChild(ta);
    ta.select();
    document.execCommand('copy');
    ta.remove();
    showToast(label);
  }
}

export function safeJsonParse(value, fallback = null) {
  try { return JSON.parse(value); } catch { return fallback; }
}

export function formatBytes(bytes = 0) {
  if (!Number.isFinite(bytes) || bytes <= 0) return '0 B';
  const units = ['B','KB','MB','GB'];
  const i = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  const val = bytes / Math.pow(1024, i);
  return `${val >= 10 || i === 0 ? val.toFixed(0) : val.toFixed(2)} ${units[i]}`;
}

export function formatDateTime(date = new Date()) {
  try { return new Intl.DateTimeFormat(undefined, { dateStyle: 'medium', timeStyle: 'medium' }).format(new Date(date)); }
  catch { return String(date); }
}

export function formatTime(date = new Date()) {
  try { return new Intl.DateTimeFormat(undefined, { timeStyle: 'medium' }).format(new Date(date)); }
  catch { return String(date); }
}

export function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>'"]/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[ch]));
}

export function setText(id, value) {
  const el = typeof id === 'string' ? document.getElementById(id) : id;
  if (el) el.textContent = value ?? '';
}
