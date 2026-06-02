export function saveState(key, value) {
  try { localStorage.setItem(key, JSON.stringify(value)); return true; }
  catch { return false; }
}

export function loadState(key, fallback = null) {
  try {
    const raw = localStorage.getItem(key);
    return raw ? JSON.parse(raw) : fallback;
  } catch { return fallback; }
}

export function removeState(key) {
  try { localStorage.removeItem(key); } catch {}
}

export function pushRecent(key, item, max = 20) {
  const list = loadState(key, []);
  const next = [item, ...list].slice(0, max);
  saveState(key, next);
  return next;
}
