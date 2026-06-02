// Talks to the independent Cloudflare Worker, with a graceful sample fallback.

export const LLM_FINDER_WORKER_BASE =
  (typeof window !== 'undefined' && window.RMV_API_BASE) ||
  'https://llm-compatibility-finder.cloud-data.workers.dev';

export function buildSearchUrl({ q, target, ramGb, vramGb, useCase, limit = 30 }) {
  const base = LLM_FINDER_WORKER_BASE.replace(/\/$/, '');
  const u = new URL(base + '/api/search');
  u.searchParams.set('q', q || '');
  u.searchParams.set('target', target || 'ollama');
  u.searchParams.set('ram_gb', ramGb != null ? ramGb : 16);
  u.searchParams.set('vram_gb', vramGb != null ? vramGb : 0);
  u.searchParams.set('use_case', useCase || 'chat');
  u.searchParams.set('limit', limit);
  return u.toString();
}

// Returns { source: 'worker' | 'sample' | 'none', data, error? }
export async function searchModels(params) {
  const url = buildSearchUrl(params);
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 20000);
  try {
    const res = await fetch(url, { signal: controller.signal });
    clearTimeout(timer);
    if (!res.ok) throw new Error('Worker responded ' + res.status);
    const data = await res.json();
    if (!data || !Array.isArray(data.results)) throw new Error('Malformed Worker response');
    return { source: 'worker', data };
  } catch (err) {
    clearTimeout(timer);
    // Fallback: bundled sample dataset so the UI still works offline / if the Worker is down.
    try {
      const res = await fetch('data/llm-finder/sample-models.json', { cache: 'no-store' });
      if (res.ok) {
        const data = await res.json();
        return { source: 'sample', data, error: err.message };
      }
    } catch (e2) { /* ignore */ }
    return { source: 'none', data: null, error: err.message };
  }
}

export async function checkHealth() {
  try {
    const res = await fetch(LLM_FINDER_WORKER_BASE.replace(/\/$/, '') + '/api/health', { cache: 'no-store' });
    return res.ok;
  } catch (e) {
    return false;
  }
}
