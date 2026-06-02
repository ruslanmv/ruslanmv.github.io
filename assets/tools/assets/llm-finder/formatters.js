// Formatting helpers for the LLM Compatibility Finder.

export function formatCount(n) {
  n = Number(n) || 0;
  if (n >= 1e9) return (n / 1e9).toFixed(1).replace(/\.0$/, '') + 'B';
  if (n >= 1e6) return (n / 1e6).toFixed(1).replace(/\.0$/, '') + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(1).replace(/\.0$/, '') + 'K';
  return String(n);
}

export function formatMemory(gb) {
  if (gb == null || !isFinite(gb)) return '—';
  return '~' + (Math.round(Number(gb) * 10) / 10) + ' GB';
}

export function relativeDate(iso) {
  if (!iso) return '—';
  const t = Date.parse(iso);
  if (!isFinite(t)) return '—';
  const days = Math.round((Date.now() - t) / 86400000);
  if (days <= 0) return 'today';
  if (days === 1) return 'yesterday';
  if (days < 30) return days + ' days ago';
  if (days < 365) { const m = Math.round(days / 30); return m + ' month' + (m > 1 ? 's' : '') + ' ago'; }
  const y = Math.round(days / 365);
  return y + ' year' + (y > 1 ? 's' : '') + ' ago';
}

export function scoreLabel(s) {
  s = Number(s) || 0;
  if (s >= 90) return 'Excellent';
  if (s >= 80) return 'Very Good';
  if (s >= 70) return 'Good';
  if (s >= 55) return 'Fair';
  if (s >= 35) return 'Limited';
  return 'Low';
}

export function scoreLevel(s) {
  s = Number(s) || 0;
  if (s >= 80) return 'excellent';
  if (s >= 70) return 'good';
  if (s >= 50) return 'fair';
  if (s >= 30) return 'limited';
  return 'low';
}

const APP_LABELS = {
  'ollama': 'Ollama',
  'llama.cpp': 'llama.cpp',
  'lm-studio': 'LM Studio',
  'jan': 'Jan',
  'vllm': 'vLLM',
  'sglang': 'SGLang',
  'mlx-lm': 'MLX',
  'colab-free': 'Colab Free',
  'hf-zerogpu': 'ZeroGPU',
  'colab': 'Colab',
  'local': 'Local'
};

export function appLabel(a) {
  return APP_LABELS[a] || a;
}
