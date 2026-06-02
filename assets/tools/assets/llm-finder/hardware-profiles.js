// Hardware presets + filter option lists for the LLM Compatibility Finder.

export const HARDWARE_PROFILES = [
  { id: 'cpu-8gb',   label: 'CPU only · 8 GB RAM',   ramGb: 8,  vramGb: 0,  target: 'local' },
  { id: 'cpu-16gb',  label: 'CPU only · 16 GB RAM',  ramGb: 16, vramGb: 0,  target: 'local' },
  { id: 'mac-16gb',  label: 'Mac · 16 GB Unified',   ramGb: 16, vramGb: 0,  unifiedMemoryGb: 16, target: 'ollama' },
  { id: 'rtx-3060',  label: 'RTX 3060 · 12 GB VRAM', ramGb: 32, vramGb: 12, gpu: 'nvidia', target: 'vllm' },
  { id: 'rtx-4090',  label: 'RTX 4090 · 24 GB VRAM', ramGb: 64, vramGb: 24, gpu: 'nvidia', target: 'vllm' },
  { id: 'colab-free',label: 'Google Colab Free',     ramGb: 12, vramGb: 16, gpu: 'nvidia', target: 'colab-free', notes: 'GPU availability is not guaranteed.' },
  { id: 'hf-zerogpu',label: 'Hugging Face ZeroGPU',  ramGb: 16, vramGb: 0,  target: 'hf-zerogpu', notes: 'Best for short Spaces demos, not persistent servers.' }
];

export const RUNTIMES = [
  { id: 'ollama',     label: 'Ollama' },
  { id: 'llama.cpp',  label: 'llama.cpp' },
  { id: 'lm-studio',  label: 'LM Studio' },
  { id: 'jan',        label: 'Jan' },
  { id: 'vllm',       label: 'vLLM' },
  { id: 'colab-free', label: 'Colab Free' },
  { id: 'hf-zerogpu', label: 'ZeroGPU' }
];

export const USE_CASES = [
  { id: 'chat',       label: 'Chat' },
  { id: 'coding',     label: 'Coding' },
  { id: 'rag',        label: 'RAG' },
  { id: 'agent',      label: 'Agent' },
  { id: 'json',       label: 'JSON' },
  { id: 'embeddings', label: 'Embeddings' }
];

// Optional client-side filter on estimated parameter count (billions).
export const MODEL_SIZES = [
  { id: 'all', label: 'All sizes',  max: Infinity },
  { id: '3',   label: '≤ 3B (tiny)', max: 3 },
  { id: '8',   label: '≤ 8B (small)', max: 8 },
  { id: '14',  label: '≤ 14B (medium)', max: 14 },
  { id: '34',  label: '≤ 34B (large)', max: 34 },
  { id: '70',  label: '≤ 70B (x-large)', max: 70 }
];

export const SORTS = [
  { id: 'score',     label: 'Best match' },
  { id: 'downloads', label: 'Most downloads' },
  { id: 'updated',   label: 'Recently updated' },
  { id: 'memory',    label: 'Smallest memory' }
];

export function profileById(id) {
  return HARDWARE_PROFILES.find((p) => p.id === id) || HARDWARE_PROFILES[1];
}
