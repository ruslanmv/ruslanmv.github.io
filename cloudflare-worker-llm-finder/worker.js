/**
 * Cloudflare Worker backend for the LLM Compatibility Finder.
 *
 * Frontend: https://ruslanmv.com/assets/tools/llm-compatibility-finder.html
 * Worker:   https://llm-compatibility-finder.cloud-data.workers.dev
 *
 * Endpoints:
 *   GET /api/health
 *   GET /api/search?q=&target=&use_case=&ram_gb=&vram_gb=&limit=
 *
 * Read-only: only calls the public Hugging Face Hub API. Results are cached.
 * Keep this Worker SEPARATE from network-api-tools.
 */
const ALLOWED_ORIGINS = [
  "http://localhost:8000",
  "http://127.0.0.1:8000",

  "http://localhost:4000",
  "http://127.0.0.1:4000",
  "http://localhost:4001",
  "http://127.0.0.1:4001",

  "https://ruslanmv.com",
  "https://www.ruslanmv.com",
  "https://ruslanmv.github.io"
];

const HF_API_BASE = "https://huggingface.co/api";
const DEFAULT_LIMIT = 12;
const MAX_LIMIT = 30;
const CACHE_TTL_SECONDS = 60 * 60;

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);

    if (request.method === "OPTIONS") {
      return corsResponse(request);
    }

    if (url.pathname === "/api/health") {
      return json(request, {
        ok: true,
        service: "llm-compatibility-finder",
        version: "1.0.0",
        detectedAt: new Date().toISOString()
      });
    }

    if (url.pathname === "/api/search") {
      return handleSearch(request, env, ctx);
    }

    return json(request, { error: "Not found" }, 404);
  }
};

async function handleSearch(request, env, ctx) {
  if (request.method !== "GET") {
    return json(request, { error: "Use GET" }, 405);
  }

  const url = new URL(request.url);

  const q = cleanText(url.searchParams.get("q") || "qwen instruct");
  const target = cleanText(url.searchParams.get("target") || "ollama");
  const useCase = cleanText(url.searchParams.get("use_case") || "chat");
  const ramGb = clampNumber(url.searchParams.get("ram_gb"), 0, 512, 16);
  const vramGb = clampNumber(url.searchParams.get("vram_gb"), 0, 192, 0);
  const limit = Math.min(
    clampNumber(url.searchParams.get("limit"), 1, MAX_LIMIT, DEFAULT_LIMIT),
    MAX_LIMIT
  );

  const cacheKey = new Request(
    `${url.origin}${url.pathname}?q=${encodeURIComponent(q)}&target=${encodeURIComponent(target)}&use_case=${encodeURIComponent(useCase)}&ram_gb=${ramGb}&vram_gb=${vramGb}&limit=${limit}`,
    request
  );

  const cache = caches.default;
  const cached = await cache.match(cacheKey);

  if (cached) {
    return withCors(request, cached);
  }

  try {
    const models = await searchHuggingFaceModels(q, limit * 2, env);
    const enriched = await enrichModels(models.slice(0, limit * 2), env);

    const results = enriched
      .map((model) =>
        scoreModel(model, {
          target,
          useCase,
          ramGb,
          vramGb
        })
      )
      .filter((item) => item.compatibilityScore > 0)
      .sort((a, b) => b.compatibilityScore - a.compatibilityScore)
      .slice(0, limit);

    const responseBody = {
      query: {
        q,
        target,
        useCase,
        ramGb,
        vramGb,
        limit
      },
      results,
      generatedAt: new Date().toISOString(),
      note: "Compatibility is estimated from Hugging Face metadata, repo files, model names, formats, quantization labels, and selected hardware profile. It is not a guarantee."
    };

    const response = json(request, responseBody, 200, {
      "Cache-Control": `public, max-age=${CACHE_TTL_SECONDS}`
    });

    ctx.waitUntil(cache.put(cacheKey, response.clone()));

    return response;
  } catch (err) {
    return json(
      request,
      {
        error: "Search failed",
        message: err.message
      },
      502
    );
  }
}

async function searchHuggingFaceModels(query, limit, env) {
  const searchUrl = new URL(`${HF_API_BASE}/models`);

  searchUrl.searchParams.set("search", query);
  searchUrl.searchParams.set("full", "true");
  searchUrl.searchParams.set("sort", "downloads");
  searchUrl.searchParams.set("direction", "-1");
  searchUrl.searchParams.set("limit", String(Math.min(limit, 50)));

  const res = await hfFetch(searchUrl.toString(), env);

  if (!res.ok) {
    throw new Error(`Hugging Face model search failed: ${res.status}`);
  }

  const data = await res.json();

  return Array.isArray(data) ? data : [];
}

async function enrichModels(models, env) {
  const enriched = [];

  for (const model of models) {
    const repoId = model.id || model.modelId;

    if (!repoId || !repoId.includes("/")) continue;

    let siblings = Array.isArray(model.siblings) ? model.siblings : [];

    if (!siblings.length) {
      try {
        const info = await getModelInfo(repoId, env);
        siblings = Array.isArray(info.siblings) ? info.siblings : [];
      } catch {
        siblings = [];
      }
    }

    const files = siblings
      .map((s) => s.rfilename || s.filename || s.path || "")
      .filter(Boolean);

    const formats = detectFormats(files, model);
    const quants = detectQuants(files, model);
    const paramsB = estimateParamsB(repoId, model);
    const task = model.pipeline_tag || model.pipelineTag || guessTask(model);

    enriched.push({
      repoId,
      name: readableModelName(repoId),
      author: repoId.split("/")[0],
      task,
      license: detectLicense(model),
      downloads: numberOrZero(model.downloads),
      likes: numberOrZero(model.likes),
      lastModified: model.lastModified || model.last_modified || null,
      tags: Array.isArray(model.tags) ? model.tags : [],
      files: files.slice(0, 80),
      formats,
      quants,
      paramsB,
      raw: model
    });
  }

  return enriched;
}

async function getModelInfo(repoId, env) {
  const res = await hfFetch(`${HF_API_BASE}/models/${repoId}`, env);

  if (!res.ok) {
    throw new Error(`Hugging Face model info failed: ${res.status}`);
  }

  return res.json();
}

async function hfFetch(url, env) {
  const headers = {
    "Accept": "application/json",
    "User-Agent": "ruslanmv-llm-compatibility-finder/1.0"
  };

  // Optional: add HF_TOKEN as a Worker secret later for higher rate limits/private access.
  if (env && env.HF_TOKEN) {
    headers.Authorization = `Bearer ${env.HF_TOKEN}`;
  }

  return fetch(url, { headers });
}

function scoreModel(model, prefs) {
  const compatibleApps = detectCompatibleApps(model);
  const recommendedQuant = recommendQuant(model.quants);
  const estimatedMemoryGb = estimateMemoryGb(model.paramsB, recommendedQuant, model);
  const commands = buildCommands(model.repoId, recommendedQuant);

  const hardwareFit = scoreHardwareFit(estimatedMemoryGb, prefs, model);
  const appFit = scoreAppFit(compatibleApps, prefs.target);
  const formatFit = scoreFormatFit(model.formats, prefs.target);
  const quantFit = recommendedQuant ? 100 : 35;
  const popularity = scorePopularity(model.downloads, model.likes);
  const licenseScore = scoreLicense(model.license);
  const freshnessScore = scoreFreshness(model.lastModified);
  const useCaseScore = scoreUseCase(model, prefs.useCase);

  let compatibilityScore = Math.round(
    0.30 * hardwareFit +
      0.20 * appFit +
      0.15 * formatFit +
      0.10 * quantFit +
      0.10 * popularity +
      0.10 * licenseScore +
      0.05 * freshnessScore
  );

  compatibilityScore = Math.round(
    0.85 * compatibilityScore + 0.15 * useCaseScore
  );

  const why = buildWhy(model, prefs, {
    compatibleApps,
    recommendedQuant,
    estimatedMemoryGb,
    hardwareFit
  });

  const warnings = buildWarnings(model, prefs, {
    estimatedMemoryGb,
    hardwareFit
  });

  return {
    repoId: model.repoId,
    name: model.name,
    task: model.task,
    license: model.license,
    downloads: model.downloads,
    likes: model.likes,
    lastModified: model.lastModified,
    compatibilityScore,
    recommendedTarget: prefs.target,
    compatibleApps,
    formats: model.formats,
    quants: model.quants,
    recommendedQuant,
    estimatedMemoryGb,
    estimatedParamsB: model.paramsB,
    commands,
    links: {
      huggingFace: `https://huggingface.co/${model.repoId}`,
      ollamaLocalApp: hfLocalAppUrl(model.repoId, "ollama"),
      llamaCppLocalApp: hfLocalAppUrl(model.repoId, "llamacpp"),
      lmStudioLocalApp: hfLocalAppUrl(model.repoId, "lmstudio")
    },
    why,
    warnings
  };
}

function detectFormats(files, model) {
  const text = `${files.join(" ")} ${(model.tags || []).join(" ")} ${model.repoId || ""}`.toLowerCase();
  const formats = new Set();

  if (text.includes(".gguf") || text.includes("gguf")) formats.add("gguf");
  if (text.includes(".safetensors") || text.includes("safetensors")) {
    formats.add("safetensors");
  }
  if (text.includes(".onnx") || text.includes("onnx")) formats.add("onnx");
  if (text.includes("mlx")) formats.add("mlx");

  return [...formats];
}

function detectQuants(files, model) {
  const source = `${files.join(" ")} ${(model.tags || []).join(" ")} ${model.id || ""}`;
  const patterns = [
    "Q2_K",
    "Q3_K_S",
    "Q3_K_M",
    "Q3_K_L",
    "Q4_0",
    "Q4_1",
    "Q4_K_S",
    "Q4_K_M",
    "Q5_0",
    "Q5_1",
    "Q5_K_S",
    "Q5_K_M",
    "Q6_K",
    "Q8_0",
    "IQ2_XS",
    "IQ3_XS",
    "IQ4_XS",
    "FP16",
    "BF16"
  ];

  const found = new Set();

  for (const p of patterns) {
    const re = new RegExp(`(^|[^A-Za-z0-9])${escapeRegExp(p)}([^A-Za-z0-9]|$)`, "i");
    if (re.test(source)) found.add(p.toUpperCase());
  }

  return [...found].sort(sortQuants);
}

function recommendQuant(quants) {
  const preference = [
    "Q4_K_M",
    "Q5_K_M",
    "Q4_K_S",
    "Q5_K_S",
    "Q6_K",
    "Q8_0",
    "Q4_0",
    "IQ4_XS",
    "Q3_K_M",
    "Q3_K_S",
    "IQ3_XS",
    "Q2_K",
    "IQ2_XS",
    "FP16",
    "BF16"
  ];

  for (const q of preference) {
    if (quants.includes(q)) return q;
  }

  return quants[0] || null;
}

function estimateParamsB(repoId, model) {
  const source = `${repoId} ${(model.tags || []).join(" ")}`.toLowerCase();

  const patterns = [
    /(\d+(?:\.\d+)?)\s*b(?:illion)?\b/i,
    /(\d+(?:\.\d+)?)[-_ ]?b[-_ ]/i,
    /[-_ ](\d+(?:\.\d+)?)b[-_ ]/i
  ];

  for (const re of patterns) {
    const m = source.match(re);
    if (m) {
      const n = Number(m[1]);
      if (n > 0 && n < 1000) return n;
    }
  }

  if (source.includes("70b")) return 70;
  if (source.includes("34b")) return 34;
  if (source.includes("32b")) return 32;
  if (source.includes("14b")) return 14;
  if (source.includes("13b")) return 13;
  if (source.includes("12b")) return 12;
  if (source.includes("8b")) return 8;
  if (source.includes("7b")) return 7;
  if (source.includes("3b")) return 3;
  if (source.includes("1b")) return 1;

  return null;
}

function estimateMemoryGb(paramsB, quant, model) {
  if (!paramsB) {
    if (model.formats.includes("gguf")) return 8;
    if (model.formats.includes("safetensors")) return 16;
    return null;
  }

  const q = String(quant || "").toUpperCase();

  let perB = 1.15;

  if (q.includes("Q2") || q.includes("IQ2")) perB = 0.35;
  else if (q.includes("Q3") || q.includes("IQ3")) perB = 0.45;
  else if (q.includes("Q4") || q.includes("IQ4")) perB = 0.60;
  else if (q.includes("Q5")) perB = 0.75;
  else if (q.includes("Q6")) perB = 0.90;
  else if (q.includes("Q8")) perB = 1.15;
  else if (q.includes("FP16") || q.includes("BF16")) perB = 2.0;

  const overhead = paramsB <= 3 ? 1.0 : paramsB <= 8 ? 1.5 : 2.5;

  return round1(paramsB * perB + overhead);
}

function detectCompatibleApps(model) {
  const apps = new Set();

  if (model.formats.includes("gguf")) {
    apps.add("ollama");
    apps.add("llama.cpp");
    apps.add("lm-studio");
    apps.add("jan");
  }

  if (model.formats.includes("safetensors")) {
    apps.add("vllm");
    apps.add("sglang");
  }

  if (model.formats.includes("mlx")) {
    apps.add("mlx-lm");
  }

  if (model.paramsB && model.paramsB <= 7) {
    apps.add("colab-free");
  }

  if (model.paramsB && model.paramsB <= 3) {
    apps.add("hf-zerogpu");
  }

  return [...apps].sort();
}

function scoreHardwareFit(estimatedMemoryGb, prefs, model) {
  if (!estimatedMemoryGb) return 45;

  const target = prefs.target;

  if (target === "colab-free") {
    if (estimatedMemoryGb <= 8) return 95;
    if (estimatedMemoryGb <= 12) return 80;
    if (estimatedMemoryGb <= 16) return 55;
    return 20;
  }

  if (target === "hf-zerogpu" || target === "zerogpu") {
    if (model.paramsB && model.paramsB <= 3) return 90;
    if (model.paramsB && model.paramsB <= 7) return 55;
    return 20;
  }

  if (prefs.vramGb > 0 && ["vllm", "sglang"].includes(target)) {
    if (estimatedMemoryGb <= prefs.vramGb * 0.85) return 95;
    if (estimatedMemoryGb <= prefs.vramGb) return 80;
    if (estimatedMemoryGb <= prefs.vramGb * 1.25) return 45;
    return 15;
  }

  if (estimatedMemoryGb <= prefs.ramGb * 0.65) return 95;
  if (estimatedMemoryGb <= prefs.ramGb * 0.85) return 80;
  if (estimatedMemoryGb <= prefs.ramGb) return 60;
  if (estimatedMemoryGb <= prefs.ramGb * 1.25) return 30;

  return 10;
}

function scoreAppFit(apps, target) {
  const normalized = normalizeTarget(target);

  if (apps.includes(normalized)) return 100;

  if (normalized === "local" && apps.some((a) => ["ollama", "llama.cpp", "lm-studio", "jan"].includes(a))) {
    return 85;
  }

  if (normalized === "lm-studio" && apps.includes("llama.cpp")) return 80;
  if (normalized === "ollama" && apps.includes("llama.cpp")) return 75;
  if (normalized === "colab-free" && apps.includes("vllm")) return 60;

  return 25;
}

function scoreFormatFit(formats, target) {
  const normalized = normalizeTarget(target);

  if (["ollama", "llama.cpp", "lm-studio", "jan", "local"].includes(normalized)) {
    return formats.includes("gguf") ? 100 : 30;
  }

  if (["vllm", "sglang", "colab-free"].includes(normalized)) {
    if (formats.includes("safetensors")) return 100;
    if (formats.includes("gguf")) return 65;
    return 30;
  }

  if (["mlx-lm", "apple"].includes(normalized)) {
    if (formats.includes("mlx")) return 100;
    if (formats.includes("gguf")) return 85;
    return 25;
  }

  if (normalized === "hf-zerogpu" || normalized === "zerogpu") {
    if (formats.includes("safetensors")) return 85;
    if (formats.includes("gguf")) return 60;
    return 35;
  }

  return formats.length ? 70 : 20;
}

function scorePopularity(downloads, likes) {
  const d = Math.min(70, Math.log10(downloads + 1) * 12);
  const l = Math.min(30, Math.log10(likes + 1) * 12);
  return Math.round(d + l);
}

function scoreLicense(license) {
  const l = String(license || "").toLowerCase();

  if (!l) return 50;
  if (["apache-2.0", "mit", "bsd-3-clause", "bsd", "cc-by-4.0"].some((x) => l.includes(x))) {
    return 100;
  }
  if (l.includes("llama") || l.includes("gemma")) return 70;
  if (l.includes("unknown") || l.includes("other")) return 35;

  return 60;
}

function scoreFreshness(lastModified) {
  if (!lastModified) return 50;

  const t = Date.parse(lastModified);
  if (!Number.isFinite(t)) return 50;

  const days = (Date.now() - t) / 86400000;

  if (days <= 7) return 100;
  if (days <= 30) return 90;
  if (days <= 90) return 75;
  if (days <= 365) return 55;

  return 35;
}

function scoreUseCase(model, useCase) {
  const text = `${model.repoId} ${model.name} ${(model.tags || []).join(" ")} ${model.task}`.toLowerCase();

  if (useCase === "coding") {
    return /coder|code|coding|starcoder|deepseek-coder|qwen.*coder/.test(text) ? 100 : 55;
  }

  if (useCase === "rag") {
    return /long|context|instruct|embedding|retrieval/.test(text) ? 90 : 60;
  }

  if (useCase === "agent") {
    return /instruct|chat|tool|function|json|reasoning/.test(text) ? 90 : 55;
  }

  if (useCase === "json") {
    return /json|instruct|chat|function|tool/.test(text) ? 90 : 55;
  }

  if (useCase === "embeddings") {
    return /embed|embedding|sentence/.test(text) ? 100 : 5;
  }

  return /instruct|chat/.test(text) ? 85 : 60;
}

function buildCommands(repoId, quant) {
  return {
    ollama: `ollama run hf.co/${repoId}${quant ? ":" + quant : ""}`,
    llamaCppServer: `llama-server -hf ${repoId}${quant ? ":" + quant : ""}`,
    llamaCppCli: `llama-cli -hf ${repoId}${quant ? ":" + quant : ""}`,
    vllm: `vllm serve ${repoId}`
  };
}

function buildWhy(model, prefs, meta) {
  const why = [];

  if (model.formats.includes("gguf")) {
    why.push("Repository appears to provide GGUF files, which are commonly used by Ollama, llama.cpp, LM Studio, and Jan.");
  }

  if (model.formats.includes("safetensors")) {
    why.push("Repository appears to provide safetensors files, which are commonly used by Transformers/vLLM-style serving.");
  }

  if (meta.recommendedQuant) {
    why.push(`Recommended quantization detected: ${meta.recommendedQuant}.`);
  }

  if (model.paramsB) {
    why.push(`Estimated model size: ${model.paramsB}B parameters.`);
  }

  if (meta.estimatedMemoryGb) {
    why.push(`Estimated memory requirement: about ${meta.estimatedMemoryGb} GB.`);
  }

  if (meta.hardwareFit >= 80) {
    why.push("Estimated memory fits the selected hardware profile.");
  } else if (meta.hardwareFit >= 50) {
    why.push("Estimated memory may fit, but runtime settings and context length matter.");
  }

  if (meta.compatibleApps.includes(normalizeTarget(prefs.target))) {
    why.push(`Direct compatibility detected for target: ${prefs.target}.`);
  }

  return why;
}

function buildWarnings(model, prefs, meta) {
  const warnings = [];

  if (prefs.target === "colab-free") {
    warnings.push("Colab Free GPU availability is not guaranteed. Treat this as an estimate.");
  }

  if (prefs.target === "hf-zerogpu" || prefs.target === "zerogpu") {
    warnings.push("ZeroGPU is best for short Spaces demos, not persistent always-on model servers.");
  }

  if (meta.estimatedMemoryGb && meta.estimatedMemoryGb > prefs.ramGb && prefs.vramGb === 0) {
    warnings.push("Estimated memory is higher than selected RAM. Choose a smaller quant or smaller model.");
  }

  if (!model.formats.includes("gguf") && ["ollama", "llama.cpp", "lm-studio", "jan"].includes(normalizeTarget(prefs.target))) {
    warnings.push("No GGUF format detected, so local desktop app compatibility may be limited.");
  }

  if (!model.license) {
    warnings.push("License metadata was not detected. Check the model card before commercial use.");
  }

  return warnings;
}

function detectLicense(model) {
  const tags = Array.isArray(model.tags) ? model.tags : [];

  for (const tag of tags) {
    if (String(tag).startsWith("license:")) {
      return String(tag).replace("license:", "");
    }
  }

  if (model.cardData && model.cardData.license) {
    return model.cardData.license;
  }

  return null;
}

function guessTask(model) {
  const text = `${model.id || ""} ${(model.tags || []).join(" ")}`.toLowerCase();

  if (text.includes("text-generation")) return "text-generation";
  if (text.includes("text2text-generation")) return "text2text-generation";
  if (text.includes("sentence-similarity")) return "sentence-similarity";
  if (text.includes("feature-extraction")) return "feature-extraction";

  return null;
}

function readableModelName(repoId) {
  return repoId
    .split("/")
    .pop()
    .replace(/[-_]+/g, " ")
    .replace(/\bgguf\b/gi, "GGUF")
    .trim();
}

function hfLocalAppUrl(repoId, app) {
  return `https://huggingface.co/${repoId}?local-app=${encodeURIComponent(app)}`;
}

function normalizeTarget(target) {
  const t = String(target || "").toLowerCase();

  if (t === "llamacpp" || t === "llama-cpp") return "llama.cpp";
  if (t === "lmstudio") return "lm-studio";
  if (t === "zero-gpu") return "hf-zerogpu";
  if (t === "apple-silicon") return "apple";

  return t;
}

function sortQuants(a, b) {
  const order = [
    "Q4_K_M",
    "Q5_K_M",
    "Q4_K_S",
    "Q5_K_S",
    "Q6_K",
    "Q8_0",
    "Q4_0",
    "IQ4_XS",
    "Q3_K_M",
    "Q3_K_S",
    "IQ3_XS",
    "Q2_K",
    "IQ2_XS",
    "FP16",
    "BF16"
  ];

  return (order.indexOf(a) === -1 ? 999 : order.indexOf(a)) -
    (order.indexOf(b) === -1 ? 999 : order.indexOf(b));
}

function cleanText(value) {
  return String(value || "")
    .replace(/[<>]/g, "")
    .trim()
    .slice(0, 120);
}

function clampNumber(value, min, max, fallback) {
  const n = Number(value);

  if (!Number.isFinite(n)) return fallback;

  return Math.max(min, Math.min(max, n));
}

function numberOrZero(value) {
  const n = Number(value);

  return Number.isFinite(n) ? n : 0;
}

function round1(value) {
  return Math.round(value * 10) / 10;
}

function escapeRegExp(value) {
  return String(value).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function corsHeaders(request) {
  const origin = request.headers.get("Origin") || "";

  const allowOrigin = ALLOWED_ORIGINS.includes(origin)
    ? origin
    : ALLOWED_ORIGINS[0];

  return {
    "Access-Control-Allow-Origin": allowOrigin,
    "Access-Control-Allow-Methods": "GET,OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, Authorization",
    "Access-Control-Max-Age": "86400",
    "Vary": "Origin"
  };
}

function corsResponse(request) {
  return new Response(null, {
    status: 204,
    headers: corsHeaders(request)
  });
}

function withCors(request, response) {
  const headers = new Headers(response.headers);

  for (const [key, value] of Object.entries(corsHeaders(request))) {
    headers.set(key, value);
  }

  return new Response(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers
  });
}

function json(request, data, status = 200, extraHeaders = {}) {
  return new Response(JSON.stringify(data, null, 2), {
    status,
    headers: {
      "Content-Type": "application/json; charset=utf-8",
      ...extraHeaders,
      ...corsHeaders(request)
    }
  });
}
