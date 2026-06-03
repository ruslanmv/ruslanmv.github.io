/**
 * Cloudflare Worker backend for the RuslanMV A2A Agent Validator.
 *
 * Frontend tool lives at:
 *   https://ruslanmv.com/tools/a2a-validator.html
 *   https://ruslanmv.github.io/assets/tools/a2a-validator.html   (GitHub Pages origin/fallback)
 *   http://localhost:4000/ or http://127.0.0.1:4001/ during Jekyll development
 *
 * Deployed Worker URL:
 *   https://a2a-validator.cloud-data.workers.dev
 *
 * Endpoints:
 *   GET     /            - service info + endpoint list
 *   GET     /health      - liveness probe
 *   OPTIONS /validate    - CORS preflight
 *   GET     /validate?url=https://your-agent.example.com[&bearer_token=…]
 *   POST    /validate    - JSON body { "url": "…", "auth": { "type": "…", … } }
 *
 * It fetches the agent's A2A Agent Card at `<url>/.well-known/agent.json`,
 * validates the basic shape (required fields, skills, capabilities) and returns
 * a structured result.
 *
 * Authentication (POST recommended so secrets never appear in the URL):
 *   - none            : no auth
 *   - bearer          : { "token": "…" }            -> Authorization: Bearer …
 *   - basic           : { "username": "…", "password": "…" } -> Authorization: Basic …
 *   - custom_headers  : { "headers": { "X-API-Key": "…" } }
 *
 * Safety: HTTP/HTTPS only, localhost + private IPv4 ranges are blocked so the
 * Worker cannot be used to probe internal networks. Hop-by-hop and spoofable
 * headers are rejected from custom_headers. Secrets are not logged or stored.
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
  "https://ruslanmv.github.io",
];

const REQUIRED_FIELDS = [
  "name",
  "description",
  "url",
  "version",
  "capabilities",
  "skills",
];

const BLOCKED_CUSTOM_HEADERS = new Set([
  "host",
  "content-length",
  "connection",
  "transfer-encoding",
  "upgrade",
  "cf-connecting-ip",
  "cf-ipcountry",
  "cf-ray",
  "x-forwarded-for",
  "x-real-ip",
]);

function getCorsHeaders(request) {
  const origin = request.headers.get("Origin");

  const headers = {
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, Authorization",
    "Access-Control-Max-Age": "86400",
  };

  if (origin && ALLOWED_ORIGINS.includes(origin)) {
    headers["Access-Control-Allow-Origin"] = origin;
    headers["Vary"] = "Origin";
  }

  return headers;
}

function jsonResponse(request, data, status = 200) {
  return new Response(JSON.stringify(data, null, 2), {
    status,
    headers: {
      "Content-Type": "application/json; charset=utf-8",
      ...getCorsHeaders(request),
    },
  });
}

function normalizeAgentCardUrl(inputUrl) {
  const clean = inputUrl.trim().replace(/\/+$/, "");

  if (clean.endsWith("/.well-known/agent.json")) {
    return clean;
  }

  return `${clean}/.well-known/agent.json`;
}

function isPrivateIPv4(hostname) {
  return (
    hostname.startsWith("10.") ||
    hostname.startsWith("192.168.") ||
    /^172\.(1[6-9]|2[0-9]|3[0-1])\./.test(hostname) ||
    /^169\.254\./.test(hostname)
  );
}

function validateInputUrl(inputUrl) {
  let parsed;

  try {
    parsed = new URL(inputUrl);
  } catch {
    return {
      ok: false,
      error: "Invalid URL.",
    };
  }

  if (!["http:", "https:"].includes(parsed.protocol)) {
    return {
      ok: false,
      error: "Only http and https URLs are allowed.",
    };
  }

  const hostname = parsed.hostname.toLowerCase();

  const blockedHosts = new Set([
    "localhost",
    "127.0.0.1",
    "0.0.0.0",
    "::1",
    "[::1]",
  ]);

  if (blockedHosts.has(hostname)) {
    return {
      ok: false,
      error: "Localhost URLs are blocked.",
    };
  }

  if (isPrivateIPv4(hostname)) {
    return {
      ok: false,
      error: "Private network URLs are blocked.",
    };
  }

  return {
    ok: true,
    url: parsed.toString(),
  };
}

function validateAgentCardShape(card) {
  const errors = [];
  const warnings = [];

  if (!card || typeof card !== "object" || Array.isArray(card)) {
    errors.push("Agent card must be a JSON object.");
    return { errors, warnings };
  }

  for (const field of REQUIRED_FIELDS) {
    if (!(field in card)) {
      errors.push(`Missing required field: ${field}`);
    }
  }

  if ("name" in card && typeof card.name !== "string") {
    errors.push("Field 'name' should be a string.");
  }

  if ("description" in card && typeof card.description !== "string") {
    errors.push("Field 'description' should be a string.");
  }

  if ("url" in card && typeof card.url !== "string") {
    errors.push("Field 'url' should be a string.");
  }

  if ("version" in card && typeof card.version !== "string") {
    errors.push("Field 'version' should be a string.");
  }

  if (
    "capabilities" in card &&
    (!card.capabilities ||
      typeof card.capabilities !== "object" ||
      Array.isArray(card.capabilities))
  ) {
    errors.push("Field 'capabilities' should be an object.");
  }

  if ("skills" in card && !Array.isArray(card.skills)) {
    errors.push("Field 'skills' should be an array.");
  }

  if (Array.isArray(card.skills)) {
    card.skills.forEach((skill, index) => {
      if (!skill || typeof skill !== "object" || Array.isArray(skill)) {
        errors.push(`Skill at index ${index} should be an object.`);
        return;
      }

      if (!("id" in skill)) {
        warnings.push(`Skill at index ${index} is missing field: id.`);
      }

      if (!("name" in skill)) {
        warnings.push(`Skill at index ${index} is missing field: name.`);
      }

      if ("id" in skill && typeof skill.id !== "string") {
        errors.push(`Skill at index ${index} field 'id' should be a string.`);
      }

      if ("name" in skill && typeof skill.name !== "string") {
        errors.push(`Skill at index ${index} field 'name' should be a string.`);
      }

      if ("description" in skill && typeof skill.description !== "string") {
        errors.push(
          `Skill at index ${index} field 'description' should be a string.`
        );
      }
    });
  }

  return { errors, warnings };
}

function buildAuthHeaders(auth) {
  const headers = {
    Accept: "application/json",
    "User-Agent": "a2a-validator-cloudflare-worker",
  };

  if (!auth || !auth.type || auth.type === "none") {
    return {
      ok: true,
      headers,
      auth_type: "none",
    };
  }

  if (auth.type === "bearer") {
    if (!auth.token || typeof auth.token !== "string") {
      return {
        ok: false,
        error: "Bearer authentication requires a token.",
      };
    }

    headers.Authorization = `Bearer ${auth.token}`;

    return {
      ok: true,
      headers,
      auth_type: "bearer",
    };
  }

  if (auth.type === "basic") {
    if (!auth.username || !auth.password) {
      return {
        ok: false,
        error: "Basic authentication requires username and password.",
      };
    }

    const encoded = btoa(`${auth.username}:${auth.password}`);
    headers.Authorization = `Basic ${encoded}`;

    return {
      ok: true,
      headers,
      auth_type: "basic",
    };
  }

  if (auth.type === "custom_headers") {
    if (!auth.headers || typeof auth.headers !== "object" || Array.isArray(auth.headers)) {
      return {
        ok: false,
        error: "Custom headers authentication requires a headers object.",
      };
    }

    for (const [key, value] of Object.entries(auth.headers)) {
      const normalizedKey = String(key).toLowerCase();

      if (BLOCKED_CUSTOM_HEADERS.has(normalizedKey)) {
        return {
          ok: false,
          error: `Header "${key}" is not allowed.`,
        };
      }

      if (typeof value !== "string") {
        return {
          ok: false,
          error: `Header "${key}" must have a string value.`,
        };
      }

      headers[key] = value;
    }

    return {
      ok: true,
      headers,
      auth_type: "custom_headers",
    };
  }

  return {
    ok: false,
    error: `Unsupported authentication type: ${auth.type}`,
  };
}

async function validateAgent(agentUrl, auth) {
  const agentCardUrl = normalizeAgentCardUrl(agentUrl);
  const checked = validateInputUrl(agentCardUrl);

  if (!checked.ok) {
    return {
      valid: false,
      agent_card_url: agentCardUrl,
      error: checked.error,
    };
  }

  const authHeaders = buildAuthHeaders(auth);

  if (!authHeaders.ok) {
    return {
      valid: false,
      agent_card_url: agentCardUrl,
      error: authHeaders.error,
    };
  }

  let response;

  try {
    response = await fetch(checked.url, {
      method: "GET",
      headers: authHeaders.headers,
      redirect: "follow",
    });
  } catch (error) {
    return {
      valid: false,
      agent_card_url: agentCardUrl,
      auth_type: authHeaders.auth_type,
      error: `Fetch failed: ${error.message || String(error)}`,
    };
  }

  const result = {
    valid: false,
    agent_card_url: agentCardUrl,
    auth_type: authHeaders.auth_type,
    status_code: response.status,
    errors: [],
    warnings: [],
  };

  if (response.status === 401) {
    result.errors.push("Authentication failed. The endpoint returned HTTP 401.");
    return result;
  }

  if (response.status === 403) {
    result.errors.push("Access forbidden. The endpoint returned HTTP 403.");
    return result;
  }

  if (!response.ok) {
    result.errors.push(`Agent card endpoint returned HTTP ${response.status}.`);
    return result;
  }

  const contentType = response.headers.get("Content-Type") || "";

  if (!contentType.toLowerCase().includes("application/json")) {
    result.warnings.push(
      `Content-Type is "${contentType}", expected application/json.`
    );
  }

  let card;

  try {
    card = await response.json();
  } catch {
    result.errors.push("Response is not valid JSON.");
    return result;
  }

  const shape = validateAgentCardShape(card);

  result.errors = shape.errors;
  result.warnings = [...result.warnings, ...shape.warnings];
  result.valid = result.errors.length === 0;
  result.agent_card = card;

  if (result.valid) {
    result.message = "A2A agent card looks valid based on basic checks.";
  }

  return result;
}

async function handleValidate(request) {
  const requestUrl = new URL(request.url);

  let targetUrl = "";
  let auth = {
    type: "none",
  };

  if (request.method === "GET") {
    targetUrl = requestUrl.searchParams.get("url") || "";

    const bearerToken = requestUrl.searchParams.get("bearer_token");

    if (bearerToken) {
      auth = {
        type: "bearer",
        token: bearerToken,
      };
    }
  } else if (request.method === "POST") {
    let body;

    try {
      body = await request.json();
    } catch {
      return jsonResponse(
        request,
        {
          valid: false,
          error: "Invalid JSON body.",
        },
        400
      );
    }

    targetUrl = body.url || "";
    auth = body.auth || { type: "none" };
  } else {
    return jsonResponse(
      request,
      {
        error: "Method not allowed.",
      },
      405
    );
  }

  if (!targetUrl) {
    return jsonResponse(
      request,
      {
        valid: false,
        error:
          'Missing URL. Use /validate?url=https://your-agent.example.com or POST { "url": "https://your-agent.example.com" }.',
      },
      400
    );
  }

  const result = await validateAgent(targetUrl, auth);

  return jsonResponse(request, result, result.valid ? 200 : 422);
}

export default {
  async fetch(request) {
    const url = new URL(request.url);

    if (request.method === "OPTIONS") {
      return new Response(null, {
        status: 204,
        headers: getCorsHeaders(request),
      });
    }

    if (url.pathname === "/") {
      return jsonResponse(request, {
        name: "A2A Validator API",
        status: "running",
        worker_url: "https://a2a-validator.cloud-data.workers.dev",
        endpoints: {
          health: "GET /health",
          validate_get:
            "GET /validate?url=https://your-agent.example.com",
          validate_post:
            'POST /validate with JSON body { "url": "https://your-agent.example.com", "auth": { "type": "none" } }',
        },
        supported_authentication: [
          "none",
          "bearer",
          "basic",
          "custom_headers",
        ],
        allowed_origins: ALLOWED_ORIGINS,
      });
    }

    if (url.pathname === "/health") {
      return jsonResponse(request, {
        ok: true,
        service: "a2a-validator",
      });
    }

    if (url.pathname === "/validate") {
      return handleValidate(request);
    }

    return jsonResponse(
      request,
      {
        error: "Not found.",
        available_endpoints: ["/", "/health", "/validate"],
      },
      404
    );
  },
};
