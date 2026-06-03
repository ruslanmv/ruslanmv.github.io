// Pure helpers that turn a Worker validation result into a field-level
// checklist for the UI. Mirrors the Worker's REQUIRED_FIELDS so the frontend
// can show *which* fields passed, not just an error list.

export const REQUIRED_FIELDS = ['name', 'description', 'url', 'version', 'capabilities', 'skills'];

const TYPE_LABEL = {
  name: 'string',
  description: 'string',
  url: 'string',
  version: 'string',
  capabilities: 'object',
  skills: 'array'
};

function typeOk(field, value) {
  if (value === undefined) return false;
  switch (TYPE_LABEL[field]) {
    case 'array': return Array.isArray(value);
    case 'object': return value && typeof value === 'object' && !Array.isArray(value);
    default: return typeof value === 'string';
  }
}

// Builds the required-field checklist from the (possibly missing) agent card.
export function fieldChecks(card) {
  const obj = card && typeof card === 'object' && !Array.isArray(card) ? card : null;
  return REQUIRED_FIELDS.map((field) => {
    if (!obj) return { field, status: 'idle', detail: 'No card fetched yet' };
    const present = field in obj;
    if (!present) return { field, status: 'fail', detail: 'Missing' };
    const ok = typeOk(field, obj[field]);
    return {
      field,
      status: ok ? 'ok' : 'fail',
      detail: ok ? `${TYPE_LABEL[field]} ✓` : `Should be ${TYPE_LABEL[field]}`
    };
  });
}

// Higher-level checklist describing the whole validation lifecycle.
export function lifecycleChecks(result) {
  const r = result || {};
  const card = r.agent_card;
  const status = (cond, idleWhen) => (idleWhen ? 'idle' : cond ? 'ok' : 'fail');

  return [
    {
      label: 'Worker reachable',
      status: !r || (r.latencyMs == null && !('valid' in r)) ? 'idle' : (r.transportError ? 'fail' : 'ok'),
      detail: r.transportError ? 'Could not reach Worker' : (r.latencyMs != null ? `${r.latencyMs} ms` : 'Run a check')
    },
    {
      label: 'Agent card endpoint reachable',
      status: r.status_code ? (r.status_code >= 200 && r.status_code < 400 ? 'ok' : 'fail') : (r.error ? 'fail' : 'idle'),
      detail: r.status_code ? `HTTP ${r.status_code}` : (r.error || 'Not checked')
    },
    {
      label: 'Valid JSON agent card',
      status: card ? 'ok' : (r.errors?.some((e) => /JSON/i.test(e)) ? 'fail' : 'idle'),
      detail: card ? 'Parsed as JSON' : 'Not parsed'
    },
    {
      label: 'All required fields present & typed',
      status: card ? (r.valid ? 'ok' : 'fail') : 'idle',
      detail: card ? (r.valid ? 'name, description, url, version, capabilities, skills' : `${(r.errors || []).length} error(s)`) : '—'
    },
    {
      label: 'Skills well-formed',
      status: Array.isArray(card?.skills)
        ? ((r.warnings || []).some((w) => /Skill/.test(w)) || (r.errors || []).some((e) => /Skill/.test(e)) ? 'warn' : 'ok')
        : (card ? 'fail' : 'idle'),
      detail: Array.isArray(card?.skills) ? `${card.skills.length} skill(s)` : 'No skills array'
    }
  ];
}
