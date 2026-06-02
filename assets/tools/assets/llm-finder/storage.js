// localStorage bridge between the finder (page 1) and the report (page 2).

const K_MODEL = 'rmv-selected-llm-model';
const K_QUERY = 'rmv-selected-llm-query';
const K_FILTERS = 'rmv-llm-last-filters';

function set(key, value) {
  try { localStorage.setItem(key, JSON.stringify(value)); } catch (e) { /* ignore */ }
}
function get(key) {
  try { return JSON.parse(localStorage.getItem(key)); } catch (e) { return null; }
}

export const saveSelectedModel = (m) => set(K_MODEL, m);
export const loadSelectedModel = () => get(K_MODEL);

export const saveQuery = (q) => set(K_QUERY, q);
export const loadQuery = () => get(K_QUERY);

export const saveFilters = (f) => set(K_FILTERS, f);
export const loadFilters = () => get(K_FILTERS);
