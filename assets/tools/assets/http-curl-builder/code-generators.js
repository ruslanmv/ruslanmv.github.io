import { buildEffectiveHeaders, buildUrlWithParams } from './curl-generator.js';

export function buildFetchCode(request) {
  const method = (request.method || 'GET').toUpperCase();
  const bodyLine = request.body && !['GET','HEAD'].includes(method) ? `,\n  body: ${JSON.stringify(request.body)}` : '';
  return `const response = await fetch(${JSON.stringify(buildUrlWithParams(request.url, request.params))}, {\n  method: ${JSON.stringify(method)},\n  headers: ${JSON.stringify(buildEffectiveHeaders(request), null, 2)}${bodyLine}\n});\n\nconst data = await response.text();`;
}

export function buildPythonRequestsCode(request) {
  const method = (request.method || 'GET').toLowerCase();
  const bodyLine = request.body && !['get','head'].includes(method) ? `,\n    data=${JSON.stringify(request.body)}` : '';
  return `import requests\n\nresponse = requests.${method}(\n    ${JSON.stringify(buildUrlWithParams(request.url, request.params))},\n    headers=${JSON.stringify(buildEffectiveHeaders(request), null, 4)}${bodyLine}\n)\n\nprint(response.status_code)\nprint(response.text)`;
}

export function buildAxiosCode(request) {
  const method = (request.method || 'GET').toUpperCase();
  const dataLine = request.body && !['GET','HEAD'].includes(method) ? `,\n  data: ${JSON.stringify(request.body)}` : '';
  return `import axios from 'axios';\n\nconst response = await axios({\n  method: ${JSON.stringify(method)},\n  url: ${JSON.stringify(buildUrlWithParams(request.url, request.params))},\n  headers: ${JSON.stringify(buildEffectiveHeaders(request), null, 2)}${dataLine}\n});\n\nconsole.log(response.data);`;
}
