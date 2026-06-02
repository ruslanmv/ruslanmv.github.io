import { copyText, escapeHtml, formatDateTime, formatTime, setText, showToast } from '../shared/tool-shell.js';
import { downloadJson, downloadFile, timestampSlug } from '../shared/export.js';
import { saveState } from '../shared/storage.js';
import { getPublicIp, guessIpVersion } from './ip-services.js';

let currentReport = null;

function collectBrowserInfo() {
  const ua = navigator.userAgent || '';
  return {
    userAgent: ua,
    browser: getBrowserName(ua),
    operatingSystem: getOSName(ua),
    language: navigator.language || '',
    languages: navigator.languages || [],
    screenSize: `${window.screen?.width || window.innerWidth} × ${window.screen?.height || window.innerHeight}`,
    viewport: `${window.innerWidth} × ${window.innerHeight}`,
    colorScheme: matchMedia('(prefers-color-scheme: dark)').matches ? 'Dark' : 'Light',
    online: navigator.onLine,
    platform: navigator.platform || '',
    hardwareConcurrency: navigator.hardwareConcurrency || null,
    deviceMemory: navigator.deviceMemory || null,
    timezone: Intl.DateTimeFormat().resolvedOptions().timeZone || '',
    localTime: new Date().toISOString()
  };
}

function getBrowserName(ua) {
  if (/Edg\//.test(ua)) return 'Microsoft Edge';
  if (/Chrome\//.test(ua) && !/Chromium/.test(ua)) return 'Google Chrome';
  if (/Firefox\//.test(ua)) return 'Mozilla Firefox';
  if (/Safari\//.test(ua) && !/Chrome\//.test(ua)) return 'Safari';
  return 'Unknown browser';
}

function getOSName(ua) {
  if (/Windows NT 10/.test(ua)) return 'Windows 10/11';
  if (/Mac OS X/.test(ua)) return 'macOS';
  if (/Linux/.test(ua)) return 'Linux';
  if (/Android/.test(ua)) return 'Android';
  if (/iPhone|iPad/.test(ua)) return 'iOS / iPadOS';
  return 'Unknown OS';
}

function buildReport(ipData) {
  const browser = collectBrowserInfo();
  const publicNet = {
    ip: ipData.ip || '',
    version: ipData.version || guessIpVersion(ipData.ip),
    isp: ipData.asOrganization || ipData.org || '',
    asn: ipData.asn || '',
    country: ipData.country || ipData.countryCode || '',
    countryCode: ipData.countryCode || '',
    region: ipData.region || '',
    city: ipData.city || '',
    timezone: ipData.timezone || browser.timezone,
    source: ipData.source || 'Cloudflare Worker / IP service'
  };
  return {
    public: publicNet,
    browser,
    security: {
      vpnProxyHint: 'Unknown from browser-only checks',
      webRtcAvailability: 'Browser capability detected',
      httpsStatus: location.protocol === 'https:' ? 'Secure' : 'Not HTTPS',
      supportSummary: 'Useful details for troubleshooting API, VPN, and browser issues.'
    },
    detectedAt: new Date().toISOString()
  };
}

function renderInfoGrid(id, rows) {
  const el = document.getElementById(id);
  if (!el) return;
  el.innerHTML = rows.map(([label, value]) => `
    <div class="info-label">${escapeHtml(label)}</div>
    <div class="info-value">${escapeHtml(value || '—')}</div>
  `).join('');
}

function renderReport(report) {
  currentReport = report;
  saveState('rmv-ip-inspector-last-report', report);

  setText('ipDisplay', report.public.ip || 'Unavailable');
  setText('ipVersionBadge', report.public.version || 'IP');
  setText('ipVersionValue', report.public.version || '—');
  setText('regionValue', report.public.country || '—');
  setText('regionNote', [report.public.city, report.public.region].filter(Boolean).join(', ') || '—');
  setText('updatedTime', formatTime(report.detectedAt));
  setText('updatedNote', formatDateTime(report.detectedAt));

  renderInfoGrid('publicGrid', [
    ['ISP / Organization', report.public.isp],
    ['Country', [report.public.countryCode ? `(${report.public.countryCode})` : '', report.public.country].filter(Boolean).join(' ')],
    ['City', [report.public.city, report.public.region].filter(Boolean).join(', ')],
    ['Timezone', report.public.timezone],
    ['ASN', report.public.asn],
    ['Source', report.public.source]
  ]);

  renderInfoGrid('browserGrid', [
    ['User Agent', report.browser.userAgent],
    ['Browser', report.browser.browser],
    ['Operating System', report.browser.operatingSystem],
    ['Language', report.browser.language],
    ['Screen Size', report.browser.screenSize],
    ['Viewport', report.browser.viewport],
    ['Color Scheme', report.browser.colorScheme],
    ['Online Status', report.browser.online ? 'Online' : 'Offline'],
    ['Time Detected', formatDateTime(report.detectedAt)]
  ]);

  const json = JSON.stringify(report, null, 2);
  setText('reportPreview', json);
}

async function refreshIp() {
  setText('ipDisplay', 'Loading…');
  let ipData = null;
  let ipError = null;
  try {
    ipData = await getPublicIp();
  } catch (error) {
    ipError = error;
  }

  // Always render browser diagnostics + report — these are local and work even
  // when no IP service (Worker / ipapi / ipify) is reachable.
  renderReport(buildReport(ipData || { ip: '', source: 'Unavailable — IP lookup failed' }));

  if (ipError) {
    setText('ipDisplay', 'Could not detect IP');
    setText('ipVersionBadge', 'No IP');
    showToast('IP lookup failed — browser diagnostics still shown');
  } else {
    showToast('Network report refreshed');
  }
}

function downloadMarkdownReport() {
  if (!currentReport) return showToast('No report available');
  const p = currentReport.public;
  const b = currentReport.browser;
  const md = `# IP & Network Inspector Report\n\nGenerated: ${formatDateTime(currentReport.detectedAt)}\n\n## Public Network\n- IP: ${p.ip}\n- Version: ${p.version}\n- ISP / Organization: ${p.isp || '—'}\n- ASN: ${p.asn || '—'}\n- Country: ${p.country || '—'}\n- City: ${p.city || '—'}\n- Timezone: ${p.timezone || '—'}\n\n## Browser\n- Browser: ${b.browser}\n- OS: ${b.operatingSystem}\n- Language: ${b.language}\n- Screen: ${b.screenSize}\n- Online: ${b.online ? 'Yes' : 'No'}\n\n## Raw JSON\n\n\`\`\`json\n${JSON.stringify(currentReport, null, 2)}\n\`\`\`\n`;
  downloadFile(`ip-network-report-${timestampSlug()}.md`, md, 'text/markdown;charset=utf-8');
}

function init() {
  document.getElementById('refreshBtn')?.addEventListener('click', refreshIp);
  document.getElementById('copyIpBtn')?.addEventListener('click', () => copyText(currentReport?.public?.ip || '', 'IP copied'));
  document.getElementById('downloadReportBtn')?.addEventListener('click', downloadMarkdownReport);
  document.getElementById('downloadJsonBtn')?.addEventListener('click', () => currentReport && downloadJson(`ip-network-report-${timestampSlug()}.json`, currentReport));
  document.getElementById('copyReportBtn')?.addEventListener('click', () => copyText(JSON.stringify(currentReport, null, 2), 'Report copied'));
  refreshIp();
}

document.addEventListener('DOMContentLoaded', init);
