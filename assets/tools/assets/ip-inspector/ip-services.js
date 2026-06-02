const DEFAULT_WORKER_BASE = 'https://network-api-tools.cloud-data.workers.dev';

export async function getPublicIp() {
  const apiBase = window.RMV_API_BASE || DEFAULT_WORKER_BASE;
  if (apiBase) {
    try {
      const res = await fetch(`${apiBase.replace(/\/$/, '')}/api/ip`, { cache: 'no-store' });
      if (res.ok) return await res.json();
    } catch (_) { /* fall back below */ }
  }

  // Public fallback for GitHub Pages/static hosting. Metadata availability varies by service.
  try {
    const res = await fetch('https://ipapi.co/json/', { cache: 'no-store' });
    if (res.ok) {
      const data = await res.json();
      return {
        ip: data.ip,
        version: data.version || guessIpVersion(data.ip),
        country: data.country_name || data.country,
        countryCode: data.country_code,
        city: data.city,
        region: data.region,
        timezone: data.timezone,
        asn: data.asn,
        asOrganization: data.org || data.asn_org,
        source: 'ipapi.co',
        detectedAt: new Date().toISOString()
      };
    }
  } catch (_) { /* continue */ }

  const res = await fetch('https://api.ipify.org?format=json', { cache: 'no-store' });
  if (!res.ok) throw new Error('Unable to fetch public IP');
  const data = await res.json();
  return {
    ip: data.ip,
    version: guessIpVersion(data.ip),
    source: 'api.ipify.org',
    detectedAt: new Date().toISOString()
  };
}

export function guessIpVersion(ip = '') {
  return String(ip).includes(':') ? 'IPv6' : 'IPv4';
}
