/**
 * SILT Studio bridge — hosted site → local engine.
 *
 * This script runs on the public SILT landing site and the hosted /studio page.
 * It forwards same-origin fetch() and EventSource requests under /api/* to the
 * configured local engine, defaulting to http://127.0.0.1:8377.
 *
 * Security constraints (do not weaken):
 *   - The engine URL is restricted to loopback / private-network hosts.
 *   - Credentials are never forwarded; the local engine owns its own session.
 *   - No wildcard origin handling happens here — that is the engine's job.
 *   - Public HTTPS origins get a setup panel instead of attempting mixed-content.
 */

(function () {
  'use strict';

  const DEFAULT_ENGINE_URL = 'http://127.0.0.1:8377';
  const API_PREFIX = '/api/';
  const STORAGE_KEY = 'silt.engineUrl';
  const PRIVATE_HOSTS = new Set(['localhost', '127.0.0.1', '[::1]']);

  function isLoopbackOrPrivateHost(hostname) {
    const h = hostname.toLowerCase();
    if (PRIVATE_HOSTS.has(h)) return true;
    if (h.startsWith('127.')) return true;
    if (h === '::1') return true;
    if (/^10\./.test(h)) return true;
    if (/^172\.(1[6-9]|2[0-9]|3[0-1])\./.test(h)) return true;
    if (/^192\.168\./.test(h)) return true;
    return false;
  }

  function isAllowedEngineUrl(url) {
    try {
      const u = new URL(url);
      if (u.protocol !== 'http:' && u.protocol !== 'https:') return false;
      return isLoopbackOrPrivateHost(u.hostname);
    } catch {
      return false;
    }
  }

  function isPublicHttpsPage() {
    return window.location.protocol === 'https:' && !isLoopbackOrPrivateHost(window.location.hostname);
  }

  function getEngineUrl() {
    const stored = window.localStorage.getItem(STORAGE_KEY);
    if (stored && isAllowedEngineUrl(stored)) return stored;
    return DEFAULT_ENGINE_URL;
  }

  function setEngineUrl(url) {
    if (!isAllowedEngineUrl(url)) {
      throw new Error('SILT engine URL must be a loopback or private-network host');
    }
    window.localStorage.setItem(STORAGE_KEY, url);
    return url;
  }

  function resetEngineUrl() {
    window.localStorage.removeItem(STORAGE_KEY);
    return DEFAULT_ENGINE_URL;
  }

  function shouldBridge(url) {
    if (isPublicHttpsPage()) return false;
    try {
      const u = new URL(url, window.location.href);
      if (u.origin !== window.location.origin) return false;
      return u.pathname.startsWith(API_PREFIX);
    } catch {
      return false;
    }
  }

  function bridgeUrl(url) {
    const engine = getEngineUrl();
    const u = new URL(url, window.location.href);
    return `${engine}${u.pathname}${u.search}`;
  }

  async function health() {
    const engine = getEngineUrl();
    const controller = new AbortController();
    const t = setTimeout(() => controller.abort(), 3000);
    try {
      const res = await fetch(`${engine}/api/health`, {
        method: 'GET',
        signal: controller.signal,
        headers: { Accept: 'application/json' },
        credentials: 'omit',
      });
      clearTimeout(t);
      return { ok: res.ok, status: res.status, engine };
    } catch (e) {
      clearTimeout(t);
      return { ok: false, error: e.message, engine };
    }
  }

  function installFetchBridge() {
    const originalFetch = window.fetch;
    window.fetch = function (resource, init = {}) {
      const url = typeof resource === 'string' ? resource : resource.url;
      if (!shouldBridge(url)) return originalFetch.apply(this, arguments);

      const target = bridgeUrl(url);
      const cfg = { ...init };
      cfg.headers = cfg.headers ? new Headers(cfg.headers) : new Headers();
      cfg.headers.set('X-SILT-Bridge-Origin', window.location.origin);
      cfg.credentials = 'omit';
      return originalFetch(target, cfg);
    };
  }

  function installEventSourceBridge() {
    const OriginalEventSource = window.EventSource;
    window.EventSource = function (url, options) {
      if (!shouldBridge(url)) {
        return new OriginalEventSource(url, options);
      }
      return new OriginalEventSource(bridgeUrl(url), options);
    };
    Object.setPrototypeOf(window.EventSource, OriginalEventSource);
    window.EventSource.prototype = OriginalEventSource.prototype;
  }

  function renderPanel() {
    const mount = document.getElementById('silt-studio-panel');
    if (!mount) return;

    const publicHttps = isPublicHttpsPage();
    const engine = getEngineUrl();

    mount.innerHTML = `
      <div id="silt-bridge" style="background:#0f1626;border:1px solid #1d2a44;border-radius:14px;padding:18px 20px;margin:18px 0;color:#e6edf7;font:14px/1.6 Inter,system-ui,sans-serif">
        <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap">
          <span id="silt-bridge-dot" style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${publicHttps ? '#fbbf24' : '#51d88a'}"></span>
          <strong style="font-size:15px">SILT Studio bridge</strong>
          <span id="silt-bridge-status" style="margin-left:auto;font:12px 'JetBrains Mono',monospace;color:#9db0cc">${publicHttps ? 'Local engine required' : 'Ready'}</span>
        </div>
        <p id="silt-bridge-msg" style="margin-top:10px;color:#9db0cc;font-size:13px">
          ${publicHttps
            ? 'The hosted site cannot reach a local HTTP engine from HTTPS. Start the engine and open it directly on <code>127.0.0.1:8377</code>. The bridge activates automatically on loopback.'
            : `This page is on a loopback / private host. API calls under <code>/api/</code> are forwarded to <code>${engine}</code>.`}
        </p>
        <div id="silt-bridge-engine" style="${publicHttps ? 'display:none' : ''};margin-top:12px">
          <label style="display:block;font-size:12px;color:#5f7396;margin-bottom:4px">Engine URL (loopback / private-network only)</label>
          <input id="silt-bridge-url" type="text" value="${engine}" style="width:100%;max-width:420px;background:#0a0f1a;border:1px solid #1d2a44;border-radius:8px;color:#e6edf7;padding:8px 10px;font:12px 'JetBrains Mono',monospace">
          <div style="margin-top:8px;display:flex;gap:10px;flex-wrap:wrap">
            <button id="silt-bridge-set" style="background:#17c3ad;color:#03211c;border:none;border-radius:6px;padding:6px 12px;font-weight:600;cursor:pointer">Set URL</button>
            <button id="silt-bridge-reset" style="background:transparent;border:1px solid #1d2a44;color:#17c3a3;border-radius:6px;padding:6px 12px;cursor:pointer">Reset to default</button>
          </div>
        </div>
        <div style="margin-top:12px;display:flex;gap:10px;flex-wrap:wrap">
          <button id="silt-bridge-health" style="background:transparent;border:1px solid #1d2a44;color:#17c3a3;border-radius:6px;padding:6px 12px;cursor:pointer">Check health</button>
          <a href="https://github.com/inbharatai/SILT#silt-studio" target="_blank" rel="noopener noreferrer" style="border:1px solid #1d2a44;color:#17c3a3;border-radius:6px;padding:6px 12px;text-decoration:none;font-weight:600">Setup guide</a>
        </div>
        <pre id="silt-bridge-log" style="margin-top:12px;background:#0a0f1a;border:1px solid #1d2a44;border-radius:8px;padding:10px;font:12px 'JetBrains Mono',monospace;color:#5f7396;min-height:1.2em;white-space:pre-wrap"></pre>
      </div>
    `;

    document.getElementById('silt-bridge-health').addEventListener('click', runHealthCheck);
    const setBtn = document.getElementById('silt-bridge-set');
    if (setBtn) setBtn.addEventListener('click', () => {
      const input = document.getElementById('silt-bridge-url');
      try {
        setEngineUrl(input.value.trim());
        log(`Engine URL set to ${input.value.trim()}.`);
        runHealthCheck();
      } catch (e) {
        log(`Error: ${e.message}`);
      }
    });
    const resetBtn = document.getElementById('silt-bridge-reset');
    if (resetBtn) resetBtn.addEventListener('click', () => {
      const url = resetEngineUrl();
      document.getElementById('silt-bridge-url').value = url;
      log(`Engine URL reset to default (${url}).`);
      runHealthCheck();
    });
  }

  function log(msg) {
    const el = document.getElementById('silt-bridge-log');
    if (el) el.textContent = msg;
  }

  function setStatus(text, color) {
    const dot = document.getElementById('silt-bridge-dot');
    const label = document.getElementById('silt-bridge-status');
    if (dot) dot.style.background = color;
    if (label) label.textContent = text;
  }

  async function runHealthCheck() {
    log('Checking local engine…');
    const result = await health();
    if (result.ok) {
      setStatus('Engine reachable', '#51d88a');
      log(`Engine reachable at ${result.engine} — HTTP ${result.status}.`);
    } else {
      setStatus('Engine unreachable', '#ff6b6b');
      log(`Could not reach ${result.engine}${result.error ? ': ' + result.error : ''}. Start the engine first (see README).`);
    }
  }

  function init() {
    window.siltEngine = {
      health,
      setUrl: setEngineUrl,
      resetUrl: resetEngineUrl,
      getUrl: getEngineUrl,
    };
    renderPanel();
    if (!isPublicHttpsPage()) {
      installFetchBridge();
      installEventSourceBridge();
      setTimeout(runHealthCheck, 500);
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
