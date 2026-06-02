/* =============================================================================
   RMVTool — shared premium toolkit for the ruslanmv.com tools
   Framework-free, ~6 KB. Provides a consistent Copy / Download / Share-URL /
   Saved-workspaces / Trust-badges layer so every tool feels like one product.

   Everything is local-only: clipboard, Blob downloads, localStorage, and a
   state-in-URL share link. No network calls. Tools must degrade gracefully if
   this file fails to load (always guard with `if (window.RMVTool)`).
   ============================================================================= */
(function () {
  'use strict';
  if (window.RMVTool) return;

  function el(tag, cls, html) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (html != null) e.innerHTML = html;
    return e;
  }
  function escapeHtml(s) {
    return String(s == null ? '' : s).replace(/[&<>"]/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c];
    });
  }

  /* ---- toast --------------------------------------------------------------- */
  var toastEl = null, toastT = null;
  function toast(msg, ms) {
    if (!toastEl) { toastEl = el('div', 'rmv-toast'); document.body.appendChild(toastEl); }
    toastEl.textContent = msg;
    toastEl.classList.add('show');
    clearTimeout(toastT);
    toastT = setTimeout(function () { toastEl.classList.remove('show'); }, ms || 1900);
  }

  /* ---- copy ---------------------------------------------------------------- */
  function copy(text) {
    text = String(text == null ? '' : text);
    function ok() { toast('Copied to clipboard'); }
    function fallback(t) {
      try {
        var ta = el('textarea'); ta.value = t;
        ta.style.cssText = 'position:fixed;opacity:0;top:0;left:0';
        document.body.appendChild(ta); ta.focus(); ta.select();
        document.execCommand('copy'); document.body.removeChild(ta); ok();
      } catch (e) { toast('Copy failed'); }
    }
    if (navigator.clipboard && navigator.clipboard.writeText) {
      return navigator.clipboard.writeText(text).then(ok, function () { fallback(text); });
    }
    fallback(text);
    return Promise.resolve();
  }

  /* ---- download ------------------------------------------------------------ */
  function download(filename, content, mime) {
    try {
      var blob = new Blob([content], { type: (mime || 'text/plain') + ';charset=utf-8' });
      var url = URL.createObjectURL(blob);
      var a = el('a'); a.href = url; a.download = filename;
      document.body.appendChild(a); a.click();
      setTimeout(function () { URL.revokeObjectURL(url); a.remove(); }, 800);
      toast('Downloaded ' + filename);
    } catch (e) { toast('Download failed'); }
  }
  function toCSV(rows) {
    return rows.map(function (r) {
      return r.map(function (c) {
        c = String(c == null ? '' : c);
        return /[",\n]/.test(c) ? '"' + c.replace(/"/g, '""') + '"' : c;
      }).join(',');
    }).join('\r\n');
  }

  /* ---- share URL (state encoded in the hash) ------------------------------- */
  function b64e(str) {
    return btoa(unescape(encodeURIComponent(str)))
      .replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
  }
  function b64d(str) {
    str = str.replace(/-/g, '+').replace(/_/g, '/');
    while (str.length % 4) str += '=';
    return decodeURIComponent(escape(atob(str)));
  }
  function buildShareURL(state) {
    return location.href.split('#')[0] + '#s=' + b64e(JSON.stringify(state));
  }
  function readShared() {
    try {
      var m = /[#&]s=([^&]+)/.exec(location.hash);
      return m ? JSON.parse(b64d(m[1])) : null;
    } catch (e) { return null; }
  }
  function share(state) {
    var url = buildShareURL(state);
    try { history.replaceState(null, '', '#s=' + b64e(JSON.stringify(state))); } catch (e) {}
    copy(url).then(function () { toast('Shareable link copied'); });
  }

  /* ---- trust badges -------------------------------------------------------- */
  var BADGE_ICON = {
    shield: '<path d="M12 3l7 3v5c0 4-3 6.5-7 8-4-1.5-7-4-7-8V6z"/><path d="M9 12l2 2 4-4"/>',
    lock:   '<rect x="5" y="10" width="14" height="10" rx="2"/><path d="M8 10V7a4 4 0 0 1 8 0v3"/>',
    down:   '<path d="M12 4v11M7 11l5 5 5-5M5 20h14"/>',
    clock:  '<circle cx="12" cy="12" r="8"/><path d="M12 8v4l3 2"/>',
    dot:    '<circle cx="12" cy="12" r="3"/>'
  };
  function badges(container, opts) {
    opts = opts || {};
    var c = (typeof container === 'string') ? document.querySelector(container) : container;
    if (!c) return;
    var items = [];
    if (opts.runsLocal !== false) items.push(['shield', 'Runs locally']);
    if (opts.noUpload  !== false) items.push(['lock', 'No upload']);
    if (opts.exportable !== false) items.push(['down', 'Export available']);
    if (opts.updated) items.push(['clock', 'Updated ' + opts.updated]);
    (opts.extra || []).forEach(function (x) { items.push(['dot', x]); });
    c.innerHTML = '<div class="rmv-badges">' + items.map(function (it) {
      return '<span class="rmv-badge"><svg viewBox="0 0 24 24" aria-hidden="true">' +
        (BADGE_ICON[it[0]] || BADGE_ICON.dot) + '</svg>' + escapeHtml(it[1]) + '</span>';
    }).join('') + '</div>';
  }

  /* ---- saved workspaces (localStorage, namespaced per tool) ---------------- */
  function Workspaces(ns) { this.key = 'rmv-ws:' + ns; }
  Workspaces.prototype._all = function () {
    try { return JSON.parse(localStorage.getItem(this.key)) || []; } catch (e) { return []; }
  };
  Workspaces.prototype._set = function (a) {
    try { localStorage.setItem(this.key, JSON.stringify(a)); return true; } catch (e) { return false; }
  };
  Workspaces.prototype.list = function () { return this._all(); };
  Workspaces.prototype.get = function (id) {
    return this._all().filter(function (w) { return w.id === id; })[0];
  };
  Workspaces.prototype.save = function (name, data) {
    var a = this._all(), id = 'w' + Date.now();
    a.unshift({ id: id, name: name || ('Workspace ' + (a.length + 1)), ts: Date.now(), data: data });
    this._set(a); return id;
  };
  Workspaces.prototype.remove = function (id) {
    this._set(this._all().filter(function (w) { return w.id !== id; }));
  };

  /* compact workspaces UI: Save button + load dropdown + delete */
  function mountWorkspaces(container, cfg) {
    var c = (typeof container === 'string') ? document.querySelector(container) : container;
    if (!c) return null;
    var ws = new Workspaces(cfg.ns);
    function render() {
      var list = ws.list();
      c.innerHTML =
        '<div class="rmv-ws">' +
          '<button type="button" class="rmv-btn rmv-btn-primary" data-act="save">Save workspace</button>' +
          '<select class="rmv-ws-sel" data-act="sel" aria-label="Load saved workspace"' + (list.length ? '' : ' disabled') + '>' +
            '<option value="">' + (list.length ? 'Load saved… (' + list.length + ')' : 'No saved workspaces') + '</option>' +
            list.map(function (w) { return '<option value="' + w.id + '">' + escapeHtml(w.name) + '</option>'; }).join('') +
          '</select>' +
          '<button type="button" class="rmv-btn" data-act="del" title="Delete selected" disabled>Delete</button>' +
        '</div>';
    }
    c.addEventListener('click', function (e) {
      var b = e.target.closest('[data-act]'); if (!b) return;
      if (b.dataset.act === 'save') {
        var name = prompt('Name this workspace:', 'Workspace ' + (ws.list().length + 1));
        if (name === null) return;
        ws.save(name, cfg.getState()); render(); toast('Workspace saved');
      } else if (b.dataset.act === 'del') {
        var sel = c.querySelector('[data-act="sel"]');
        if (sel && sel.value) { ws.remove(sel.value); render(); toast('Workspace deleted'); }
      }
    });
    c.addEventListener('change', function (e) {
      var sel = e.target.closest('[data-act="sel"]'); if (!sel) return;
      var del = c.querySelector('[data-act="del"]'); if (del) del.disabled = !sel.value;
      if (sel.value) {
        var w = ws.get(sel.value);
        if (w) { cfg.applyState(w.data); toast('Loaded “' + w.name + '”'); }
      }
    });
    render();
    return ws;
  }

  window.RMVTool = {
    toast: toast, copy: copy, download: download, toCSV: toCSV,
    share: share, buildShareURL: buildShareURL, readShared: readShared,
    badges: badges, Workspaces: Workspaces, mountWorkspaces: mountWorkspaces,
    escapeHtml: escapeHtml
  };
})();
