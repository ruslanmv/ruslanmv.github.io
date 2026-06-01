/* Contact form — validates, POSTs JSON to the serverless endpoint (Cloudflare
   Worker → Resend). No API keys here. Falls back to mailto if the endpoint is
   not configured yet. */
(function () {
  function ready(fn){ if (document.readyState !== "loading") fn(); else document.addEventListener("DOMContentLoaded", fn); }
  ready(function () {
    var wrap = document.querySelector(".ct");
    var form = document.getElementById("ct-form");
    if (!form) return;
    var endpoint = (wrap && wrap.getAttribute("data-endpoint")) || "";
    var configured = endpoint && endpoint.indexOf("YOUR-SUBDOMAIN") === -1 && endpoint.indexOf("YOUR-WORKER") === -1;
    var turnstileKey = (wrap && wrap.getAttribute("data-turnstile-key")) || "";
    var turnstileWidgetId = null;

    // Render the Turnstile widget once its script has loaded (only if a key is set).
    function initTurnstile() {
      if (!turnstileKey) return;
      var holder = document.getElementById("ct-turnstile");
      if (!holder || !window.turnstile) return;
      turnstileWidgetId = window.turnstile.render(holder, {
        sitekey: turnstileKey,
        theme: "light",
        appearance: "always" // make sure visitors see the widget, never invisible
      });
    }
    if (turnstileKey) {
      if (window.turnstile) initTurnstile();
      else { window.onloadTurnstileCallback = initTurnstile;
             var t = setInterval(function () { if (window.turnstile) { clearInterval(t); initTurnstile(); } }, 200); }
    }
    function turnstileToken() {
      if (!turnstileKey || !window.turnstile) return "";
      try { return window.turnstile.getResponse(turnstileWidgetId) || ""; } catch (e) { return ""; }
    }
    function resetTurnstile() {
      if (turnstileKey && window.turnstile) { try { window.turnstile.reset(turnstileWidgetId); } catch (e) {} }
    }

    var statusEl = document.getElementById("ct-status");
    var btn = form.querySelector(".ct-submit");
    var btnLabel = form.querySelector(".ct-submit__label");
    var defaultLabel = btnLabel ? btnLabel.textContent : "Send Message";

    function setStatus(msg, kind) {
      statusEl.textContent = msg;
      statusEl.className = "ct-status is-show is-" + kind;
    }
    function clearStatus(){ statusEl.textContent = ""; statusEl.className = "ct-status"; }
    function fieldOf(input){ return input.closest(".ct-field"); }
    function markInvalid(input){ var f = fieldOf(input); if (f) f.classList.add("is-invalid"); }
    function clearInvalid(){ form.querySelectorAll(".ct-field.is-invalid").forEach(function(f){ f.classList.remove("is-invalid"); }); }

    var EMAIL_RE = /^[^@\s]+@[^@\s]+\.[^@\s]+$/;

    function validate(d) {
      clearInvalid();
      var first = null;
      function bad(id){ var el = document.getElementById(id); markInvalid(el); if (!first) first = el; }
      if (!d.name)  bad("ct-name");
      if (!d.email || !EMAIL_RE.test(d.email)) bad("ct-email");
      if (!d.topic) bad("ct-topic");
      if (!d.message || d.message.length < 20) bad("ct-message");
      var consent = document.getElementById("ct-consent");
      if (consent && !consent.checked) { if (!first) first = consent; }
      if (first) { first.focus(); }
      return !first;
    }

    form.addEventListener("submit", function (e) {
      e.preventDefault();
      clearStatus();

      var data = {
        name:    form.name.value.trim(),
        email:   form.email.value.trim(),
        topic:   form.topic.value,
        message: form.message.value.trim(),
        company: form.company ? form.company.value.trim() : "", // honeypot
        turnstileToken: turnstileToken()                        // Cloudflare Turnstile
      };

      if (!validate(data)) {
        setStatus("Please complete the highlighted fields (message needs at least 20 characters) and accept the privacy policy.", "err");
        return;
      }

      // If Turnstile is enabled but not yet solved, ask the visitor to complete it.
      if (turnstileKey && !data.turnstileToken) {
        var holder = document.getElementById("ct-turnstile");
        if (holder) holder.scrollIntoView({ behavior: "smooth", block: "center" });
        setStatus("Please tick the \"Verify you're human\" box above before sending.", "err");
        return;
      }

      // Honeypot filled → pretend success, send nothing.
      if (data.company) { form.reset(); setStatus("Thank you. Your message has been sent.", "ok"); return; }

      // No backend configured yet → graceful mailto fallback.
      if (!configured) {
        var body = "Topic: " + data.topic + "\n\n" + data.message + "\n\n— " + data.name + " (" + data.email + ")";
        var href = "mailto:contact@ruslanmv.com?subject=" + encodeURIComponent("[ruslanmv.com] " + data.topic + " — " + data.name) +
                   "&body=" + encodeURIComponent(body);
        setStatus("Opening your email app to send the message…", "info");
        window.location.href = href;
        return;
      }

      // Send to the serverless endpoint. Loading state: disable + spinner + label swap.
      btn.disabled = true;
      btn.classList.add("is-loading");
      if (btnLabel) btnLabel.textContent = "Sending…";
      clearStatus();

      fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data)
      })
      .then(function (r) { return r.json().then(function (j){ return { ok: r.ok, body: j }; }); })
      .then(function (res) {
        if (res.ok && res.body && res.body.ok) {
          form.reset();
          clearStatus();
          openModal();
        } else {
          var err = (res.body && res.body.error) || "";
          // Friendly messaging for the most common cases.
          if (/Verification/i.test(err)) {
            var holder = document.getElementById("ct-turnstile");
            if (holder) holder.scrollIntoView({ behavior: "smooth", block: "center" });
            setStatus("Please tick the \"Verify you're human\" box above and try again.", "err");
          } else {
            setStatus("Sorry, the message could not be sent (" + (err || "unknown error") + "). Please email contact@ruslanmv.com directly.", "err");
          }
        }
      })
      .catch(function () {
        setStatus("Network error — please email contact@ruslanmv.com directly.", "err");
      })
      .finally(function () {
        btn.disabled = false;
        btn.classList.remove("is-loading");
        if (btnLabel) btnLabel.textContent = defaultLabel;
        resetTurnstile(); // each submission needs a fresh token
      });
    });

    /* ---- Success modal: open / close / focus trap / Send-another reset ---- */
    var modal = document.getElementById("ct-modal");
    var banner = document.getElementById("ct-banner");
    var lastFocus = null;

    function focusables() {
      if (!modal) return [];
      return Array.prototype.slice.call(modal.querySelectorAll(
        'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
      )).filter(function (el) { return !el.disabled && el.offsetParent !== null; });
    }
    function openModal() {
      if (!modal) return;
      lastFocus = document.activeElement;
      modal.hidden = false;
      modal.setAttribute("aria-hidden", "false");
      document.body.classList.add("ct-modal-open");
      var f = focusables();
      if (f.length) f[0].focus();
    }
    function closeModal(showBanner) {
      if (!modal) return;
      modal.setAttribute("aria-hidden", "true");
      modal.hidden = true;
      document.body.classList.remove("ct-modal-open");
      if (showBanner !== false && banner) banner.hidden = false;
      if (lastFocus && typeof lastFocus.focus === "function") lastFocus.focus();
    }
    function sendAnother() {
      // Close modal, hide banner, focus the Name field for a fresh message.
      if (banner) banner.hidden = true;
      closeModal(false);
      var name = document.getElementById("ct-name");
      if (name) { try { name.scrollIntoView({ behavior: "smooth", block: "center" }); } catch (e) {} name.focus(); }
    }

    if (modal) {
      modal.addEventListener("click", function (e) {
        var t = e.target;
        if (t.closest && t.closest("[data-ct-action='another']")) { sendAnother(); return; }
        if (t.closest && t.closest("[data-ct-close]")) { closeModal(true); }
      });
      document.addEventListener("keydown", function (e) {
        if (modal.getAttribute("aria-hidden") === "true") return;
        if (e.key === "Escape") { e.preventDefault(); closeModal(true); return; }
        if (e.key === "Tab") {
          var f = focusables(); if (!f.length) return;
          var first = f[0], last = f[f.length - 1];
          if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
          else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
        }
      });
    }
    if (banner) {
      banner.addEventListener("click", function (e) {
        if (e.target.closest("[data-banner-close]")) banner.hidden = true;
      });
    }
  });
})();
