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
      turnstileWidgetId = window.turnstile.render(holder, { sitekey: turnstileKey });
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
        setStatus("Please complete the verification challenge below the form.", "err");
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

      // Send to the serverless endpoint.
      btn.disabled = true;
      if (btnLabel) btnLabel.textContent = "Sending message…";
      setStatus("Sending message…", "info");

      fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data)
      })
      .then(function (r) { return r.json().then(function (j){ return { ok: r.ok, body: j }; }); })
      .then(function (res) {
        if (res.ok && res.body && res.body.ok) {
          form.reset();
          setStatus("Thank you. Your message has been sent — I'll reply by email, usually within 1–2 business days.", "ok");
        } else {
          var msg = (res.body && res.body.error) ? res.body.error : "Something went wrong.";
          setStatus("Sorry, the message could not be sent (" + msg + "). Please email contact@ruslanmv.com directly.", "err");
        }
      })
      .catch(function () {
        setStatus("Network error — please email contact@ruslanmv.com directly.", "err");
      })
      .finally(function () {
        btn.disabled = false;
        if (btnLabel) btnLabel.textContent = defaultLabel;
        resetTurnstile(); // each submission needs a fresh token
      });
    });
  });
})();
