---
title: "Contact"
permalink: /contact/
layout: contact
sitemap: true
canonical_url: https://ruslanmv.com/contact/
excerpt: "For collaborations, consulting, research, speaking, or applied AI systems work — send a message."
header:
  og_image: /assets/images/essays-hero.svg

# Hero
hero_kicker: "CONTACT"
hero_title: "Contact"
hero_sub: "For collaborations, consulting, research, speaking, or AI systems work, I'd love to hear from you."

# The serverless endpoint that receives the form POST (Cloudflare Worker).
# Live Worker on the cloud-data.workers.dev subdomain.
contact_endpoint: "https://ruslanmv-contact.cloud-data.workers.dev/contact"

# Cloudflare Turnstile (anti-spam). Paste the SITE key from the Turnstile
# dashboard here to turn it on. Leave blank to disable (form works either way).
# The matching SECRET key goes in the Worker: wrangler secret put TURNSTILE_SECRET_KEY
turnstile_site_key: "0x4AAAAAADclldrQolt2BJdP"
---

{% include contact-form.html %}
