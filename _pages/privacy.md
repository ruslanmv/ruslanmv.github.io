---
title: "Privacy Policy"
permalink: /privacy/
layout: privacy
sitemap: true
canonical_url: https://ruslanmv.com/privacy/
excerpt: "How ruslanmv.com handles your information when you visit the site, contact me, or read my work."
header:
  og_image: /assets/images/essays-hero.svg

hero_kicker: "PRIVACY"
hero_title: "Privacy Policy"
hero_sub: "How ruslanmv.com handles your information when you visit, read, or contact me."
updated: "1 June 2026"
---

This page explains what data this site collects, why, and what your rights are. It applies to everything served from **ruslanmv.com** (and `www.ruslanmv.com`).

## At a glance

- I do not sell your data. I do not run advertising trackers beyond the analytics described below.
- The contact form is processed by a serverless function I operate; submissions are emailed to me and stored only as ordinary email.
- Standard infrastructure providers (Cloudflare, GitHub Pages, Google Analytics, Resend) see request metadata as part of normal operation.

---

## 1. Who I am

This site is operated by Ruslan Magana Vsevolodovna (the "site owner"), based in Genoa, Italy. You can contact me at **[contact@ruslanmv.com](mailto:contact@ruslanmv.com)**.

## 2. What I collect

**a. Contact-form submissions.** When you send a message via the [contact form](/contact/), I collect the fields you provide:

- Your name
- Your email address
- The topic you select
- The message you write

I do **not** collect anything else from the form. The submission is delivered to me by email and reaches your verification proof through Cloudflare Turnstile (see §5).

**b. Server and request data.** Your browser, like every browser, sends a request that includes your IP address, user-agent string, referring page, and the URL you asked for. This is necessary for the page to load. Cloudflare (the CDN in front of the site) and GitHub Pages (the host) process this metadata to deliver the page and protect the site from abuse.

**c. Analytics.** The site uses **Google Analytics** to understand which articles are read and where readers come from. Google Analytics may set cookies and collect a hashed identifier, page paths, approximate location (city level), device type, and engagement metrics. I have enabled IP-address anonymisation where supported.

**d. Comments (where present).** Some older posts include comments powered by **Disqus**. If you leave a comment, you are using Disqus directly; their privacy policy applies.

I do **not** collect: payment information, sensitive personal data, biometric data, or data from children under 16 to my knowledge.

## 3. Why I process this data

| Purpose | Data used | Legal basis (GDPR) |
| --- | --- | --- |
| Reply to your message | contact form fields | your explicit request / consent |
| Run and protect the site | request metadata, IP, user-agent | legitimate interest |
| Understand audience and improve content | Google Analytics events | legitimate interest (anonymised) |
| Stop spam and bots on the contact form | Cloudflare Turnstile challenge data | legitimate interest |

## 4. Who sees your data (sub-processors)

The site is built with off-the-shelf infrastructure. Each sub-processor only sees what it needs to:

- **GitHub Pages** — serves the static site files. Sees: request metadata.
- **Cloudflare** — CDN, DNS, and the Worker that handles contact submissions. Sees: request metadata, contact-form payload (only at the moment of submission, never stored beyond the request).
- **Resend** — the email-sending API used by the contact Worker. Sees: the contents of the notification and confirmation emails it sends, transient delivery logs.
- **Mail.ru** — the mailbox provider for `@ruslanmv.com`. Sees: incoming notification emails.
- **Google Analytics** — sees aggregated, mostly anonymised browsing events.
- **Disqus** — sees comments you post directly on a comment thread, if any.

None of these providers receive your data for advertising profiling on my behalf.

## 5. Spam protection (Cloudflare Turnstile)

The contact form is protected by **Cloudflare Turnstile**, an alternative to traditional CAPTCHAs that verifies you are a real person without showing image puzzles. Turnstile may briefly evaluate signals such as your IP, browser characteristics, and behaviour on the page to decide whether to issue a verification token. It does **not** track you across other websites and does not show ads.

## 6. Cookies

This site uses a small number of cookies:

- **Google Analytics** sets `_ga`, `_gid`, and related cookies to count visits and sessions.
- **Cloudflare** may set a security cookie (`__cf_bm`) to distinguish humans from bots.
- **Disqus** may set its own cookies on pages where its comment widget is loaded.

I do not use cookies for advertising. You can clear or block these cookies in your browser at any time without breaking the site.

## 7. How long data is kept

- **Contact form submissions** live in my email inbox until I delete the thread, like any normal email.
- **Resend delivery logs** are retained for a short period (days) by Resend for diagnostic purposes.
- **Google Analytics** data retention is configured to the shortest practical period (currently 14 months) and is anonymised.
- **Server logs** at GitHub Pages and Cloudflare follow each provider's standard short-term retention.

## 8. Your rights

If you are in the EU/EEA, the UK, or another jurisdiction with similar laws, you have the right to:

- **access** the data I hold about you,
- **correct** it if it is wrong,
- **delete** it ("right to be forgotten"),
- **object** to processing based on legitimate interest,
- **port** your data to another service,
- **withdraw consent** at any time (where consent is the legal basis),
- **complain** to a supervisory authority (in Italy, the [Garante per la protezione dei dati personali](https://www.garanteprivacy.it/)).

To exercise any of these rights, email me at **[contact@ruslanmv.com](mailto:contact@ruslanmv.com)** with the request. I will respond within 30 days.

## 9. Security

Communications are encrypted in transit (HTTPS via Cloudflare). The contact-form Worker stores the Resend API key as an encrypted Cloudflare Worker secret — never in the repository or the browser. Visitor input is validated and HTML-escaped before being placed into emails.

## 10. International transfers

Some sub-processors (Cloudflare, Google, Resend) are based in the United States and may process data there. Where required, these providers rely on the EU Commission's Standard Contractual Clauses or equivalent safeguards for international transfers.

## 11. Changes to this policy

If this policy changes materially I will update the date below and, where appropriate, mention the change on the site. Minor wording or housekeeping updates may be made without notice.

## 12. Contact

Questions about this policy, or about how your data is handled?

**Email:** [contact@ruslanmv.com](mailto:contact@ruslanmv.com)

---

*Last updated: {{ page.updated }}.*
