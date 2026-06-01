# Contact backend — setup guide (Cloudflare Worker + Resend)

The `/contact` page on ruslanmv.com posts the form as JSON to a Cloudflare
Worker, which sends email through [Resend](https://resend.com). The Resend API
key lives **only** in the Worker as a secret — never in the browser, never in
GitHub (GitHub Pages is static, so the frontend can never safely hold secrets).

```
/contact page  →  fetch() POST JSON  →  Cloudflare Worker  →  Resend API  ┬→ notification → contact@ruslanmv.com
                   (no secrets)         (RESEND_API_KEY secret)           └→ auto-confirmation → the sender
```

On every valid submission the Worker sends **two** emails:
1. a **notification** to you (`contact@ruslanmv.com`), with `reply_to` set to the
   visitor — so you just hit Reply;
2. a **professional auto-confirmation** to the visitor, acknowledging receipt.

Until the Worker is deployed, the form **degrades gracefully**: it opens the
visitor's email client with the message pre-filled (mailto fallback), so the
page is usable even before you finish these steps.

---

## ⚠️ Where does the RESEND API key go? (Important)

**NOT in GitHub.** The key must never be in the repo or any GitHub setting that
ends up in the built site. It is a **Cloudflare Worker secret**. GitHub only
ever holds the public frontend; Cloudflare holds the secret and does the
sending. (A GitHub *Actions* secret would only make sense if a GitHub Action
did the sending — it does not here, so we don't use one.)

---

## 1. Domain status — already done ✅

Your domain `ruslanmv.com` is **verified** in Resend (the dashboard shows
"Domain verified: Your domain is ready to send emails"), with:

- **DKIM**: `TXT  resend._domainkey  p=MIGfMA0...` (signs your mail)
- **SPF** : `MX  send  feedback-smtp.us-east-1.amazonses.com` (prio 10) and
            `TXT send  v=spf1 include:amazonses.com ~all`
- **DMARC** (optional): `TXT _dmarc  v=DMARC1; p=none;`

Because the domain is verified, you can send **from `contact@ruslanmv.com`**.
Nothing more to do here.

## 2. Create a Resend API key

1. Resend dashboard → **API Keys** → **Create API Key**.
2. Name it e.g. `ruslanmv-contact-worker`, permission **Sending access**.
3. Copy the key (starts with `re_…`). You'll paste it as a Worker secret next —
   you won't be able to see it again.

## 3. Install Wrangler and create the Worker

```bash
cd contact-backend
npm install -g wrangler        # Cloudflare's CLI
wrangler login                 # opens the browser to authorize
```

## 4. Set the API key as a Worker secret (this is the key step)

```bash
wrangler secret put RESEND_API_KEY
#   → paste your re_... key when prompted, press Enter
```

This stores the key encrypted in Cloudflare, scoped to this Worker. It is
injected as `env.RESEND_API_KEY` at runtime and is never visible in code,
logs, or the browser.

`CONTACT_TO_EMAIL` and `CONTACT_FROM_EMAIL` are non-secret and live in
`wrangler.toml` under `[vars]`. (If you'd rather keep them secret too, remove
them from `[vars]` and run `wrangler secret put CONTACT_TO_EMAIL` etc.)

## 5. Deploy

```bash
wrangler deploy
```

Wrangler prints the Worker URL, e.g.
`https://ruslanmv-contact.<your-subdomain>.workers.dev`.

The endpoint the page calls is that URL **+ `/contact`**:
`https://ruslanmv-contact.<your-subdomain>.workers.dev/contact`

## 6. Point the page at the Worker

Edit `_pages/contact.md` and set:

```yaml
contact_endpoint: "https://ruslanmv-contact.<your-subdomain>.workers.dev/contact"
```

Rebuild the site (`make build`). The form now sends through the Worker, and the
mailto fallback automatically switches off.

## 7. Test it

```bash
curl -i -X POST \
  https://ruslanmv-contact.<your-subdomain>.workers.dev/contact \
  -H "Origin: https://ruslanmv.com" \
  -H "Content-Type: application/json" \
  -d '{"name":"Test User","email":"you@example.com","topic":"Consulting","message":"This is a test message that is at least twenty characters long."}'
```

Expect `HTTP/1.1 200` and `{"ok":true}`. You should receive the **notification**
at `contact@ruslanmv.com`, and the address you put in `"email"` should receive
the **auto-confirmation**.

---

## 8. Read what users sent — give `contact@ruslanmv.com` an inbox

Resend only **sends** email. A "verified domain" in Resend means Resend may send
*as* `contact@ruslanmv.com` — it does **not** create a mailbox you can open. To
**read** mail that arrives at `contact@ruslanmv.com`, the address must be hosted
by an email mailbox/forwarding provider.

### ✅ Current setup: Mail.ru hosts the inbox — nothing to add

`ruslanmv.com` already has receiving configured for **Mail.ru**:

```
ruslanmv.com   MX    emx.mail.ru   (priority 10)
ruslanmv.com   TXT   v=spf1 redirect=_spf.mail.ru
mailru._domainkey.ruslanmv.com  TXT  v=DKIM1; ...
```

So `contact@ruslanmv.com` is a **real mailbox at Mail.ru**. Contact-form
notifications are delivered there — just log in to Mail.ru and read them
(check Spam the first time). Reply normally; the notification's `reply_to`
is the visitor, so Reply goes straight to them.

> Do NOT add Zoho/Google/M365 or Cloudflare Email Routing — they would fight
> Mail.ru's root `MX emx.mail.ru` and could break receiving. Leave DNS as is.

Sending (Resend on `send.ruslanmv.com` + `resend._domainkey`) and receiving
(Mail.ru on the root MX) live on different records and don't conflict — keep
all of them.

### Recommended: add a DMARC record (deliverability)

The Worker sends *from* `contact@ruslanmv.com` via Resend, to the same domain
hosted at Mail.ru. Add DMARC so this is trusted and less likely to be filtered
(Cloudflare also nudges you to add it). Safe, monitor-only:

| Type | Name | Content |
| --- | --- | --- |
| `TXT` | `_dmarc` | `v=DMARC1; p=none;` |

Resend mail already aligns on DKIM (`resend._domainkey` matches the From
domain), so DMARC will pass.

---

### Alternative providers (only if you ever move OFF Mail.ru)

If you weren't already on Mail.ru, you'd host the mailbox elsewhere instead.
Documented here for reference only — **not needed in your current setup.**

#### Option A — make `contact@ruslanmv.com` a real mailbox

Use this if `contact@ruslanmv.com` is your actual personal inbox that you log
into directly (no Gmail involved). Host the mailbox with an email provider:

| Provider | Cost | Notes |
| --- | --- | --- |
| **Zoho Mail** | Free (1 user, custom domain) | Real `contact@ruslanmv.com` inbox, web + mobile |
| **Google Workspace** | ~$6/mo | Gmail-style inbox on *your* domain |
| **Microsoft 365** | ~$6/mo | Outlook-style |

Setup (Zoho example):
1. Sign up at Zoho Mail → add domain `ruslanmv.com` → create user `contact@ruslanmv.com`.
2. Zoho gives you **MX records** → add them in Cloudflare DNS for the **root** domain.
3. Log in at the provider and read/reply at `contact@ruslanmv.com` directly.

### Option B — forward `contact@ruslanmv.com` to another inbox (Email Routing)

Use this only if you'd rather read the mail somewhere else (e.g. Gmail). Free,
~1 minute, but it makes `contact@ruslanmv.com` a *forwarder*, not a standalone
mailbox:

1. Cloudflare dashboard → **ruslanmv.com** → **Email** → **Email Routing** → Enable.
2. **Destination addresses** → add your target inbox → confirm via Cloudflare's email.
3. **Routing rules** → **Custom address** → `contact@ruslanmv.com` → **Send to** → that inbox.

### Pick ONE — they conflict on MX
A mailbox host (Zoho/Google/M365) **and** Cloudflare Email Routing both want to
own the **root-domain MX** records. Choose one, not both. For a personal inbox
at `contact@ruslanmv.com`, use **Option A**.

### Does this break Resend sending? No.
- Resend's **sending** records — `resend._domainkey` (DKIM) and SPF/MX on the
  `send.ruslanmv.com` **subdomain** — are independent of the **root** MX your
  mailbox host adds for **receiving**. Keep all the Resend records and add the
  mailbox host's MX alongside them.

Either way you reply normally — the notification's `reply_to` is already the
visitor, so hitting Reply goes straight to them.

### Optional — also CC a second address
`CONTACT_TO_EMAIL` accepts a **comma-separated list**, so you can send the
notification to more than one inbox at once, e.g.
`contact@ruslanmv.com, you@example.com`. Set it in `wrangler.toml` `[vars]` and
`wrangler deploy`. (Don't run `wrangler secret put CONTACT_TO_EMAIL` while it's
also a `[vars]` entry — Cloudflare rejects a secret whose name collides with a
var. Use one or the other.)

### Why NOT use Resend's "receiving"/inbound API for this
Resend's `emails.receiving.get()` / inbound webhooks are a *different, advanced*
feature: you point your domain's **MX to Resend**, Resend POSTs a webhook to a
server you run, and you call the API to fetch the body as JSON. That's for
programmatically processing mail people send to your domain — not for getting a
readable inbox. For a contact form it's unnecessary complexity, it would replace
the MX you'd use for normal receiving, and it ends in JSON rather than an email
you can read. Two outbound sends (notification + confirmation) + Email Routing is
the correct, simple design.

---

## 9. Anti-spam: Cloudflare Turnstile (optional, recommended)

Turnstile is Cloudflare's invisible CAPTCHA. The form keeps working with no
Turnstile at all; it only activates when **both** keys below are set. Two keys:

| Key | Goes where | Public? |
| --- | --- | --- |
| **Site key** | `_pages/contact.md` → `turnstile_site_key` | public (safe to commit) |
| **Secret key** | Worker secret `TURNSTILE_SECRET_KEY` | secret (never in repo) |

### Step 1 — create the widget
Cloudflare dashboard → **Turnstile** → **Add widget**:
- Name: `ruslanmv-contact`
- Hostnames: `ruslanmv.com`, `www.ruslanmv.com`
- Mode: **Managed** (invisible for most users)
- Create → copy the **Site key** and **Secret key**.

### Step 2 — secret key into the Worker
```bash
cd contact-backend
wrangler secret put TURNSTILE_SECRET_KEY   # paste the SECRET key
wrangler deploy
```

### Step 3 — site key into the page
Edit `_pages/contact.md`:
```yaml
turnstile_site_key: "0x4AAAAAAA...your-site-key"
```
Rebuild and push. The widget appears and the Worker starts verifying tokens.

### How enforcement works
- Frontend: widget renders only when `turnstile_site_key` is non-empty; the
  solved token is sent as `turnstileToken` and reset after each submit.
- Worker: when `TURNSTILE_SECRET_KEY` exists, it verifies the token via
  Cloudflare `siteverify` (missing token → 400, failed → 403) **before** sending
  email. Honeypot is checked first, so bots are dropped even earlier.
- Use the **Site** key on the page and the **Secret** key in the Worker — never
  swap them (Site key starts `0x4...`).

---

## Future: AI-written replies

The auto-confirmation body is produced by `buildConfirmationEmail()` in
`worker.js`. To later have an AI assistant draft the reply, make that function
`async` and call your AI model (e.g. the Anthropic API or a second Worker),
returning `{ subject, html, text }`. The send pipeline stays identical — only
the body generation changes. Today we ship the deterministic auto-confirmation;
the AI layer drops in without touching anything else.

## Security recap

- `RESEND_API_KEY` is a Cloudflare Worker secret — never in GitHub or the browser.
- CORS locked to `ruslanmv.com`, `www.ruslanmv.com`, localhost (`ALLOWED_ORIGINS`).
- Server-side validation: required fields, email format, message length 20–4000.
- Honeypot (`company`) silently drops bots.
- All visitor input is HTML-escaped before it enters an email body.
- Consider [Cloudflare Turnstile](https://developers.cloudflare.com/turnstile/)
  and/or Worker rate limiting if the endpoint sees abuse.

## Local development

`wrangler dev` runs the Worker locally on `http://127.0.0.1:8787`. Temporarily
set `contact_endpoint` to `http://127.0.0.1:8787/contact`; `localhost:4000` is
already in the CORS allowlist for `jekyll serve`.
