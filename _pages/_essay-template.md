---
# =============================================================================
#  PURE ESSAY TEMPLATE  —  copy-paste starter
#  Appears on /essays, NOT in /blog.
#
#  This file is listed in `exclude:` in _config.yml, so the template itself is
#  never built and never appears anywhere. To use it:
#    1. Copy to a new name, e.g.  _pages/my-new-essay.md   (any name NOT starting
#       with "_" and NOT in the exclude list — a normal name just works).
#    2. Set a real `permalink`, fill in the fields, write your prose.
#    3. Build. It auto-appears at the top of /essays.
#
#  Want it ALSO in /blog? Put the copy in _posts/ with a dated filename instead
#  (YYYY-MM-DD-title.md). Same fields work.
# =============================================================================

title: "Your Essay Title"
permalink: /your-essay-slug/           # clean URL, not under /blog
layout: essay                          # premium editorial essay layout
sitemap: true
canonical_url: https://ruslanmv.com/your-essay-slug/

# ---- editorial hero (rendered by _layouts/essay.html) ----
eyebrow: "Topic · Essay"
headline: "Your Essay Title"
subtitle: "An optional second line under the title"
thesis: "One sentence that tells the reader why this essay matters."
author_name: "Ruslan Magana Vsevolodovna"
author_role: "Machine Learning Engineer · Data Scientist · Physicist"
read_time: "10 min read"
author_links:
  - label: "LinkedIn"
    url: "https://www.linkedin.com/in/ruslanmv/"
  - label: "GitHub"
    url: "https://github.com/ruslanmv"
  - label: "Email"
    url: "mailto:contact@ruslanmv.com"

# ---- optional "In this essay" summary card ----
summary: "A short paragraph summarising the essay for the reader before they start."

# ---- on-this-page table of contents (desktop) ----
toc: true
toc_items:
  - { id: "section-one",   title: "Section One" }
  - { id: "section-two",   title: "Section Two" }

# ---- /essays archive listing (this is what surfaces it) ----
essay: true
essay_date: "2026-01-01"               # controls order + the date shown on the card
card_image: /assets/images/essays/your-essay.svg   # optional; falls back to header image
card_excerpt: "One-line thesis shown on the /essays card."
---

<h2 id="section-one">Section One</h2>

<p>Open on a tension the reader already feels. Develop one central idea in calm,
confident prose. Earn your links — place each where the reader is naturally
curious — and give the piece one quotable sentence.</p>

<h2 id="section-two">Section Two</h2>

<p>Close by pointing forward, and link back into your wider body of work so the
essays reinforce each other.</p>
