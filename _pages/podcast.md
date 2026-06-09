---
title: "Podcast"
layout: single
permalink: /podcast/
sitemap: true
canonical_url: https://ruslanmv.com/podcast/
description: "The Ruslan Magana Vsevolodovna Podcast — audio essays on AI, software, learning, research, and systems thinking. Subscribe by RSS, Apple Podcasts, Spotify, and more."
excerpt: "Audio essays on AI, software, learning, research, and systems thinking."
author_profile: false
classes: wide
header:
  og_image: /assets/images/podcast-cover.png
---

<style>
.pod{--brand:#0a7d4f;--brand-2:#2bbf86;--brand-deep:#0b5a3f;--ink:#1c2320;--muted:#6c726c;--faint:#98a09a;--line:#e7e6df;--card:#fff;--hero:#08120d;--hero-text:#cfe9dc;max-width:880px;margin:0 auto}
.pod *{box-sizing:border-box}
/* hero */
.pod__hero{position:relative;overflow:hidden;border-radius:24px;padding:2rem 2.2rem;display:flex;gap:1.8rem;align-items:center;flex-wrap:wrap;background:linear-gradient(180deg,#f7f8f5 0%,#ffffff 100%);border:1px solid var(--line);box-shadow:0 10px 30px rgba(8,18,13,.05)}
.pod__hero:before{content:"";position:absolute;inset:0;background:radial-gradient(55% 80% at 88% 8%,rgba(10,125,79,.07),transparent 70%);pointer-events:none}
.pod__art{position:relative;width:140px;height:140px;border-radius:24px;flex:none;object-fit:cover;object-position:center 18%;background:linear-gradient(155deg,#e9f5ef 0%,#ffffff 75%);border:1px solid var(--line);box-shadow:0 14px 34px rgba(8,18,13,.10)}
.pod__hgroup{position:relative;flex:1;min-width:250px}
.pod__eyebrow{font-family:'Montserrat',sans-serif;font-weight:700;letter-spacing:.18em;text-transform:uppercase;font-size:.72rem;color:var(--brand);margin:0 0 .5rem}
.pod__title{font-family:'Montserrat',sans-serif;font-weight:800;letter-spacing:-.02em;line-height:1.05;font-size:clamp(1.7rem,4.2vw,2.5rem);color:var(--ink);margin:0}
.pod__sub{color:var(--muted);font-size:1.02rem;line-height:1.55;margin:.7rem 0 1.2rem;max-width:32rem}
.pod__cta{display:flex;gap:.6rem;flex-wrap:wrap}
.pod__btn{display:inline-flex;align-items:center;gap:.5rem;font-family:'Montserrat',sans-serif;font-weight:700;font-size:.92rem;text-decoration:none;padding:.6rem 1.05rem;border-radius:12px;transition:transform .16s,filter .16s,border-color .16s,color .16s}
.pod__btn svg{width:16px;height:16px}
.pod__btn--solid{color:#fff;background:linear-gradient(135deg,#2bbf86,#0a7d4f);box-shadow:0 8px 20px rgba(10,125,79,.22)}
.pod__btn--ghost{color:var(--ink);background:#fff;border:1px solid var(--line)}
.pod__btn--ghost:hover{border-color:var(--brand);color:var(--brand)}
.pod__btn:hover{transform:translateY(-2px)}
.pod__btn--solid:hover{filter:brightness(1.04)}
/* section heads */
.pod__h{font-family:'Montserrat',sans-serif;font-weight:800;letter-spacing:-.01em;color:var(--ink);font-size:1.25rem;margin:2.3rem 0 .4rem}
.pod__p{color:var(--muted);line-height:1.6;margin:0 0 1rem}
/* RSS feed card */
.pod__feed{display:flex;align-items:center;gap:.9rem;flex-wrap:wrap;padding:1.05rem 1.15rem;border:1px solid var(--line);border-radius:16px;background:var(--card);box-shadow:0 12px 34px rgba(8,18,13,.05)}
.pod__feed-ic{flex:none;width:38px;height:38px;border-radius:11px;display:grid;place-items:center;background:#eef6f1;color:var(--brand)}
.pod__feed-ic svg{width:20px;height:20px}
.pod__url{flex:1;min-width:200px;font-family:Monaco,Consolas,monospace;font-size:.9rem;color:var(--ink);background:#f6f6f3;border:1px solid var(--line);border-radius:9px;padding:.5rem .7rem;word-break:break-all}
.pod__feed-act{display:flex;gap:.5rem}
.pod__mini{cursor:pointer;font-family:'Montserrat',sans-serif;font-weight:700;font-size:.85rem;text-decoration:none;border-radius:10px;padding:.5rem .85rem;border:1px solid var(--line);color:var(--brand-deep);background:#fff;transition:border-color .16s,color .16s,background .16s}
.pod__mini--solid{color:#fff;border-color:transparent;background:linear-gradient(135deg,var(--brand-2),var(--brand))}
.pod__mini:hover{border-color:var(--brand);color:var(--brand)}
.pod__mini--solid:hover{color:#fff;filter:brightness(1.05)}
/* platform grid */
.pod__grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:.85rem;margin:.2rem 0 1.4rem}
.pod__pcard{display:flex;align-items:center;gap:.8rem;padding:1rem 1.1rem;border:1px solid var(--line);border-radius:16px;background:var(--card);text-decoration:none;transition:transform .18s,box-shadow .18s,border-color .18s}
.pod__pcard:hover{transform:translateY(-3px);box-shadow:0 16px 38px rgba(8,18,13,.08);border-color:rgba(10,125,79,.4)}
.pod__pic{flex:none;width:36px;height:36px;border-radius:10px;display:grid;place-items:center;background:#eef6f1;color:var(--brand);transition:background .18s,color .18s}
.pod__pcard:hover .pod__pic{background:var(--brand);color:#fff}
.pod__pic svg{width:20px;height:20px}
.pod__pname{font-family:'Montserrat',sans-serif;font-weight:700;color:var(--ink);font-size:.95rem;line-height:1.15}
.pod__psub{display:block;color:var(--muted);font-size:.78rem;font-weight:400;margin-top:.12rem}
.pod__note{color:var(--faint);font-size:.9rem;line-height:1.55;border-left:3px solid var(--line);padding:.1rem 0 .1rem 1rem;margin:.4rem 0 0}
/* the hero renders its own H1; hide the theme's duplicate page title */
.page__title{display:none}
/* desktop: give the persona portrait more presence as the page's brand anchor */
@media (min-width:680px){
  .pod__hero{gap:2.2rem;padding:2.3rem 2.5rem;align-items:center}
  .pod__art{width:178px;height:178px;border-radius:28px;object-position:center 16%}
  .pod__title{font-size:clamp(2rem,4.2vw,2.7rem)}
}
</style>

<div class="pod" markdown="0">
  <div class="pod__hero">
    <img class="pod__art" src="/assets/images/ruslan.png" alt="Ruslan Magana Vsevolodovna">
    <div class="pod__hgroup">
      <p class="pod__eyebrow">Audio</p>
      <h1 class="pod__title">Podcast</h1>
      <p class="pod__sub"><strong style="color:#1c2320;font-weight:700">Audio essays by Ruslan Magana Vsevolodovna.</strong> Every audio essay is also released as a podcast episode — subscribe once and new essays arrive automatically in your app.</p>
      <div class="pod__cta">
        <a class="pod__btn pod__btn--solid" href="#rss">
          <svg viewBox="0 0 24 24" fill="currentColor"><circle cx="6.2" cy="17.8" r="2.2"/><path d="M4 10.2a9.8 9.8 0 0 1 9.8 9.8h-3A6.8 6.8 0 0 0 4 13.2z"/><path d="M4 4.2A15.8 15.8 0 0 1 19.8 20h-3A12.8 12.8 0 0 0 4 7.2z"/></svg>
          Subscribe by RSS
        </a>
        <a class="pod__btn pod__btn--ghost" href="https://podcasts.apple.com/search?term=Ruslan%20Magana" rel="noopener">Apple Podcasts</a>
        <a class="pod__btn pod__btn--ghost" href="https://open.spotify.com/search/Ruslan%20Magana/podcasts" rel="noopener">Spotify</a>
      </div>
    </div>
  </div>

  <h2 class="pod__h" id="rss">RSS Feed</h2>
  <p class="pod__p">Use this feed in any podcast app that supports <em>“Add by URL”</em> — Apple Podcasts, Pocket Casts, Overcast, AntennaPod, Podcast Addict, Castro, and more.</p>
  <div class="pod__feed">
    <span class="pod__feed-ic"><svg viewBox="0 0 24 24" fill="currentColor"><circle cx="6.2" cy="17.8" r="2.2"/><path d="M4 10.2a9.8 9.8 0 0 1 9.8 9.8h-3A6.8 6.8 0 0 0 4 13.2z"/><path d="M4 4.2A15.8 15.8 0 0 1 19.8 20h-3A12.8 12.8 0 0 0 4 7.2z"/></svg></span>
    <code class="pod__url">https://ruslanmv.com/podcast.xml</code>
    <span class="pod__feed-act">
      <button type="button" class="pod__mini pod__mini--solid" onclick="navigator.clipboard.writeText('https://ruslanmv.com/podcast.xml').then(()=>{this.textContent='Copied!';setTimeout(()=>this.textContent='Copy RSS',1500)})">Copy RSS</button>
      <a class="pod__mini" href="/podcast.xml">Open feed</a>
    </span>
  </div>

  <h2 class="pod__h">Listen in your app</h2>
  <div class="pod__grid">
    <a class="pod__pcard" href="https://podcasts.apple.com/search?term=Ruslan%20Magana" rel="noopener">
      <span class="pod__pic"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="12" cy="12" r="9"/><circle cx="12" cy="10" r="2.4"/><path d="M8.5 18c.6-2.4 1.4-3.6 3.5-3.6s2.9 1.2 3.5 3.6"/></svg></span>
      <span class="pod__pname">Apple Podcasts<span class="pod__psub">Search & subscribe</span></span>
    </a>
    <a class="pod__pcard" href="https://open.spotify.com/search/Ruslan%20Magana/podcasts" rel="noopener">
      <span class="pod__pic"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="12" cy="12" r="9"/><path d="M7.5 9.5c3-.9 6.5-.6 9 1M8 13c2.4-.7 5-.4 7 .8M8.5 16c1.8-.5 3.7-.3 5.2.6"/></svg></span>
      <span class="pod__pname">Spotify<span class="pod__psub">Follow the show</span></span>
    </a>
    <a class="pod__pcard" href="https://music.youtube.com/search?q=Ruslan+Magana" rel="noopener">
      <span class="pod__pic"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="12" cy="12" r="9"/><path d="M10 9l5 3-5 3z" fill="currentColor" stroke="none"/></svg></span>
      <span class="pod__pname">YouTube Music<span class="pod__psub">Listen on YouTube</span></span>
    </a>
    <a class="pod__pcard" href="https://podcastindex.org/search?q=Ruslan+Magana" rel="noopener">
      <span class="pod__pic"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="2.2" fill="currentColor" stroke="none"/><path d="M12 3v3M12 18v3M3 12h3M18 12h3"/></svg></span>
      <span class="pod__pname">Podcast Index<span class="pod__psub">Open directory</span></span>
    </a>
    <a class="pod__pcard" href="https://www.listennotes.com/search/?q=Ruslan%20Magana" rel="noopener">
      <span class="pod__pic"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="11" cy="11" r="6.5"/><path d="M16 16l4 4"/></svg></span>
      <span class="pod__pname">Listen Notes<span class="pod__psub">Search the web</span></span>
    </a>
    <a class="pod__pcard" href="/essays/">
      <span class="pod__pic"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M5 4h11l3 3v13H5z"/><path d="M9 11h7M9 15h5"/></svg></span>
      <span class="pod__pname">Browse episodes<span class="pod__psub">Read & listen on site</span></span>
    </a>
  </div>

  <p class="pod__note">New episodes may take a little while to appear on the big platforms after their first submission. The RSS feed is always available immediately.</p>
</div>
