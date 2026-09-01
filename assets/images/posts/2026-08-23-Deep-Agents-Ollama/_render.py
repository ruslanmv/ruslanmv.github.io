"""Render _banner.svg and _teaser.svg to the JPEGs the post frontmatter points at.

Needs playwright and a Chromium. Run from the repo root:

    python assets/images/posts/2026-08-23-Deep-Agents-Ollama/_render.py

Edit the SVG and re-run; never retouch the JPEG.
"""
import pathlib
import re

from playwright.sync_api import sync_playwright

D = pathlib.Path("assets/images/posts/2026-08-23-Deep-Agents-Ollama").resolve()

JOBS = [
    # (source svg, output jpg, device scale factor)
    ("_banner.svg", "deep-agents-ollama.jpg", 1.5),
    ("_teaser.svg", "deep-agents-ollama-teaser.jpg", 2.0),
]

with sync_playwright() as p:
    b = p.chromium.launch(executable_path="/opt/pw-browsers/chromium")
    for src, out, dsf in JOBS:
        svg = D / src
        if not svg.exists():
            print(f"  skip {src} (missing)")
            continue
        text = svg.read_text()
        m = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', text)
        W, H = (int(float(m.group(1))), int(float(m.group(2))))
        ctx = b.new_context(viewport={"width": W, "height": H},
                            device_scale_factor=dsf)
        pg = ctx.new_page()
        # Body margin 0 so the SVG fills the shot exactly.
        pg.set_content(
            f'<style>html,body{{margin:0;padding:0;background:#0a0c18}}'
            f'svg{{display:block}}</style>{text}',
            wait_until="load",
        )
        pg.wait_for_timeout(600)  # let webfonts settle
        dest = D / out
        pg.screenshot(path=str(dest), type="jpeg", quality=94,
                      clip={"x": 0, "y": 0, "width": W, "height": H})
        kb = dest.stat().st_size / 1024
        print(f"  {out:44} {int(W*dsf)}x{int(H*dsf)}  {kb:.0f} KB")
        ctx.close()
    b.close()
