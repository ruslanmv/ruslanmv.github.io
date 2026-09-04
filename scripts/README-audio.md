# Essay audio narration (▷ Audio)

Adds a subtle inline **▷ Audio** control to essays and posts. Narration is
generated once with [Kokoro ONNX](https://github.com/thewh1teagle/kokoro-onnx),
stored as an MP3 in Cloudflare R2, and pointed to from
`_data/audio_manifest.json`. The browser only streams the MP3 after a click
(`preload="none"`).

---

## 🟢 Quick start for bloggers (the simple version)

You only edit **one file** — your post — and add **three things**. You do *not*
run any commands; a robot (GitHub Actions) makes the audio for you after you
push.

### Step 1 — turn audio on

At the very top of your post, inside the `---` block (the "front matter"), add
two lines:

```yaml
audio: true
audio_slug: my-post          # a short nickname, lowercase-with-dashes, unique
```

### Step 2 — mark where the reading starts and stops

In the body of your post, put two invisible markers around the text you want
read out loud. They are HTML comments, so **readers never see them**:

```markdown
<!-- audio:start -->

Everything between these two markers is read aloud.
Put them around your real article text.

<!-- audio:end -->
```

Anything **outside** the markers (intro notes, image captions, "further
reading", code, tables) is **not** read. That is the whole trick.

### Step 3 — save, commit, push

Push your change to GitHub. The workflow generates the MP3, uploads it, and the
**▷ Audio** button appears on your post automatically. Done. 🎉

**Optional helpers inside the markers:**

```markdown
<!-- audio:skip:start -->  shown on the page but NOT read aloud  <!-- audio:skip:end -->
<!-- audio:pause -->       a short silence
```

> **Tip:** the start/end markers are the **standard, safe way** — you decide
> exactly what is spoken. There is also an optional *beta* that guesses the text
> for you (see "Auto-detect"), but the markers are recommended.

That's everything a blogger needs. The sections below are for maintainers.

---

## Moving parts

| File | Role |
| --- | --- |
| `scripts/generate_audio.py` | Reads tagged Markdown → cleaned text → MP3 + manifest |
| `scripts/setup_audio.py` | Gradio UI to choose/preview/save the speaker voice + settings |
| `_data/audio_settings.yml` | Saved speaker settings (voice, speed, lang, bitrate, normalization) |
| `.github/workflows/generate-audio.yml` | CI generation + R2 upload + manifest commit |
| `_data/audio_manifest.json` | Maps `audio_slug` → `{ audio_url, duration, voice, content_hash, … }` |
| `_includes/listen.html` | Renders the control when `page.audio` + a manifest entry exist |
| `_includes/listen-player.html` | The inline `▷ Audio` trigger + compact panel |
| `_includes/guided-tour.html` | Chaptered player that scrolls the page in sync with the narration |
| `_pages/ecosystem-narration.md` | Voice-over script + chapter cues for `/ecosystem/` |
| `_layouts/essay.html` | Premium byline: `Author · ▷ Audio · ◷ read time` |
| `_layouts/single.html` | Blog meta: `◷ read time · ▷ Audio` |

The generator scans both `_pages/` and `_posts/` for `audio: true` pages
(`--all`), or takes explicit file paths.

## Choose the narrator voice

```bash
make audio-ui        # sets up the venv from pyproject.toml and opens the app
```

(or manually, from the repo root: `pip install ".[ui]"` then
`python scripts/setup_audio.py`).

In the app: pick a **model** and click **⬇ Download model** to fetch the Kokoro
files, choose a **voice**, set speed/bitrate, **▶ Preview** a line, then
**💾 Save** — this writes `_data/audio_settings.yml`, which `generate_audio.py`
reads as its defaults. Commit that file to apply it.

> **Dependencies** live in `pyproject.toml`. `make audio-deps`, `make audio-ui`,
> and CI all install from it (`".[ui]"` / `".[upload]"`) into the project venv,
> so the system Python is never used by accident — which is what caused
> `ModuleNotFoundError: No module named 'kokoro_onnx'`.
>
> **Tooling:** the venv and installs use [uv](https://docs.astral.sh/uv/) when
> available (≈10–100× faster than pip — venv in ~0.1s, full install in seconds).
> If `uv` isn't on the system it is bootstrapped into the venv automatically,
> with a plain `pip` fallback. CI installs via `uv pip install --system`.

Model variants (all from the `kokoro-onnx` GitHub release):

| File | Notes |
| --- | --- |
| `kokoro-v1.0.fp16.onnx` | fp16, recommended balance (~177 MB) — default |
| `kokoro-v1.0.onnx` | full precision, best quality (~325 MB) |
| `kokoro-v1.0.int8.onnx` | int8 quantized, smallest/fastest (~92 MB) |

The model files are git-ignored. They download automatically when needed
(`KOKORO_AUTO_DOWNLOAD=1`, the default) and can be pre-fetched with:

```bash
python scripts/generate_audio.py --ensure-models   # uses the configured model
```

`scripts/kokoro_models.py` is the shared (std-lib only) downloader used by the
UI, the generator, and CI.

## Editorial intro (audiobook opening)

Every narration opens with a short spoken intro built from the page front
matter, so it feels like an audio essay rather than starting mid-sentence:

```
<eyebrow/category>.
<title>. <subtitle>.
Written by <author_name>. <author_role>.
<thesis / summary / excerpt>.
Section one. <first section title>.
<body…>
```

Fields used (each optional, skipped if absent): `eyebrow`, `headline`/`title`,
`subtitle`, `author_name` (falls back to `_config.yml` `author.name`),
`author_role`, and `thesis`/`summary`/`excerpt`. Numbered sections such as
`1 · The pile` are spoken as "Section one. The pile.". Set `audio_intro: false`
in a page's front matter to opt out. The intro is part of the content hash, so
changing the title/author/abstract refreshes the audio automatically.

## What gets read aloud (text extraction)

The script narrates **prose only**. It reads the Markdown source (never the
rendered theme, so nav/sidebars/footers are already excluded) and strips the
non-spoken parts, the way Reader View / readability / `trafilatura` do for
articles:

- removed: images, fenced/indented code, tables, `<figure>`+captions, raw URLs
  (anchor text is kept), Liquid tags, HTML comments/tags, horizontal rules;
- cleaned: numbered headings (`## 1 · The pile` → "The pile."), list/quote
  markers, emphasis markers.

### Choosing the narrated region

**Standard (default, recommended): explicit tags.** You control exactly what is
spoken. A page with `audio: true` but **no** tags and no opt-in is **skipped**
(safer than accidentally narrating chrome). Tags are HTML comments, so they stay
invisible and work in Markdown **or** HTML bodies:

```markdown
<!-- audio:start -->
...essay body...
<!-- audio:skip:start --> shown on page, not narrated <!-- audio:skip:end -->
<!-- audio:pause -->
<!-- audio:end -->
```

**Optional: auto-detect (beta) — zero tags.** Opt in per page with front matter
`audio_auto: true`, or per run with `--auto`. It selects the main prose using
reader-mode boilerplate trimming: drops `<script>`/`<style>`, trailing
"Continue/Related" cards, a leading companion-note blockquote, and trailing meta
(after a horizontal rule, or emoji/italic/sign-off paragraphs such as "Further
reading" / acknowledgements). On this site's 7 essays it produced
**byte-identical** narration to the hand-placed tags. Preview it on any file:

```bash
python scripts/generate_audio.py path/to/post.md --auto --check-tags
```

Industry note: explicit opt-in regions (like Medium/Substack "Listen") give the
cleanest control and are the standard here; reader-mode extraction (Mozilla
Readability, `trafilatura`, `newspaper3k`) is the standard for *automatic*
main-content detection and powers the optional beta.

## Chapter cues and the guided tour

A page can play its narration **and scroll itself** through the story, one
section at a time. That needs one extra thing from the pipeline: the exact
second each chapter starts.

Mark the chapters in the narrated text:

```markdown
<!-- audio:cue: architecture | Five projects. One personal stack -->
```

The id is required, the title after `|` is optional (it labels the chapter in
the player and is never spoken). Cues split the text into segments that are
synthesized in order, so the running sample count at each boundary *is* the
chapter start — accurate to a few hundredths of a second, and re-scaled to the
encoded MP3 after loudness normalization. They land in the manifest:

```json
"chapters": [
  { "id": "hero",         "title": "I wanted a personal AI", "start": 22.02 },
  { "id": "architecture", "title": "Five projects…",         "start": 99.5  }
]
```

Then tag the matching sections on the page and drop in the player:

```html
<section data-tour-chapter="architecture"> … </section>

{% raw %}{% include guided-tour.html audio=site.data.audio_manifest.ecosystem
                          slug="ecosystem" label="Play the guided tour" %}{% endraw %}
```

`_includes/guided-tour.html` renders the launcher, the player, the chapter rail
and the spotlight/auto-scroll behaviour. A cue with no matching section is
skipped with a console warning; with no chapters at all the include degrades to
a plain narration player, and with no JavaScript to a native `<audio controls>`.

Playback has one visible control at a time — the launcher **or** the player,
never both:

| State | On screen |
| --- | --- |
| `idle` | large "Start the guided tour"; no player |
| `playing-collapsed` | compact mini-player, bottom-right — **the normal state**: play/pause, `2 / 14`, `1:23 / 10:12`, level meter, expand. No chapter title. |
| `playing-expanded` | adds the chapter title, progress bar and prev/next/close; only after the reader opens it, folds back after 5 s idle |
| `paused-collapsed` | same mini-player, play icon |
| `closed` | no player; the launcher (or a quiet "Resume at m:ss" chip once it has scrolled away) |

Collapse and close are different actions: `⌄` keeps the narration running, `×`
stops it.

The chapter list is a **drawer**, not a column. Closed it is a hairline and one
violet dot at the right edge; it opens on the rail's chevron or the chapter
button in the full transport, and closes on an outside click, a chapter choice,
or `Esc`. It never opens by itself — the player already names the live chapter.
Below 1280px nothing is reserved for it at all and it slides in over the page.

The host page reserves only what is permanent — the player's bottom strip and
the drawer's closed hairline — see `body.gt-active` in
`_layouts/ecosystem.html`.

**Preview the chapter split without rendering audio:**

```bash
python scripts/generate_audio.py _pages/ecosystem-narration.md --check-tags
```

**Cues are part of the content hash**, so moving one marks the audio stale — but
the default policy still never regenerates on its own. Refresh on purpose:

```bash
python scripts/generate_audio.py _pages/ecosystem-narration.md --force
```

An MP3 that predates its cues has no chapter marks, and the generator says so
(`[no chapter marks — run --force to record them]`).

### Narration scripts for hand-built pages

A visual landing page (`_pages/ecosystem.html`) is mostly labels, diagram nodes
and card text — narrating it verbatim sounds like a screen reader. So the audio
gets its own source, the way a voice-over script is separate from the layout:
`_pages/ecosystem-narration.md` carries `published: false` (Jekyll emits no
page), `audio_slug: ecosystem`, the cue markers, and prose written to be *heard*.
The page then reads the manifest entry by that slug. The cue ids in the script
and the `data-tour-chapter` ids on the page are the contract between them.

Pages written in HTML can also opt in directly — `audio: true` is honoured in
`.html` front matter, and tags/cues work there unchanged.

## Add audio to a page

1. Front matter (omit `slug` for posts using `/blog/:title` so the URL is
   unchanged; `audio_slug` is the manifest key and never affects the URL):

   ```yaml
   audio_slug: my-essay
   audio: true
   ```

2. Add `<!-- audio:start/end -->` tags around the prose (standard). To use the
   optional beta instead, add `audio_auto: true` and skip the tags.

3. Preview the exact narration text (no model needed):

   ```bash
   python scripts/generate_audio.py _posts/2026-06-05-my-essay.md \
     --check-tags --preview-text /tmp/spoken.txt
   ```

4. Generate locally (needs `ffmpeg`, `espeak-ng`, the Kokoro model files, and
   `pip install -r requirements.txt`):

   ```bash
   export KOKORO_VOICE=am_michael KOKORO_SPEED=0.96
   python scripts/generate_audio.py _posts/2026-06-05-my-essay.md --force
   ```

## Caching: generate only what is missing

`_data/audio_manifest.json` is the index **and** the cache. Each entry stores a
`content_hash` = `sha256` of `{ cleaned text, voice, speed, model, bitrate,
normalization }`.

**Default policy — never recreate audio we already have.** For each `audio: true`
page the script:

- **already has the MP3** (manifest entry + object present locally or in R2,
  checked with `aws s3api head-object`) → **skip**, regardless of the hash. This
  keeps merges/CI from re-running expensive TTS for essays that already exist.
  If the text changed, it skips with a note: *"content changed — run --force."*
- **missing** (no entry, or the R2 object is gone) → generate it;
- `--force` (or `make audio-force`) → regenerate everything intentionally.

To restore the old "auto-regenerate whenever the content hash changes" behaviour,
set `AUDIO_REGEN_ON_CHANGE=1`.

So editing an essay does **not** silently re-run TTS — refresh it on purpose
with `--force` (the `content_hash` still tracks what changed).

Manifest entry shape:

```json
{
  "matrix-context": {
    "audio_url": "https://pub-….r2.dev/essays/2026/matrix-context.mp3",
    "object_key": "essays/2026/matrix-context.mp3",
    "source_path": "_posts/2026-06-05-matrix-context.md",
    "content_hash": "sha256-…",
    "duration": "8:06",
    "voice": "kokoro-am_michael",
    "speed": "0.96",
    "model": "kokoro-v1.0.fp16",
    "updated_at": "2026-06-06T00:00:00Z"
  }
}
```

## Production (Cloudflare R2)

CI generates the MP3 and uploads it to R2. MP3s are **not** committed (see
`.gitignore`); only `_data/audio_manifest.json` is tracked.

Required GitHub Actions secrets (Settings → Secrets and variables → Actions):

| Secret | Example |
| --- | --- |
| `R2_ACCOUNT_ID` | `d25a9fb6761ec9c7c7f25529f93f5acb` |
| `R2_BUCKET` | `blog-audio` |
| `R2_ACCESS_KEY_ID` | `f3e55617225140daae6c8d7072248e47` |
| `R2_SECRET_ACCESS_KEY` | *(from Cloudflare — never committed or logged)* |
| `R2_PUBLIC_BASE_URL` | `https://pub-18ecc6bab6074b2e89efa5c36d39a544.r2.dev` |

- S3 endpoint (no bucket in the host): `https://<R2_ACCOUNT_ID>.r2.cloudflarestorage.com`
- Object key convention: `essays/<year>/<slug>.mp3` (stable across base-URL changes)
- Public URL today: `https://pub-…r2.dev/essays/2026/matrix-context.mp3`
- Later (only the base URL changes): `https://audio.ruslanmv.com/essays/2026/matrix-context.mp3`

`R2_SECRET_ACCESS_KEY` must live **only** as a GitHub Actions secret — never in
`_config.yml`, `_data/`, the README, or workflow logs.
