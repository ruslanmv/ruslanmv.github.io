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
