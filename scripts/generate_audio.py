#!/usr/bin/env python3
"""
generate_audio.py — Generate high-quality MP3 narration for Jekyll / Minimal Mistakes essays.

Main idea:
  - If the Markdown file contains audio tags, only text between the tags is narrated.
  - If no tags exist, the whole Markdown body is narrated after cleanup.
  - The script creates an MP3 and updates a Jekyll-friendly manifest at _data/audio_manifest.json.

Recommended tags inside Markdown:

  <!-- audio:start -->
  Text that should be read aloud.
  <!-- audio:end -->

Optional exclusion tags inside a narrated region:

  <!-- audio:skip:start -->
  This part will not be read.
  <!-- audio:skip:end -->

Optional pause tag:

  <!-- audio:pause -->

Local test:
  python scripts/generate_audio.py _pages/matrix-context.md --preview-text /tmp/matrix-context-spoken.txt
  python scripts/generate_audio.py _pages/matrix-context.md --force

Environment variables:
  AUDIO_OUT_DIR          public/audio
  MANIFEST_PATH          _data/audio_manifest.json
  R2_PUBLIC_BASE_URL     https://pub-18ecc6bab6074b2e89efa5c36d39a544.r2.dev
  R2_BUCKET              blog-audio          (optional, for R2 existence check)
  R2_ACCOUNT_ID          <account id>        (optional, for R2 existence check)
  R2_ENDPOINT_URL        <override endpoint> (optional)
  KOKORO_MODEL           kokoro-v1.0.fp16.onnx
  KOKORO_VOICES          voices-v1.0.bin
  KOKORO_VOICE           am_michael
  KOKORO_LANG            en-us
  KOKORO_SPEED           0.96
  MP3_BITRATE            192k
  MAX_CHARS              350
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable

import frontmatter  # python-frontmatter
import numpy as np
import soundfile as sf

# Shared Kokoro model metadata + downloader (std-lib only). Ensure the scripts
# directory is importable whether run directly, from CI, or via importlib.
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
import kokoro_models

POSTS_DIR = Path(os.environ.get("POSTS_DIR", "_posts"))
PAGES_DIR = Path(os.environ.get("PAGES_DIR", "_pages"))
AUDIO_OUT_DIR = Path(os.environ.get("AUDIO_OUT_DIR", "public/audio"))
MANIFEST_PATH = Path(os.environ.get("MANIFEST_PATH", "_data/audio_manifest.json"))
SETTINGS_PATH = Path(os.environ.get("AUDIO_SETTINGS_PATH", "_data/audio_settings.yml"))
R2_PUBLIC_BASE_URL = os.environ.get(
    "R2_PUBLIC_BASE_URL", "https://pub-18ecc6bab6074b2e89efa5c36d39a544.r2.dev"
).rstrip("/")

# R2 / S3 config (used only for the optional object-existence check). The
# secret access key is read by the AWS CLI from the environment and is never
# logged or written to the manifest.
R2_BUCKET = os.environ.get("R2_BUCKET", "")
R2_ACCOUNT_ID = os.environ.get("R2_ACCOUNT_ID", "")
R2_ENDPOINT_URL = os.environ.get("R2_ENDPOINT_URL", "")


def load_speaker_settings() -> dict:
    """Read the saved speaker settings (voice/speed/lang) from _data.

    Written by scripts/setup_audio.py. Environment variables still win over the
    file, and the file wins over the built-in defaults. Returns {} if missing.
    """
    if not SETTINGS_PATH.exists():
        return {}
    try:
        import yaml  # provided transitively by python-frontmatter
        return yaml.safe_load(SETTINGS_PATH.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # pragma: no cover - non-fatal
        print(f"warning: could not read {SETTINGS_PATH}: {exc}", file=sys.stderr)
        return {}


_SPEAKER = load_speaker_settings()

KOKORO_MODEL = os.environ.get("KOKORO_MODEL") or str(_SPEAKER.get("model", "kokoro-v1.0.fp16.onnx"))
KOKORO_VOICES = os.environ.get("KOKORO_VOICES", "voices-v1.0.bin")
# Auto-download the model files when missing (set KOKORO_AUTO_DOWNLOAD=0 to disable).
KOKORO_AUTO_DOWNLOAD = os.environ.get("KOKORO_AUTO_DOWNLOAD", "1").lower() not in ("0", "false", "no", "")
# Default: only generate MISSING audio (never auto-recreate existing). Set
# AUDIO_REGEN_ON_CHANGE=1 to also regenerate when the content hash changes.
REGEN_ON_CHANGE = os.environ.get("AUDIO_REGEN_ON_CHANGE", "0").lower() in ("1", "true", "yes")
KOKORO_VOICE = os.environ.get("KOKORO_VOICE") or str(_SPEAKER.get("voice", "am_michael"))
KOKORO_LANG = os.environ.get("KOKORO_LANG") or str(_SPEAKER.get("lang", "en-us"))
KOKORO_SPEED = float(os.environ.get("KOKORO_SPEED") or _SPEAKER.get("speed", 0.96))

# Model identifier recorded in the manifest and folded into the content hash.
# Derived from the model filename (e.g. "kokoro-v1.0.fp16.onnx" -> "kokoro-v1.0.fp16").
MODEL_ID = os.environ.get("KOKORO_MODEL_ID", Path(KOKORO_MODEL).stem)

MP3_BITRATE = os.environ.get("MP3_BITRATE") or str(_SPEAKER.get("bitrate", "192k"))
# Loudness normalization filter applied at encode time; part of the hash so that
# changing it invalidates cached audio.
NORMALIZATION = os.environ.get("AUDIO_NORMALIZATION") or str(
    _SPEAKER.get("normalization", "loudnorm=I=-16:TP=-1.5:LRA=11")
)
MAX_CHARS = int(os.environ.get("MAX_CHARS", "350"))
SAMPLE_RATE = 24000
GAP_SECONDS = float(os.environ.get("GAP_SECONDS", "0.35"))

AUDIO_START_RE = re.compile(r"<!--\s*(?:audio|tts):start\s*-->", re.IGNORECASE)
AUDIO_END_RE = re.compile(r"<!--\s*(?:audio|tts):end\s*-->", re.IGNORECASE)
AUDIO_SKIP_START_RE = re.compile(r"<!--\s*(?:audio|tts):skip:start\s*-->", re.IGNORECASE)
AUDIO_SKIP_END_RE = re.compile(r"<!--\s*(?:audio|tts):skip:end\s*-->", re.IGNORECASE)
AUDIO_PAUSE_RE = re.compile(r"<!--\s*(?:audio|tts):pause\s*-->", re.IGNORECASE)
HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)


# --- Auto-detect (beta): reader-mode boilerplate trimming -------------------
# Industry approach: keep the main article prose and drop chrome, the way
# Reader View / readability / trafilatura do. We work on the Markdown body
# (never the rendered theme), so navigation, sidebars, and footers are already
# out of scope; this trims the in-body boilerplate that remains.
_HR_RE = re.compile(r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$")
_META_MARKERS_RE = re.compile(
    r"(further reading|acknowledg|companion to|official site|related (?:posts|reading)"
    r"|continue reading|see also|references?\b|read (?:more|next))",
    re.IGNORECASE,
)
# A trailing paragraph is treated as boilerplate when it starts with one of
# these: emoji link bullets, italics-only notes, blockquote sign-offs, or a
# short bold sign-off line.
_META_PARA_RE = re.compile(r"^\s*(?:[\U0001F300-\U0001FAFF←-⇿☀-➿]|>|\*(?!\*))", re.UNICODE)


def _looks_meta(segment: str) -> bool:
    """True if a trailing segment looks like a boilerplate note rather than prose."""
    text = segment.strip()
    if not text:
        return True
    if len(text) > 700:
        return False
    if _META_MARKERS_RE.search(text):
        return True
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    metaish = sum(1 for ln in lines if _META_PARA_RE.match(ln))
    return bool(lines) and metaish >= max(1, len(lines) // 2)


def auto_select(md: str) -> str:
    """Best-effort extraction of the narratable prose from a Markdown body (beta).

    Removes: <script>/<style> and trailing "Continue/Related" card containers;
    a leading companion-note blockquote; and trailing meta sections (after a
    horizontal rule, or trailing emoji/italic/sign-off paragraphs). Explicit
    <!-- audio:start/end --> tags are always preferred and override this.
    """
    s = re.sub(r"<script\b.*?</script>", " ", md, flags=re.DOTALL | re.IGNORECASE)
    s = re.sub(r"<style\b.*?</style>", " ", s, flags=re.DOTALL | re.IGNORECASE)
    # Trailing "continue/related" HTML card container to end of document.
    s = re.sub(r"\n<div[^>]*\bae-(?:continue|cards)\b.*$", "\n", s, flags=re.DOTALL | re.IGNORECASE)

    lines = s.split("\n")

    # Leading companion-note blockquote (e.g. "> *A companion to ...*").
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i < len(lines) and lines[i].lstrip().startswith(">"):
        j = i
        while j < len(lines) and lines[j].lstrip().startswith(">"):
            j += 1
        quote = " ".join(lines[i:j])
        if ("](" in quote or "*" in quote) and len(quote) < 600:
            lines = lines[j:]

    # Trailing meta section(s) after the last horizontal rule.
    while True:
        hrs = [k for k, ln in enumerate(lines) if _HR_RE.match(ln)]
        if not hrs:
            break
        last = hrs[-1]
        if _looks_meta("\n".join(lines[last + 1:])):
            lines = lines[:last]
        else:
            break

    # Trailing emoji/italic/sign-off paragraphs that have no preceding rule.
    paras = re.split(r"\n\s*\n", "\n".join(lines).strip())
    while paras:
        last = paras[-1].strip()
        if not last:
            paras.pop()
            continue
        short_signoff = last.startswith("**") and len(last) < 120
        if _META_PARA_RE.match(last) or _META_MARKERS_RE.search(last) or short_signoff:
            paras.pop()
        else:
            break
    return "\n\n".join(paras).strip()


def has_audio_tags(md: str) -> bool:
    """True if the body contains an explicit <!-- audio:start --> marker."""
    return bool(AUDIO_START_RE.search(md))


def extract_audio_regions(md: str) -> tuple[str, str]:
    """Return (markdown_to_narrate, mode). mode is 'tagged' or 'auto'.

    If audio:start/audio:end tags exist, narrate only those blocks (the explicit,
    author-controlled path). Otherwise fall back to auto-detection (beta), which
    keeps the main prose and trims boilerplate. Multiple tagged blocks are allowed
    and are joined with a paragraph break.
    """
    starts = list(AUDIO_START_RE.finditer(md))
    ends = list(AUDIO_END_RE.finditer(md))

    if not starts and not ends:
        return auto_select(md), "auto"
    if len(starts) != len(ends):
        raise ValueError(
            f"Audio tag mismatch: found {len(starts)} audio:start tag(s) and {len(ends)} audio:end tag(s)."
        )

    blocks: list[str] = []
    search_from = 0
    for start in starts:
        end = AUDIO_END_RE.search(md, start.end())
        if not end:
            raise ValueError("Found audio:start without a following audio:end.")
        if start.start() < search_from:
            raise ValueError("Overlapping or nested audio:start/audio:end regions are not supported.")
        block = md[start.end() : end.start()]
        blocks.append(block.strip())
        search_from = end.end()

    return "\n\n".join(blocks), "tagged"


def remove_skip_regions(md: str) -> str:
    starts = list(AUDIO_SKIP_START_RE.finditer(md))
    ends = list(AUDIO_SKIP_END_RE.finditer(md))
    if len(starts) != len(ends):
        raise ValueError(
            f"Audio skip tag mismatch: found {len(starts)} audio:skip:start tag(s) and {len(ends)} audio:skip:end tag(s)."
        )
    return re.sub(
        r"<!--\s*(?:audio|tts):skip:start\s*-->.*?<!--\s*(?:audio|tts):skip:end\s*-->",
        " ",
        md,
        flags=re.IGNORECASE | re.DOTALL,
    )


_NUM_WORDS = {
    1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven",
    8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen",
    14: "fourteen", 15: "fifteen", 16: "sixteen", 17: "seventeen", 18: "eighteen",
    19: "nineteen", 20: "twenty",
}
# Numbered section heading like "1 · The pile" (middot separator only, so it
# never matches sub-numbering such as "2.1" or "Chapter 1:").
_SECTION_RE = re.compile(r"^\s*(\d{1,2})\s*·\s*(.+)$")


def spoken_heading(raw: str) -> str:
    """Turn a heading into a spoken, sentence-like pause.

    "1 · The pile"  -> "Section one. The pile."
    "The throughline" -> "The throughline."
    """
    raw = raw.strip().rstrip(".")
    if not raw:
        return "\n\n"
    m = _SECTION_RE.match(raw)
    if m:
        word = _NUM_WORDS.get(int(m.group(1)), m.group(1))
        title = m.group(2).strip()
        return f"\n\nSection {word}. {title}.\n\n" if title else f"\n\nSection {word}.\n\n"
    return f"\n\n{raw}.\n\n"


def clean_markdown(md: str) -> str:
    """Turn selected Markdown into plain spoken text."""
    text = remove_skip_regions(md)
    text = AUDIO_PAUSE_RE.sub(".\n\n", text)

    # Drop Jekyll/Liquid includes and assignments. They should not be narrated.
    text = re.sub(r"{%.*?%}", " ", text, flags=re.DOTALL)
    text = re.sub(r"{{.*?}}", " ", text, flags=re.DOTALL)

    # Drop fenced code blocks and indented code-ish blocks.
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"~~~.*?~~~", " ", text, flags=re.DOTALL)

    # Drop whole <figure> blocks (including any <figcaption>): they are visual,
    # and their captions should not be read aloud.
    text = re.sub(r"<figure\b.*?</figure>", " ", text, flags=re.DOTALL | re.IGNORECASE)

    # Remove images entirely. Keep visible link text, remove URL.
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)

    # Strip raw URLs.
    text = re.sub(r"https?://\S+", " ", text)

    # Inline code: keep the content as words.
    text = re.sub(r"`([^`]*)`", r"\1", text)

    # Markdown emphasis markers.
    text = re.sub(r"(\*\*|__|\*|_|~~)", "", text)

    # Headings become sentence-like pauses; numbered sections become "Section N.".
    text = re.sub(r"^(#{1,6})\s+(.*)$", lambda m: spoken_heading(m.group(2)), text, flags=re.MULTILINE)

    # Blockquotes and list markers.
    text = re.sub(r"^\s{0,3}>\s?", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)

    # Tables and horizontal rules.
    text = re.sub(r"^\s*\|.*\|\s*$", " ", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*([-=*]\s*){3,}$", " ", text, flags=re.MULTILINE)

    # Strip HTML comments after extracting audio markers.
    text = HTML_COMMENT_RE.sub(" ", text)

    # HTML headings (<h1>..<h6>) get the same treatment, so "<h2>1 · The pile</h2>"
    # is spoken as "Section one. The pile.".
    def _html_heading(m: re.Match) -> str:
        title = re.sub(r"<[^>]+>", " ", m.group(1))
        title = re.sub(r"\s+", " ", title)
        return spoken_heading(title)

    text = re.sub(r"<h[1-6][^>]*>(.*?)</h[1-6]>", _html_heading, text, flags=re.IGNORECASE | re.DOTALL)

    # Strip any remaining HTML tags.
    text = re.sub(r"<[^>]+>", " ", text)

    # Gentle pronunciation improvements for common technical text.
    replacements = {
        "AI": "A I",
        "LLM": "L L M",
        "LLMs": "L L M's",
        "RAG": "R A G",
        "API": "A P I",
        "URL": "U R L",
        "GitHub": "Git Hub",
    }
    for k, v in replacements.items():
        text = re.sub(rf"\b{re.escape(k)}\b", v, text)

    # Collapse whitespace and remove accidental doubled punctuation from pause tags.
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"([.!?])\s+\.", r"\1", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text.strip()


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def chunk_text(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    chunks: list[str] = []
    buf = ""
    for sentence in _SENT_SPLIT.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > max_chars:
            if buf:
                chunks.append(buf.strip())
                buf = ""
            piece = ""
            for clause in re.split(r"(?<=,)\s+", sentence):
                if len(piece) + len(clause) + 1 <= max_chars:
                    piece = f"{piece} {clause}".strip()
                else:
                    if piece:
                        chunks.append(piece.strip())
                    piece = clause
            if piece:
                chunks.append(piece.strip())
            continue
        if len(buf) + len(sentence) + 1 <= max_chars:
            buf = f"{buf} {sentence}".strip()
        else:
            chunks.append(buf.strip())
            buf = sentence
    if buf:
        chunks.append(buf.strip())
    return [c for c in chunks if c]


_KOKORO = None


def ensure_models() -> None:
    """Make sure the configured Kokoro model + voices files are present locally,
    downloading them from the GitHub release if needed."""
    if Path(KOKORO_MODEL).exists() and Path(KOKORO_VOICES).exists():
        return
    if not KOKORO_AUTO_DOWNLOAD:
        raise FileNotFoundError(
            f"Model files not found: {KOKORO_MODEL}, {KOKORO_VOICES}. "
            "Download them, or set KOKORO_AUTO_DOWNLOAD=1."
        )

    def _progress(done: int, total: int) -> None:
        mb, tot = done / (1 << 20), (total / (1 << 20)) if total else 0
        end = "\n" if total and done >= total else "\r"
        print(f"    {mb:6.1f} / {tot:6.1f} MB", end=end, flush=True)

    print(f"  downloading Kokoro model files from {kokoro_models.RELEASE_BASE} ...")
    for name, dest, status in kokoro_models.ensure_models(KOKORO_MODEL, KOKORO_VOICES, progress=_progress):
        print(f"    {status}: {dest}")


def _get_kokoro():
    global _KOKORO
    if _KOKORO is None:
        from kokoro_onnx import Kokoro
        ensure_models()
        _KOKORO = Kokoro(KOKORO_MODEL, KOKORO_VOICES)
    return _KOKORO


def synthesize(text: str, dry_run: bool = False) -> np.ndarray:
    chunks = chunk_text(text)
    gap = np.zeros(int(GAP_SECONDS * SAMPLE_RATE), dtype=np.float32)
    pieces: list[np.ndarray] = []

    if dry_run:
        rng = np.random.default_rng(0)
        for _ in chunks:
            pieces.append((rng.standard_normal(int(0.1 * SAMPLE_RATE)) * 1e-3).astype(np.float32))
            pieces.append(gap)
        return np.concatenate(pieces) if pieces else np.zeros(SAMPLE_RATE, dtype=np.float32)

    kokoro = _get_kokoro()
    for i, chunk in enumerate(chunks, 1):
        samples, sr = kokoro.create(chunk, voice=KOKORO_VOICE, speed=KOKORO_SPEED, lang=KOKORO_LANG)
        if sr != SAMPLE_RATE:
            raise RuntimeError(f"Unexpected Kokoro sample rate: {sr}")
        pieces.append(np.asarray(samples, dtype=np.float32))
        pieces.append(gap)
        print(f"    chunk {i}/{len(chunks)} ({len(chunk)} chars)")
    return np.concatenate(pieces) if pieces else np.zeros(1, dtype=np.float32)


def encode_mp3(wav: np.ndarray, out_mp3: Path) -> None:
    out_mp3.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_wav = Path(tmp.name)
    try:
        sf.write(tmp_wav, wav, SAMPLE_RATE)
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(tmp_wav),
            "-af", NORMALIZATION,
            "-codec:a", "libmp3lame", "-b:a", MP3_BITRATE,
            str(out_mp3),
        ]
        subprocess.run(cmd, check=True)
    finally:
        tmp_wav.unlink(missing_ok=True)


def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {}


def save_manifest(manifest: dict) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def slug_for(path: Path, post) -> str:
    if post.get("audio_slug"):
        return str(post["audio_slug"])
    if post.get("slug"):
        return str(post["slug"])
    permalink = str(post.get("permalink", "")).strip("/")
    if permalink:
        return Path(permalink).name
    return path.stem


def year_for(post, path: Path | None = None) -> str:
    # Prefer an explicit date, then essay_date, then the YYYY-MM-DD- filename
    # prefix (Jekyll posts), then fall back to the current year.
    for key in ("date", "essay_date"):
        date = post.get(key)
        if isinstance(date, (_dt.date, _dt.datetime)):
            return f"{date.year}"
        if isinstance(date, str):
            m = re.match(r"(\d{4})", date)
            if m:
                return m.group(1)
    if path is not None:
        m = re.match(r"(\d{4})-\d{2}-\d{2}-", Path(path).name)
        if m:
            return m.group(1)
    return f"{_dt.date.today().year}"


def content_hash(text: str, voice: str, speed: float) -> str:
    """Stable hash of everything that affects the rendered audio.

    Folding the voice, speed, model, bitrate, and normalization filter into the
    hash means a Markdown edit that does not change the narrated text (or any of
    these settings) will not trigger regeneration. Returned as "sha256-<hex>".
    """
    hash_input = {
        "text": text,
        "voice": voice,
        "speed": f"{speed}",
        "model": MODEL_ID,
        "bitrate": MP3_BITRATE,
        "normalization": NORMALIZATION,
    }
    digest = hashlib.sha256(
        json.dumps(hash_input, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return f"sha256-{digest}"


def r2_object_exists(object_key: str) -> bool | None:
    """Best-effort check whether an object already exists in the R2 bucket.

    Returns:
      True  - the object is present in R2.
      False - credentials/config are available but the object is missing.
      None  - cannot check (no credentials, no bucket, or AWS CLI unavailable),
              in which case callers should fall back to the manifest hash.
    """
    if not (R2_BUCKET and (R2_ENDPOINT_URL or R2_ACCOUNT_ID)):
        return None
    if not (os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY")):
        return None
    endpoint = R2_ENDPOINT_URL or f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
    cmd = [
        "aws", "s3api", "head-object",
        "--bucket", R2_BUCKET,
        "--key", object_key,
        "--endpoint-url", endpoint,
    ]
    env = {**os.environ, "AWS_DEFAULT_REGION": os.environ.get("AWS_DEFAULT_REGION", "auto")}
    try:
        result = subprocess.run(
            cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    except FileNotFoundError:
        return None  # AWS CLI not installed (e.g. local dev)
    return result.returncode == 0


def fmt_duration(n_samples: int) -> str:
    total = int(round(n_samples / SAMPLE_RATE))
    # Non-padded minutes to match the player display (e.g. "8:06", "10:18").
    return f"{total // 60}:{total % 60:02d}"


def probe_duration(path: Path) -> str:
    """Duration ("m:ss") of an existing MP3 via ffprobe, or "" if unavailable."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True,
        )
        total = int(round(float(out.stdout.strip())))
        return f"{total // 60}:{total % 60:02d}"
    except Exception:
        return ""


def manifest_entry(object_key: str, source_path, h: str, duration: str, mode: str) -> dict:
    # Cache-busting ?v=<hash> on the public URL (object key stays clean).
    version = h.split("-")[-1][:8]
    return {
        "audio_url": f"{R2_PUBLIC_BASE_URL}/{object_key}?v={version}",
        "object_key": object_key,
        "source_path": str(source_path),
        "content_hash": h,
        "duration": duration,
        "voice": f"kokoro-{KOKORO_VOICE}",
        "speed": f"{KOKORO_SPEED}",
        "model": MODEL_ID,
        "selection": mode,
        "updated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }


def discover_posts(paths: Iterable[str]) -> list[Path]:
    explicit = [Path(a) for a in paths]
    if explicit:
        return [p for p in explicit if p.suffix.lower() in (".md", ".markdown")]
    found: list[Path] = []
    for base in (PAGES_DIR, POSTS_DIR):
        if base.exists():
            found.extend(base.rglob("*.md"))
    return sorted(set(found))


_SITE_AUTHOR = None


def site_author_name() -> str:
    """Author name from _config.yml (fallback for posts without author_name)."""
    global _SITE_AUTHOR
    if _SITE_AUTHOR is None:
        _SITE_AUTHOR = ""
        cfg = Path("_config.yml")
        if cfg.exists():
            try:
                import yaml
                data = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
                _SITE_AUTHOR = str((data.get("author") or {}).get("name", "") or "")
            except Exception:
                pass
    return _SITE_AUTHOR


def _sentence(text: str) -> str:
    """Normalize a front-matter value into one spoken sentence."""
    t = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", str(text))   # links -> text
    t = re.sub(r"(\*\*|__|\*|_|`)", "", t)                    # emphasis markers
    t = t.replace("·", ". ")                                  # middot -> sentence break
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return ""
    return t if t[-1] in ".!?:" else t + "."


def build_intro(meta: dict) -> str:
    """Editorial audiobook intro from front matter:
    category, title + subtitle, "Written by <author>. <role>.", then the abstract.
    Returned as plain Markdown-free text (it is fed through clean_markdown with the
    body so pronunciation fixes apply uniformly). Empty if opted out.
    """
    if meta.get("audio_intro") is False:
        return ""
    parts: list[str] = []

    if meta.get("eyebrow"):
        parts.append(_sentence(meta["eyebrow"]))

    title = meta.get("headline") or meta.get("title")
    if title:
        line = _sentence(title)
        if meta.get("subtitle"):
            line += " " + _sentence(meta["subtitle"])
        parts.append(line)

    author = meta.get("author_name") or site_author_name()
    if author:
        line = f"Written by {str(author).strip()}."
        if meta.get("author_role"):
            role = str(meta["author_role"]).replace("·", ",")
            line += " " + _sentence(role)
        parts.append(line)

    abstract = meta.get("thesis") or meta.get("summary") or meta.get("excerpt")
    if abstract:
        parts.append(_sentence(abstract))

    return "\n\n".join(p for p in parts if p)


def build_spoken_text(post_content: str, intro_md: str = "", force_auto: bool = False) -> tuple[str, str]:
    if force_auto:
        selected_md, mode = auto_select(post_content), "auto"
    else:
        selected_md, mode = extract_audio_regions(post_content)
    if intro_md:
        # Prepend the intro to the selected body, then clean together so the same
        # pronunciation/emphasis rules apply to both.
        selected_md = intro_md + "\n\n" + selected_md
    spoken = clean_markdown(selected_md)
    return spoken, mode


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate MP3 narration for Jekyll Markdown essays.")
    parser.add_argument("paths", nargs="*", help="Markdown files. If omitted (or --all), scans POSTS_DIR.")
    parser.add_argument("--all", action="store_true", help="Scan every Markdown file under POSTS_DIR for audio: true pages.")
    parser.add_argument("--force", action="store_true", help="Regenerate even if the manifest content_hash is unchanged.")
    parser.add_argument("--dry-run", action="store_true", help="Create a tiny synthetic MP3 without loading the TTS model.")
    parser.add_argument("--check-tags", action="store_true", help="Only validate audio tags and print what would be narrated.")
    parser.add_argument("--auto", action="store_true", help="Beta: ignore tags and use auto-detection to select the narrated text.")
    parser.add_argument("--preview-text", type=Path, help="Write the cleaned spoken text to a .txt file for review.")
    parser.add_argument("--ensure-models", action="store_true", help="Download the Kokoro model + voices files if missing, then exit.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print a line for every skipped (non-audio) file.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    if args.ensure_models:
        for name, dest, status in kokoro_models.ensure_models(KOKORO_MODEL, KOKORO_VOICES):
            print(f"{status}: {dest}")
        return 0

    posts = discover_posts(args.paths)
    if not posts:
        print(f"No posts found under {POSTS_DIR}")
        return 0

    manifest = load_manifest()
    rendered = 0

    skipped = 0
    for path in posts:
        if not path.exists():
            if args.verbose:
                print(f"skip (missing): {path}")
            skipped += 1
            continue
        post = frontmatter.load(path)
        meta = post.metadata

        # Only narrate pages/posts that explicitly opt in with `audio: true`.
        # This mirrors the Jekyll include logic ({% if page.audio %}) and keeps
        # bulk/push-triggered runs from narrating every post in the repo.
        # These skips are the common case (most posts have no audio), so they are
        # silent unless --verbose, to keep `make serve` output readable.
        if not meta.get("audio") or meta.get("audio_disabled") is True:
            if args.verbose:
                print(f"skip (no audio: true): {path}")
            skipped += 1
            continue

        # Standard procedure: explicit <!-- audio:start/end --> tags decide what
        # is narrated. Auto-detection is an OPT-IN beta — only used when asked
        # for with --auto or front-matter `audio_auto: true`. An audio: true page
        # with no tags and no opt-in is skipped (safer than narrating chrome).
        tagged = has_audio_tags(post.content)
        auto_opt_in = args.auto or bool(meta.get("audio_auto"))
        if not tagged and not auto_opt_in:
            print(
                f"skip (no audio tags): {path}\n"
                "      add <!-- audio:start --> / <!-- audio:end --> (standard), or\n"
                "      opt into auto-detect (beta) with `audio_auto: true` or --auto."
            )
            continue

        slug = slug_for(path, post)
        year = year_for(post, path)
        try:
            intro_md = build_intro(meta)
            spoken, mode = build_spoken_text(
                post.content, intro_md=intro_md, force_auto=(args.auto or not tagged)
            )
        except ValueError as exc:
            print(f"ERROR in {path}: {exc}", file=sys.stderr)
            return 2

        if args.preview_text:
            args.preview_text.parent.mkdir(parents=True, exist_ok=True)
            args.preview_text.write_text(spoken + "\n", encoding="utf-8")
            print(f"preview text written: {args.preview_text}")

        print(f"{slug}: mode={mode}, spoken_chars={len(spoken)}, chunks={len(chunk_text(spoken))}")
        if args.check_tags:
            print("--- preview ---")
            print(spoken[:2000] + ("..." if len(spoken) > 2000 else ""))
            continue

        if len(spoken) < 20:
            print(f"skip (too short after cleaning): {path}")
            continue

        h = content_hash(spoken, KOKORO_VOICE, KOKORO_SPEED)
        object_key = f"essays/{year}/{slug}.mp3"
        out_mp3 = AUDIO_OUT_DIR / object_key
        prior = manifest.get(slug)

        # Default policy: generate only MISSING audio — never automatically
        # recreate audio we already have. CI fetches existing MP3s from R2 into
        # AUDIO_OUT_DIR before this runs, so a present local file is the reliable
        # "we already have it" signal (no dependency on head-object or the
        # committed hash, which makes this resilient to branch/manifest drift).
        # To intentionally refresh after editing an essay, use --force
        # (or `make audio-force`). Set AUDIO_REGEN_ON_CHANGE=1 to restore the
        # old "regenerate when the content hash changes" behaviour.
        if not args.force:
            local = out_mp3.exists()
            r2_state = True if local else r2_object_exists(object_key)
            exists = local or r2_state is True or (r2_state is None and prior is not None)
            hash_changed = bool(prior) and prior.get("content_hash") != h
            if exists and not (REGEN_ON_CHANGE and hash_changed):
                where = "local" if local else ("R2" if r2_state is True else "manifest")
                note = "  [content changed — run --force to refresh]" if hash_changed else ""
                # Keep the manifest complete even if its entry was missing/drifted
                # (e.g. after a branch merge): index the existing audio, no TTS.
                if not prior:
                    manifest[slug] = manifest_entry(object_key, path, h, probe_duration(out_mp3), mode)
                    note = "  [indexed existing audio]"
                print(f"up-to-date ({where}): {slug}{note}")
                continue
            if not exists:
                print(f"generate (missing): {slug}")
            else:
                print(f"regenerate (content changed): {slug}")

        print(f"render: {slug} -> {out_mp3}")
        wav = synthesize(spoken, dry_run=args.dry_run)
        encode_mp3(wav, out_mp3)
        manifest[slug] = manifest_entry(object_key, path, h, fmt_duration(len(wav)), mode)
        rendered += 1

    if not args.check_tags:
        save_manifest(manifest)
        extra = f", skipped {skipped} non-audio file(s)" if skipped else ""
        print(f"\nDone. Rendered {rendered} post(s){extra}. Manifest: {MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
