#!/usr/bin/env python3
"""
setup_audio.py — pick the narration voice/model and settings, download the
Kokoro model files, preview the voice, and save everything into the repo at
_data/audio_settings.yml.

Those settings become the defaults used by scripts/generate_audio.py (matching
environment variables still override them). Because voice/speed/model/bitrate/
normalization feed the audio content hash, changing them will cause the next
generation run to refresh the affected MP3s.

Run (recommended — sets up the venv and launches the app):
    make audio-ui

Or manually, from the repo root:
    pip install ".[ui]"          # core + UI deps, from pyproject.toml
    python scripts/setup_audio.py

Use the "Download model" button in the UI to fetch the model files.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import gradio as gr

# Shared, dependency-light model metadata + downloader.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import kokoro_models

REPO_ROOT = Path(__file__).resolve().parents[1]
SETTINGS_PATH = REPO_ROOT / "_data" / "audio_settings.yml"

# Curated Kokoro voices. label -> voice id.
VOICES: dict[str, str] = {
    "am_michael — calm American male (default)": "am_michael",
    "am_adam — deeper American male": "am_adam",
    "af_heart — warm American female": "af_heart",
    "af_bella — bright American female": "af_bella",
    "af_nicole — soft American female": "af_nicole",
    "bf_emma — British female": "bf_emma",
    "bf_isabella — British female": "bf_isabella",
    "bm_george — British male": "bm_george",
    "bm_lewis — British male": "bm_lewis",
}

MODEL_LABELS = [kokoro_models.model_label(f) for f in kokoro_models.MODEL_CHOICES]
LANGS = ["en-us", "en-gb"]
BITRATES = ["128k", "160k", "192k", "256k", "320k"]
DEFAULT_NORMALIZATION = "loudnorm=I=-16:TP=-1.5:LRA=11"
SAMPLE_TEXT = (
    "A flat retriever sees similar text. A typed memory sees the shape of the "
    "idea, and can always show you why it chose what it chose."
)


def _voice_label_for(voice_id: str) -> str:
    for label, vid in VOICES.items():
        if vid == voice_id:
            return label
    return next(iter(VOICES))


def load_settings() -> dict:
    if not SETTINGS_PATH.exists():
        return {}
    import yaml
    return yaml.safe_load(SETTINGS_PATH.read_text(encoding="utf-8")) or {}


def save_settings(voice: str, model_label: str, speed: float, lang: str,
                  bitrate: str, normalization: str) -> str:
    voice_id = VOICES.get(voice, voice)
    model_file = kokoro_models.filename_from_label(model_label)
    speed = round(float(speed), 3)
    content = f"""# Speaker / narration settings for the essay audio pipeline.
#
# Edit with the Gradio UI:  python scripts/setup_audio.py
# Or by hand. scripts/generate_audio.py reads these as defaults; matching
# environment variables (KOKORO_VOICE, KOKORO_SPEED, ...) still override them.
#
# These values feed the content hash, so changing voice/speed/model/bitrate/
# normalization will (correctly) regenerate audio on the next run.

# Kokoro voice id. See scripts/setup_audio.py for the full list.
voice: {voice_id}

# Kokoro ONNX model file (downloaded by setup_audio.py / the pipeline).
model: {model_file}

# Narration speed. 0.95–0.98 reads well for essays.
speed: {speed}

# Language / accent code passed to Kokoro.
lang: {lang}

# MP3 encode bitrate.
bitrate: {bitrate}

# ffmpeg loudness normalization filter (podcast-style, EBU R128).
normalization: {normalization}
"""
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(content, encoding="utf-8")
    try:
        rel = SETTINGS_PATH.relative_to(REPO_ROOT)
    except ValueError:
        rel = SETTINGS_PATH
    return (
        f"✅ Saved to {rel}\n"
        f"voice={voice_id}  model={model_file}  speed={speed}\n"
        f"lang={lang}  bitrate={bitrate}\n"
        f"normalization={normalization}\n\n"
        "Commit the file to apply it. The next generation run will refresh audio "
        "whose voice/speed/model/bitrate/normalization changed."
    )


def download_model(model_label: str, progress=gr.Progress()) -> str:
    """Download the selected model file + voices into the repo root."""
    model_file = kokoro_models.filename_from_label(model_label)
    progress(0, desc=f"Preparing {model_file} …")

    def cb(done: int, total: int) -> None:
        frac = (done / total) if total else 0
        mb, tot = done / (1 << 20), (total / (1 << 20)) if total else 0
        progress(frac, desc=f"{mb:.0f} / {tot:.0f} MB")

    try:
        results = kokoro_models.ensure_models(model_file, kokoro_models.VOICES_FILE,
                                              root=REPO_ROOT, progress=cb)
    except Exception as exc:
        raise gr.Error(f"Download failed: {exc}")

    lines = [f"{status}: {path.name} ({path.stat().st_size / (1 << 20):.1f} MB)"
             for _, path, status in results]
    return "✅ Models ready:\n" + "\n".join(lines) + "\n\nYou can now Preview the voice."


def preview(text: str, voice: str, model_label: str, speed: float, lang: str):
    """Synthesize a short WAV preview with the selected settings."""
    text = (text or "").strip()
    if not text:
        raise gr.Error("Enter some sample text to preview.")
    voice_id = VOICES.get(voice, voice)
    model_file = kokoro_models.filename_from_label(model_label)

    model_path = REPO_ROOT / model_file
    voices_path = REPO_ROOT / kokoro_models.VOICES_FILE
    if not (model_path.exists() and voices_path.exists()):
        raise gr.Error(
            f"Model files not found. Click “⬇ Download model” first to fetch "
            f"{model_file} and {kokoro_models.VOICES_FILE}."
        )
    try:
        from kokoro_onnx import Kokoro
        import soundfile as sf
    except Exception as exc:
        raise gr.Error(
            f"Missing audio deps ({exc}). Run the app via the project venv: "
            "`make audio-ui` — or `pip install \".[ui]\"` from the repo root."
        )

    kokoro = Kokoro(str(model_path), str(voices_path))
    samples, sr = kokoro.create(text, voice=voice_id, speed=float(speed), lang=lang)
    out = Path(tempfile.mkdtemp()) / f"preview-{voice_id}.wav"
    sf.write(out, samples, sr)
    return str(out)


def build_ui() -> gr.Blocks:
    cur = load_settings()
    cur_voice_label = _voice_label_for(str(cur.get("voice", "am_michael")))
    cur_model_label = kokoro_models.model_label(str(cur.get("model", "kokoro-v1.0.fp16.onnx")))
    if cur_model_label not in MODEL_LABELS:
        cur_model_label = MODEL_LABELS[0]
    cur_speed = float(cur.get("speed", 0.96))
    cur_lang = str(cur.get("lang", "en-us"))
    cur_bitrate = str(cur.get("bitrate", "192k"))
    cur_norm = str(cur.get("normalization", DEFAULT_NORMALIZATION))

    with gr.Blocks(title="Essay audio — speaker setup") as demo:
        gr.Markdown(
            "# 🎙️ Essay audio — speaker setup\n"
            "Choose the model and voice, **download** the model, **preview** a "
            "line, then **save** to `_data/audio_settings.yml` — the defaults for "
            "`scripts/generate_audio.py`."
        )
        with gr.Row():
            with gr.Column():
                model = gr.Dropdown(choices=MODEL_LABELS, value=cur_model_label, label="Kokoro model")
                download_btn = gr.Button("⬇ Download model", variant="secondary")
                download_status = gr.Textbox(label="Model status", interactive=False, lines=3)
                voice = gr.Dropdown(choices=list(VOICES.keys()), value=cur_voice_label, label="Voice")
                speed = gr.Slider(0.80, 1.20, value=cur_speed, step=0.01, label="Speed")
                lang = gr.Dropdown(choices=LANGS, value=cur_lang, label="Language / accent")
                bitrate = gr.Dropdown(choices=BITRATES, value=cur_bitrate, label="MP3 bitrate")
                normalization = gr.Textbox(value=cur_norm, label="ffmpeg loudness normalization")
            with gr.Column():
                sample = gr.Textbox(value=SAMPLE_TEXT, lines=4, label="Sample text to preview")
                preview_btn = gr.Button("▶ Preview voice", variant="secondary")
                audio_out = gr.Audio(label="Preview", type="filepath")
                save_btn = gr.Button("💾 Save settings", variant="primary")
                status = gr.Textbox(label="Status", interactive=False, lines=4)

        download_btn.click(download_model, [model], download_status)
        preview_btn.click(preview, [sample, voice, model, speed, lang], audio_out)
        save_btn.click(save_settings, [voice, model, speed, lang, bitrate, normalization], status)
    return demo


if __name__ == "__main__":
    build_ui().launch()
