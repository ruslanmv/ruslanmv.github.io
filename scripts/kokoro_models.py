#!/usr/bin/env python3
"""
kokoro_models.py — shared Kokoro model metadata + downloader.

Deliberately dependency-light (only the standard library) so it can be imported
by both the Gradio setup UI (scripts/setup_audio.py) and the generation pipeline
(scripts/generate_audio.py) without pulling in numpy/onnxruntime.
"""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from typing import Callable

# GitHub release that hosts the Kokoro ONNX model + voices files.
RELEASE_BASE = os.environ.get(
    "KOKORO_RELEASE_BASE",
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0",
)

VOICES_FILE = os.environ.get("KOKORO_VOICES", "voices-v1.0.bin")

# Selectable model variants: filename -> human description (with approx size).
MODEL_CHOICES: dict[str, str] = {
    "kokoro-v1.0.fp16.onnx": "fp16 — recommended balance (~177 MB)",
    "kokoro-v1.0.onnx": "full precision — best quality (~325 MB)",
    "kokoro-v1.0.int8.onnx": "int8 quantized — smallest, fastest (~92 MB)",
}

ProgressFn = Callable[[int, int], None]


def model_label(filename: str) -> str:
    desc = MODEL_CHOICES.get(filename)
    return f"{filename} — {desc}" if desc else filename


def filename_from_label(label: str) -> str:
    return label.split(" — ", 1)[0].strip()


def download_file(url: str, dest: str | Path, progress: ProgressFn | None = None,
                  chunk: int = 1 << 20) -> Path:
    """Stream `url` to `dest` atomically (.part then rename). Calls progress(done, total)."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "ruslanmv-audio/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        with open(tmp, "wb") as fh:
            while True:
                block = resp.read(chunk)
                if not block:
                    break
                fh.write(block)
                done += len(block)
                if progress:
                    progress(done, total)
    tmp.replace(dest)
    return dest


def ensure_models(model: str = "kokoro-v1.0.fp16.onnx", voices: str = VOICES_FILE,
                  root: str | Path = ".", progress: ProgressFn | None = None
                  ) -> list[tuple[str, Path, str]]:
    """Download the model + voices files into `root` if they are missing.

    Returns a list of (filename, path, status) where status is "present" or
    "downloaded".
    """
    root = Path(root)
    results: list[tuple[str, Path, str]] = []
    for name in (Path(model).name, Path(voices).name):
        dest = root / name
        if dest.exists() and dest.stat().st_size > 0:
            results.append((name, dest, "present"))
            continue
        download_file(f"{RELEASE_BASE}/{name}", dest, progress=progress)
        results.append((name, dest, "downloaded"))
    return results


if __name__ == "__main__":
    import sys

    model = sys.argv[1] if len(sys.argv) > 1 else "kokoro-v1.0.fp16.onnx"

    def _cli(done: int, total: int) -> None:
        mb = done / (1 << 20)
        tot = total / (1 << 20) if total else 0
        end = "\n" if total and done >= total else "\r"
        print(f"  {mb:6.1f} / {tot:6.1f} MB", end=end, flush=True)

    for name, path, status in ensure_models(model, progress=_cli):
        print(f"{status}: {path}")
