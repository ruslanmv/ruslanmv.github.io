# Makefile for building ruslanmv.com locally with Jekyll

# Ruby / Bundler / Jekyll configuration
RUBY_VERSION ?= 3.1.4
BUNDLE_EXEC ?= bundle exec

# Jekyll environment and variables
JEKYLL_ENV ?= development
# The site is served at the domain root (no baseurl in _config.yml), so default
# to empty — avoids the "http://localhost:4000//" double slash on `make serve`.
BASEURL ?=
BUILD_DIR ?= _site

# OS Detection
OS := $(shell uname -s 2>/dev/null || echo Windows)

# ---------------------------------------------------------
#  Audio pipeline configuration
# ---------------------------------------------------------
# Local R2 credentials (and any overrides) are read from a git-ignored .env
# file. Copy .env.example to .env and fill it in. Never commit .env.
-include .env

PY            ?= python3
VENV          ?= .venv
VENV_PY       := $(VENV)/bin/python
AWS           ?= $(VENV)/bin/aws
# Prefer uv (fast, modern Python package manager). Detected on PATH here; if it
# is missing it is bootstrapped into the venv automatically (see audio-deps),
# with a plain pip fallback as a last resort.
UV_SYS        := $(shell command -v uv 2>/dev/null)

AUDIO_OUT_DIR ?= public/audio
MANIFEST_PATH ?= _data/audio_manifest.json
# Public base URL the site links to. Default matches the committed manifest so
# `make serve` simulates production (audio streamed from R2).
R2_PUBLIC_BASE_URL ?= https://pub-18ecc6bab6074b2e89efa5c36d39a544.r2.dev

# Master switch: `make serve AUDIO=0` serves without touching audio/R2.
AUDIO ?= 1

# Export R2 settings so the Python pipeline and AWS CLI see them in recipes.
export R2_ACCOUNT_ID R2_BUCKET R2_ACCESS_KEY_ID R2_SECRET_ACCESS_KEY
export R2_PUBLIC_BASE_URL AUDIO_OUT_DIR MANIFEST_PATH

.PHONY: all install install-deps build serve production clean help \
        audio-venv audio-deps audio-system-deps audio-models audio-ui \
        audio audio-force audio-check audio-upload ingest-audio audio-clean

all: build

help:
	@echo "Makefile for ruslanmv.com"
	@echo ""
	@echo "  Site:"
	@echo "    make install        Auto-detect OS and install Ruby + Gems"
	@echo "    make serve          Serve locally; builds missing audio + uploads to R2"
	@echo "                        (simulates production; 'make serve AUDIO=0' to skip)"
	@echo "    make build          Build static site"
	@echo ""
	@echo "  Audio pipeline (essay/blog narration):"
	@echo "    make audio          Generate only missing/changed MP3s (content-hash cached)"
	@echo "    make audio-force    Regenerate every MP3"
	@echo "    make audio-upload   Submit generated MP3s to Cloudflare R2"
	@echo "    make ingest-audio   audio + audio-upload (the full local pipeline)"
	@echo "    make audio-ui       Launch the speaker-setup UI (Gradio, in the venv)"
	@echo "    make audio-check FILE=_posts/x.md   Preview the narrated text"
	@echo "    make audio-models   Download the Kokoro model files"
	@echo "    make audio-system-deps   Install ffmpeg + espeak-ng"
	@echo "    make audio-clean    Remove local audio, venv, and model files"

# ---------------------------------------------------------
#  Target: install
# ---------------------------------------------------------
install:
	@echo "🔍 Checking system dependencies for $(OS)..."
	@$(MAKE) install-os-dep

	@echo "💎 Checking / installing Bundler..."
	@if ! command -v gem >/dev/null 2>&1; then \
		echo "❌ 'gem' command not found. Ruby installation may have failed."; \
		exit 1; \
	fi
	@if ! command -v bundle >/dev/null 2>&1; then \
		echo "➡️  Bundler not found, installing..."; \
		gem install bundler --no-document || { \
			echo "⚠️  Permission denied. Retrying with sudo..."; \
			sudo gem install bundler --no-document; \
		}; \
	else \
		echo "✅ Bundler already installed."; \
	fi

	@echo "⚙️  Configuring Bundler to install gems locally (vendor/bundle)..."
	@bundle config set --local path 'vendor/bundle'

	@echo "📦 Installing Project Dependencies..."
	@bundle install

# Internal target to handle OS-specific script execution
install-os-dep:
ifeq ($(OS),Darwin)
	@tr -d '\r' < scripts/install_macos.sh | bash
else ifeq ($(OS),Linux)
	@if [ -f /etc/debian_version ]; then \
		tr -d '\r' < scripts/install_ubuntu.sh | bash; \
	elif [ -f /etc/redhat-release ]; then \
		tr -d '\r' < scripts/install_fedora.sh | bash; \
	else \
		echo "⚠️  Linux distro not auto-detected. Please run script manually."; \
	fi
else
	@echo "⚠️  Windows detected. Run 'scripts/install_windows.ps1' if needed."
endif

# ---------------------------------------------------------
#  Standard Jekyll Targets
# ---------------------------------------------------------

build:
	@echo "Building site into ./$(BUILD_DIR)..."
	JEKYLL_ENV=$(JEKYLL_ENV) GITHUB_PAGES=true $(BUNDLE_EXEC) jekyll build --baseurl "$(BASEURL)" --destination "$(BUILD_DIR)"

serve:
	@if [ "$(AUDIO)" != "0" ]; then \
		echo "🎙️  Preparing audio (set AUDIO=0 to skip)..."; \
		$(MAKE) ingest-audio; \
	fi
	@echo "Serving site at http://localhost:4000$(BASEURL)..."
	JEKYLL_ENV=$(JEKYLL_ENV) GITHUB_PAGES=false $(BUNDLE_EXEC) jekyll serve --livereload --baseurl "$(BASEURL)"

production:
	$(MAKE) build JEKYLL_ENV=production

clean:
	rm -rf "$(BUILD_DIR)" .jekyll-cache vendor .bundle

# ---------------------------------------------------------
#  Audio pipeline targets
# ---------------------------------------------------------

# Create the Python virtualenv used by the audio scripts.
# Create the Python venv — with uv when available (fast), else stdlib venv.
audio-venv:
	@if [ -n "$(UV_SYS)" ]; then \
		test -d "$(VENV)" || { echo "⚡ Creating venv at $(VENV) with uv..."; "$(UV_SYS)" venv "$(VENV)"; }; \
	else \
		test -d "$(VENV)" || { echo "🐍 Creating venv at $(VENV)..."; $(PY) -m venv "$(VENV)"; }; \
	fi

# Install Python deps (generation + UI + AWS CLI, from pyproject.toml). Uses uv
# for fast installs: a system uv if present, otherwise one bootstrapped into the
# venv; falls back to pip only if uv cannot be obtained.
audio-deps: audio-venv
	@set -e; \
	if [ -n "$(UV_SYS)" ]; then UV="$(UV_SYS)"; \
	elif [ -x "$(VENV)/bin/uv" ]; then UV="$(VENV)/bin/uv"; \
	else echo "⚡ Bootstrapping uv into the venv..."; \
		"$(VENV_PY)" -m pip install -q uv >/dev/null 2>&1 && UV="$(VENV)/bin/uv" || UV=""; \
	fi; \
	if [ -n "$$UV" ]; then \
		echo "⚡ Installing deps with uv..."; \
		"$$UV" pip install -q --python "$(VENV_PY)" ".[ui,upload]"; \
	else \
		echo "ℹ️  uv unavailable — falling back to pip."; \
		"$(VENV_PY)" -m pip install -q --upgrade pip; \
		"$(VENV_PY)" -m pip install -q ".[ui,upload]"; \
	fi
	@command -v ffmpeg    >/dev/null 2>&1 || echo "⚠️  ffmpeg not found — run 'make audio-system-deps' (needed to encode MP3s)."
	@command -v espeak-ng >/dev/null 2>&1 || echo "⚠️  espeak-ng not found — run 'make audio-system-deps'."

# Launch the Gradio speaker-setup app in the project venv (so kokoro-onnx and
# friends are always available — never the system Python).
audio-ui: audio-deps
	@echo "🎙️  Opening speaker setup at http://127.0.0.1:7860 (Ctrl-C to stop)..."
	@$(VENV_PY) scripts/setup_audio.py

# Install the system tools the TTS pipeline needs.
audio-system-deps:
ifeq ($(OS),Darwin)
	brew install ffmpeg espeak-ng
else ifeq ($(OS),Linux)
	sudo apt-get update && sudo apt-get install -y ffmpeg espeak-ng
else
	@echo "⚠️  Please install ffmpeg and espeak-ng manually."
endif

# Download the Kokoro model files (no-op if already present).
audio-models: audio-deps
	@$(VENV_PY) scripts/generate_audio.py --ensure-models

# Generate ONLY missing/changed audio. The content hash in the manifest is the
# cache key, so unchanged essays are skipped instantly (models are fetched
# automatically the first time they are actually needed).
audio: audio-deps
	@echo "🎧 Generating missing/changed audio..."
	@$(VENV_PY) scripts/generate_audio.py --all

# Force-regenerate every MP3.
audio-force: audio-deps
	@$(VENV_PY) scripts/generate_audio.py --all --force

# Preview the narrated text for one file:  make audio-check FILE=_posts/x.md
audio-check: audio-deps
	@test -n "$(FILE)" || { echo "Usage: make audio-check FILE=_posts/your-post.md"; exit 1; }
	@$(VENV_PY) scripts/generate_audio.py "$(FILE)" --check-tags

# Submit locally generated MP3s to Cloudflare R2. Self-skips if R2 is not
# configured in .env, so it is safe to call unconditionally.
audio-upload: audio-deps
	@if [ -z "$$R2_BUCKET" ] || [ -z "$$R2_ACCESS_KEY_ID" ] || [ -z "$$R2_SECRET_ACCESS_KEY" ] || [ -z "$$R2_ACCOUNT_ID" ]; then \
		echo "⏭  R2 not configured (set R2_* in .env) — skipping upload. Audio stays local."; \
	elif [ ! -d "$(AUDIO_OUT_DIR)/essays" ]; then \
		echo "ℹ️  No local audio in $(AUDIO_OUT_DIR)/essays yet — run 'make audio' first."; \
	else \
		echo "☁️  Uploading $(AUDIO_OUT_DIR)/essays → s3://$$R2_BUCKET/essays ..."; \
		AWS_ACCESS_KEY_ID="$$R2_ACCESS_KEY_ID" AWS_SECRET_ACCESS_KEY="$$R2_SECRET_ACCESS_KEY" AWS_DEFAULT_REGION=auto \
		"$(AWS)" s3 sync "$(AUDIO_OUT_DIR)/essays" "s3://$$R2_BUCKET/essays" \
			--endpoint-url "https://$$R2_ACCOUNT_ID.r2.cloudflarestorage.com" \
			--content-type audio/mpeg \
			--cache-control "public, max-age=31536000, immutable" \
			--size-only --exclude "*" --include "*.mp3"; \
		echo "✅ R2 upload complete."; \
	fi

# Full local pipeline: generate what is missing, then submit to R2.
ingest-audio: audio audio-upload

# Remove local audio artifacts, the venv, and downloaded model files.
audio-clean:
	rm -rf "$(AUDIO_OUT_DIR)" "$(VENV)" kokoro-*.onnx voices-*.bin
