# Makefile for building ruslanmv.com locally with Jekyll

# Ruby / Bundler / Jekyll configuration
RUBY_VERSION ?= 3.1.4
BUNDLE_EXEC ?= bundle exec

# Jekyll environment and variables
JEKYLL_ENV ?= development
BASEURL ?= /
BUILD_DIR ?= _site

# OS Detection
OS := $(shell uname -s 2>/dev/null || echo Windows)

.PHONY: all install install-deps build serve production clean help

all: build

help:
	@echo "Makefile for ruslanmv.com"
	@echo "  make install      Auto-detect OS and install Ruby + Gems"
	@echo "  make serve        Serve site locally"
	@echo "  make build        Build static site"

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
	@chmod +x scripts/install_macos.sh
	@./scripts/install_macos.sh
else ifeq ($(OS),Linux)
	@if [ -f /etc/debian_version ]; then \
		chmod +x scripts/install_ubuntu.sh; \
		./scripts/install_ubuntu.sh; \
	elif [ -f /etc/redhat-release ]; then \
		chmod +x scripts/install_fedora.sh; \
		./scripts/install_fedora.sh; \
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
	@echo "Serving site at http://localhost:4000$(BASEURL)..."
	JEKYLL_ENV=$(JEKYLL_ENV) GITHUB_PAGES=false $(BUNDLE_EXEC) jekyll serve --livereload --baseurl "$(BASEURL)"

production:
	$(MAKE) build JEKYLL_ENV=production

clean:
	rm -rf "$(BUILD_DIR)" .jekyll-cache vendor .bundle
