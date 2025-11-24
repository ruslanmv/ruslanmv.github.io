# Makefile for Jekyll Blog Site Management
# ruslanmv.github.io

.PHONY: help install serve build clean deploy doctor

# Default target
.DEFAULT_GOAL := help

##@ General

help: ## Display this help message
	@echo "Jekyll Blog - ruslanmv.github.io"
	@echo "=================================="
	@echo ""
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)
	@echo ""

##@ Development

install: ## Install Ruby dependencies (bundle install)
	@echo "Installing Ruby dependencies..."
	bundle install --path vendor/bundle
	@echo "✓ Dependencies installed successfully!"

update: ## Update Ruby dependencies (bundle update)
	@echo "Updating Ruby dependencies..."
	bundle update
	@echo "✓ Dependencies updated successfully!"

serve: ## Start Jekyll development server (http://127.0.0.1:4001)
	@echo "Starting Jekyll development server..."
	@echo "Server will be available at: http://127.0.0.1:4001"
	bundle exec jekyll serve --host 127.0.0.1 --port 4001

serve-drafts: ## Start Jekyll server with drafts enabled
	@echo "Starting Jekyll server with drafts..."
	bundle exec jekyll serve --host 127.0.0.1 --port 4001 --drafts

serve-live: ## Start Jekyll server with live reload
	@echo "Starting Jekyll server with live reload..."
	bundle exec jekyll serve --host 127.0.0.1 --port 4001 --livereload

##@ Build

build: ## Build the Jekyll site for production
	@echo "Building Jekyll site for production..."
	JEKYLL_ENV=production bundle exec jekyll build
	@echo "✓ Site built successfully in _site/"

build-dev: ## Build the Jekyll site for development
	@echo "Building Jekyll site for development..."
	bundle exec jekyll build
	@echo "✓ Site built successfully in _site/"

##@ Maintenance

clean: ## Clean generated site files
	@echo "Cleaning generated files..."
	bundle exec jekyll clean
	rm -rf _site .jekyll-cache .jekyll-metadata
	@echo "✓ Clean complete!"

clean-all: clean ## Clean all generated files including vendor
	@echo "Removing vendor directory..."
	rm -rf vendor/bundle
	@echo "✓ Deep clean complete!"

doctor: ## Run Jekyll doctor to check for issues
	@echo "Running Jekyll doctor..."
	bundle exec jekyll doctor

##@ Git Operations

status: ## Show git status
	@git status

commit: ## Commit changes (use msg="your message")
	@if [ -z "$(msg)" ]; then \
		echo "Error: Please provide a commit message using msg=\"your message\""; \
		exit 1; \
	fi
	@git add .
	@git commit -m "$(msg)"
	@echo "✓ Changes committed!"

push: ## Push changes to remote
	@git push
	@echo "✓ Changes pushed to remote!"

##@ Utilities

new-post: ## Create a new blog post (use title="Post Title")
	@if [ -z "$(title)" ]; then \
		echo "Error: Please provide a title using title=\"Your Post Title\""; \
		exit 1; \
	fi
	@DATE=$$(date +%Y-%m-%d); \
	SLUG=$$(echo "$(title)" | tr '[:upper:]' '[:lower:]' | sed 's/ /-/g' | sed 's/[^a-z0-9-]//g'); \
	FILE="_posts/$$DATE-$$SLUG.md"; \
	echo "---" > $$FILE; \
	echo "layout: post" >> $$FILE; \
	echo "title: \"$(title)\"" >> $$FILE; \
	echo "date: $$DATE" >> $$FILE; \
	echo "categories: []" >> $$FILE; \
	echo "tags: []" >> $$FILE; \
	echo "---" >> $$FILE; \
	echo "" >> $$FILE; \
	echo "Content goes here..." >> $$FILE; \
	echo "✓ Created new post: $$FILE"

check-links: ## Check for broken links in the site
	@echo "Checking for broken links..."
	@if command -v htmlproofer > /dev/null; then \
		htmlproofer ./_site --disable-external; \
	else \
		echo "htmlproofer not installed. Install with: gem install html-proofer"; \
	fi

##@ Information

version: ## Display Jekyll and Ruby versions
	@echo "Ruby version:"
	@ruby --version
	@echo ""
	@echo "Bundler version:"
	@bundle --version
	@echo ""
	@echo "Jekyll version:"
	@bundle exec jekyll --version

deps: ## List installed dependencies
	@bundle list
