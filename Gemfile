source "https://rubygems.org"

# Gemfile for ruslanmv.github.io
#
# This setup mirrors the GitHub Pages environment as closely as possible.
# - `github-pages` pins Jekyll and all core plugins to the same versions
#   that GitHub Pages uses in production.
# - Additional gems are limited to those that GitHub Pages supports and
#   that are required for local development.

# Use the same stack as GitHub Pages
# Check https://pages.github.com/versions/ for the current version.
gem "github-pages", "= 228", group: :jekyll_plugins

# Required by the Minimal Mistakes theme when used as a remote_theme
# Make sure `_config.yml` includes:
#   remote_theme: "mmistakes/minimal-mistakes"
#   plugins:
#     - jekyll-include-cache
gem "jekyll-include-cache", group: :jekyll_plugins

# Required to run `bundle exec jekyll serve` with Ruby 3.x locally
gem "webrick", "~> 1.8"

# Timezone data for Windows and JRuby environments
gem "tzinfo-data", platforms: %i[mingw mswin x64_mingw jruby]

# For Faraday v2 retry middleware (used by some plugins / GitHub API)
gem "faraday-retry"