#!/bin/bash
echo "🍎 Detected macOS..."

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo "Error: Homebrew is not installed. Please install it from https://brew.sh/"
    exit 1
fi

# Install Ruby
brew install ruby

# Configure PATH for Homebrew Ruby (often missed!)
# This adds the brew ruby path to shell config if it's not there
SHELL_CONFIG="$HOME/.zshrc"
if [ -f "$HOME/.bashrc" ]; then SHELL_CONFIG="$HOME/.bashrc"; fi

RUBY_PATH="$(brew --prefix ruby)/bin"

if [[ ":$PATH:" != *":$RUBY_PATH:"* ]]; then
    echo "export PATH=\"$RUBY_PATH:\$PATH\"" >> "$SHELL_CONFIG"
    echo "⚠️  Added Ruby to PATH in $SHELL_CONFIG. Please restart your terminal or run: source $SHELL_CONFIG"
fi

echo "✅ Ruby installed via Homebrew."