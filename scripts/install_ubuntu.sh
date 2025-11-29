#!/bin/bash
echo "🐧 Detected Debian/Ubuntu based system..."

# Update package lists
sudo apt-get update

# Install Ruby, Bundler dependencies, and Build Tools
# build-essential is required for compiling native extensions (GCC, Make)
sudo apt-get install -y ruby-full build-essential zlib1g-dev

echo "✅ Ruby and build tools installed."
ruby --version || echo "⚠️ Ruby not found on PATH after installation."
