#!/bin/bash
echo "🎩 Detected Fedora/RedHat based system..."

# Install Ruby and Development Tools
sudo dnf install -y ruby ruby-devel redhat-rpm-config gcc-c++ make

echo "✅ Ruby and dev tools installed."