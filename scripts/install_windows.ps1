Write-Host "🪟 Detected Windows..."

# Check for Chocolatey
if (Get-Command choco -ErrorAction SilentlyContinue) {
    Write-Host "Installing Ruby via Chocolatey..."
    choco install ruby -y
    Refreshenv
} else {
    Write-Host "Chocolatey not found. Please download the Ruby Installer manually:"
    Write-Host "👉 https://rubyinstaller.org/downloads/"
}