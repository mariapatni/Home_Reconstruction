#!/bin/bash
# Setup script for Home_Reconstruction project
# Installs system dependencies and Python packages (no venv)
set -e

echo "======================================================================"
echo "Home_Reconstruction Setup"
echo "======================================================================"
echo ""

# Prevent interactive prompts
export DEBIAN_FRONTEND=noninteractive

# System dependencies
echo "[1/3] Installing system dependencies..."
apt-get update -qq
apt-get install -y unzip python3-scipy python3-numpy

echo "✓ System dependencies installed"
echo ""

# Clean up any broken pip installations
echo "[2/3] Cleaning up previous installations..."
pip uninstall -y scipy matplotlib plotly 2>/dev/null || true

echo "✓ Cleanup complete"
echo ""

# Python dependencies (one at a time to avoid memory issues)
echo "[3/3] Installing Python packages..."

# Install packages one by one with error handling
packages=(
    "Pillow"
    "OpenEXR"
    "plyfile"
    "plotly"
    "matplotlib"
)

for package in "${packages[@]}"; do
    echo "  Installing $package..."
    pip install --no-cache-dir --break-system-packages "$package" || {
        echo "  ⚠️  Failed to install $package, trying again..."
        pip install --no-cache-dir --break-system-packages --force-reinstall "$package"
    }
done

echo ""
echo "✓ All packages installed"
echo ""