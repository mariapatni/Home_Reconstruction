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

# Use workspace for temp to avoid memory issues
export TMPDIR=/workspace/tmp
mkdir -p /workspace/tmp

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

# Verify installations
echo "======================================================================"
echo "Verification"
echo "======================================================================"

python3 -c "
import sys
packages = {
    'numpy': 'NumPy',
    'scipy': 'SciPy',
    'PIL': 'Pillow',
    'OpenEXR': 'OpenEXR',
    'plyfile': 'plyfile',
    'plotly': 'Plotly',
    'matplotlib': 'Matplotlib'
}

failed = []
for module, name in packages.items():
    try:
        __import__(module)
        print(f'✓ {name}')
    except ImportError:
        print(f'✗ {name} - FAILED')
        failed.append(name)

if failed:
    print(f'\n⚠️  Installation incomplete. Failed: {', '.join(failed)}')
    sys.exit(1)
else:
    print('\n✓ All packages verified successfully!')
"

echo ""
echo "======================================================================"
echo "Setup Complete!"
echo "======================================================================"
echo ""
echo "Note: Restart your Jupyter kernel to use the new packages"
echo ""