#!/bin/bash
set -e

echo "=============================================================="
echo " Home_Reconstruction clean environment setup"
echo "=============================================================="

PROJECT_ROOT="/workspace/Home_Reconstruction"
VENV_PATH="$PROJECT_ROOT/venv"
TOKEN_FILE="$PROJECT_ROOT/.tokens"

# -------------------------
# Handle active venv
# -------------------------
if [ -n "$VIRTUAL_ENV" ]; then
    if type deactivate &>/dev/null; then
        echo "Deactivating active venv..."
        deactivate
    else
        echo "⚠ Active venv detected ($VIRTUAL_ENV), cannot deactivate automatically."
        echo "   It's recommended to deactivate manually before running this script."
    fi
fi

# -------------------------
# System dependencies
# -------------------------
echo "[1/5] Installing system dependencies..."
export DEBIAN_FRONTEND=noninteractive

apt-get update -qq
apt-get install -y \
    git \
    unzip \
    python3-venv \
    libgl1 \
    libglib2.0-0

echo "✓ System dependencies installed"
echo ""

# -------------------------
# Create clean venv
# -------------------------
echo "[2/5] Creating clean virtual environment..."
cd "$PROJECT_ROOT"

if [ -d "$VENV_PATH" ]; then
    echo "  Removing existing venv..."
    rm -rf "$VENV_PATH"
fi

python3 -m venv venv --system-site-packages
source "$VENV_PATH/bin/activate"

pip install --upgrade pip setuptools wheel
echo "✓ Virtual environment created and activated"
echo ""

# -------------------------
# Install Python dependencies
# -------------------------
echo "[3/5] Installing Python dependencies..."
pip install -r requirements.txt
echo "✓ Python dependencies installed"
echo ""

# -------------------------
# Install SAM3
# -------------------------
echo "[4/5] Installing SAM3..."
cd /workspace
if [ ! -d "sam3" ]; then
    git clone https://github.com/facebookresearch/sam3.git
fi
cd sam3
pip install -e .
echo "✓ SAM3 installed"
echo ""

# -------------------------
# Make venv + tokens auto-activate
# -------------------------
echo "[5/5] Making venv + tokens auto-activate in new terminals..."

AUTO_ACTIVATE_SCRIPT="# Home_Reconstruction venv auto-activation
if [ -f \"$VENV_PATH/bin/activate\" ]; then
    source \"$VENV_PATH/bin/activate\"
fi
if [ -f \"$TOKEN_FILE\" ]; then
    export \$(grep -v '^#' \"$TOKEN_FILE\" | xargs)
fi
"

# Add to .bashrc if not already added
if ! grep -Fxq "# Home_Reconstruction venv auto-activation" ~/.bashrc; then
    echo "$AUTO_ACTIVATE_SCRIPT" >> ~/.bashrc
    echo "✓ Added venv auto-activation to ~/.bashrc"
else
    echo "✓ Venv auto-activation already in ~/.bashrc"
fi

echo ""
echo "=============================================================="
echo " Setup complete."
echo " - Venv and tokens will now auto-activate in every new terminal."
echo "=============================================================="
