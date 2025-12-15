#!/bin/bash

# Usage: ./reassemble_and_unpack.sh <input_name>
# Example: ./reassemble_and_unpack.sh my_big_file

set -e

if [ $# -ne 1 ]; then
    echo "Usage: $0 <input_name>"
    exit 1
fi

INPUT_NAME="$1"

# Paths
WORKSPACE="/workspace"
CHUNK_DIR="$WORKSPACE"
DATA_SCENES_DIR="$WORKSPACE/Home_Reconstruction/data_scenes"
TARGET_ZIP="$DATA_SCENES_DIR/${INPUT_NAME}.zip"
TARGET_EXTRACT_DIR="$DATA_SCENES_DIR"

# Ensure output directories exist
mkdir -p "$DATA_SCENES_DIR"
mkdir -p "$TARGET_EXTRACT_DIR"

#Step 1: Combine chunks
echo "Combining chunk files into $TARGET_ZIP..."
cat "$CHUNK_DIR"/chunk_* > "$TARGET_ZIP"

# Step 2: Extract main zip
echo "Unzipping main archive into $TARGET_EXTRACT_DIR..."
unzip -o "$TARGET_ZIP" -d "$TARGET_EXTRACT_DIR"

echo "All zip files extracted."
echo "Final contents are in: $TARGET_EXTRACT_DIR"

