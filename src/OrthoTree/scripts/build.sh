#!/bin/bash

# Extract the root directory based on the script's location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)" # Adjusted to move one level further up
BUILD_DIR="$ROOT_DIR/build"

# Detect the maximum number of processors
if command -v nproc &>/dev/null; then
    MAX_PROCS=$(nproc) # Linux
elif [[ "$(uname)" == "Darwin" ]]; then
    MAX_PROCS=$(sysctl -n hw.logicalcpu) # macOS
else
    echo "Unable to determine the number of processors. Defaulting to 1."
    MAX_PROCS=1
fi

echo "Maximum processors detected: $MAX_PROCS"

# Create and navigate to the build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR" || exit 1

# Configure the build with CMake
echo "Configuring the build with CMake..."
cmake "$ROOT_DIR" -DCMAKE_CXX_STANDARD=20 -DENABLE_TESTS=True -DKokkos_VERSION=4.2.00 -DENABLE_UNIT_TESTS=True
