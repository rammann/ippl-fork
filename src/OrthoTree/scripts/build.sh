#!/bin/bash

# Extract the root directory based on the script's location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)" # Adjusted to move one level further up
BUILD_DIR="$ROOT_DIR/build"

# Create and navigate to the build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR" || exit 1

# Configure the build with CMake
cmake "$ROOT_DIR" -DCMAKE_CXX_STANDARD=20 -DENABLE_TESTS=True -DKokkos_VERSION=4.2.00 -DENABLE_UNIT_TESTS=True
