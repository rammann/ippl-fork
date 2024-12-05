#!/bin/bash

NUM_PROCESSORS=${1:-4}

# Extract the root directory based on the script's location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="$ROOT_DIR/build"
UNIT_TEST_DIR="$BUILD_DIR/unit_tests/OrthoTree"

# Create and navigate to the build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR" || exit 1

# Configure the build with CMake
cmake "$ROOT_DIR" -DCMAKE_CXX_STANDARD=20 -DENABLE_TESTS=True -DKokkos_VERSION=4.2.00 -DENABLE_UNIT_TESTS=True

# Navigate to the unit test directory
cd "$UNIT_TEST_DIR" || exit 1

# Build the tests
if make -j$(nproc); then
    echo "Unit tests build successful."
else
    echo "Unit tests build failed."
    exit 1
fi

# Execute all test executables in the unit test directory with mpiexec
for test_executable in "$UNIT_TEST_DIR"/*; do
    if [[ -f "$test_executable" && -x "$test_executable" ]]; then
        if [[ "$(basename "$test_executable")" == "parallel_tree_test" ]]; then
            PROCS=4
        else
            PROCS=$NUM_PROCESSORS
        fi

        echo "Running test with $PROCS processors: $test_executable"
        mpiexec -n "$PROCS" "$test_executable"

        if [[ $? -ne 0 ]]; then
            echo "Test $test_executable failed."
            exit 1
        fi
    fi
done

