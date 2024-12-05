#!/bin/bash

# Ensure the user specifies the number of processors
if [ -z "$1" ]; then
    echo "Error: Please specify the number of processors as the first argument."
    echo "Usage: $0 <num_processors>"
    exit 1
fi

NUM_PROCESSORS=$1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)/build/test/orthotree"
TEST_EXE="OrthoTreeTest"

# Run the build script
"$SCRIPT_DIR/build.sh" || exit 1

# Navigate to the test directory
cd "$TEST_DIR" || exit 1

# Build the tests
if make -j$(nproc); then
    echo "Experimental build successful."
else
    echo "Experimental build failed."
    exit 1
fi

# Run the test executable
echo "Running test with $NUM_PROCESSORS processors: $TEST_EXE"
mpiexec -n "$NUM_PROCESSORS" "$TEST_EXE"

# Check the test result
if [[ $? -ne 0 ]]; then
    echo "Test $TEST_EXE failed."
    exit 1
fi

echo "Test $TEST_EXE completed successfully."
