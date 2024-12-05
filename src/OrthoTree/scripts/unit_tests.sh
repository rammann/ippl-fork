#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIT_TEST_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)/build/unit_tests/OrthoTree"

# Function to execute test executables
run_tests() {
    local test_dir=$1
    echo "Running tests in: $test_dir"

    for test_executable in "$test_dir"/*; do
        if [[ -f "$test_executable" && -x "$test_executable" ]]; then
            if [[ "$(basename "$test_executable")" == "parallel_tree_test" ]]; then
                PROCS=4
            else
                PROCS=1
            fi

            echo "Running test with $PROCS processors: $test_executable"
            mpiexec -n "$PROCS" "$test_executable"

            if [[ $? -ne 0 ]]; then
                echo "Test $test_executable failed."
                exit 1
            fi
        fi
    done
}

cd "$UNIT_TEST_DIR" || exit 1
if make -j$(nproc); then
    echo "Unit tests build successful."
else
    echo "Unit tests build failed."
    exit 1
fi

# Run tests in each subfolder and directly in unit_tests/OrthoTree
for subfolder in "balancing" "helpers" "parallel_construction"; do
    run_tests "$UNIT_TEST_DIR/$subfolder"
done

# run remaining tests
run_tests "$UNIT_TEST_DIR"
