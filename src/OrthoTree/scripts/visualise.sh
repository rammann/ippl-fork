#!/bin/bash

# Ensure the user specifies the number of processors
if [ -z "$1" ]; then
    echo "Error: Please specify the number of processors as the first argument."
    echo "Usage: $0 <num_processors> [additional_args...]"
    exit 1
fi

NUM_PROCESSORS=$1
shift # Remove the first argument, so $@ now contains only additional arguments

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Ensure the output folder exists and clear its contents
if [ -d "$SCRIPT_DIR/output" ]; then
    rm -rf "$SCRIPT_DIR/output"/*
else
    mkdir -p "$SCRIPT_DIR/output"
fi

# Step 2: Run the experiment script with additional arguments
echo "Running the experiment with $NUM_PROCESSORS processors..."
"$SCRIPT_DIR/experiment.sh" "$NUM_PROCESSORS" "$@" || {
    echo "Experiment failed. Exiting pipeline."
    exit 1
}

# Step 3: Run the visualization script
echo "Running the visualization script..."
python3 "$SCRIPT_DIR/visualise.py" || {
    echo "Visualization failed. Exiting pipeline."
    exit 1
}

echo "Pipeline completed successfully!"
