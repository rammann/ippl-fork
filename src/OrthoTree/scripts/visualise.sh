#!/bin/bash

# Ensure the user specifies the number of processors
if [ -z "$1" ]; then
    echo "Error: Please specify the number of processors as the first argument."
    echo "Usage: $0 <num_processors>"
    exit 1
fi

NUM_PROCESSORS=$1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Step 1: Run the build script
echo "Running the build script..."
"$SCRIPT_DIR/build.sh" || {
    echo "Build failed. Exiting pipeline."
    exit 1
}

# Step 2: Run the experiment script
echo "Running the experiment with $NUM_PROCESSORS processors..."
"$SCRIPT_DIR/experiment.sh" "$NUM_PROCESSORS" || {
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
