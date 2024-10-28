#!/bin/bash

# Set up environment variables
export PYTHONPATH=/home/xidaren2/SHARK-Platform:$PYTHONPATH

# Define paths
CONFIG_PATH="/tmp/sharktank/llama/config.json"
OUTPUT_DIR="/tmp/inputs"
PROMPT="1 2 3 4 5 "

# Generate input data
echo "Generating input data..."
python -m sharktank.models.llama.tools.generate_data \
  --tokenizer=openlm-research/open_llama_3b_v2 \
  --config="$CONFIG_PATH" \
  --output-dir="$OUTPUT_DIR" \
  --prompt="$PROMPT"

# Check if data generation was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to generate input data."
    exit 1
fi

# Run the chat model script
echo "Running chat model..."
python ~/xshortfin/chat_with_model.py

# Check if the chat model ran successfully
if [ $? -ne 0 ]; then
    echo "Error: Failed to run chat model."
    exit 1
fi

echo "Chat model execution completed successfully."
