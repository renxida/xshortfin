#!/bin/bash

WD=$HOME/xshortfin/goldens/exported_llama_model


# Define absolute paths
TARGET_FILE="/home/$USER/SHARK-Platform/sharktank/sharktank/examples/paged_llm_v1.py"


# Check if target file exists
if [ ! -f "$TARGET_FILE" ]; then
    echo "Error: Target file $TARGET_FILE not found"
    exit 1
fi

# Create backup
cp "$TARGET_FILE" "$TARGET_FILE.bak"

# Apply the patch
patch "$TARGET_FILE" ./sharktank_dump.patch

# Check if patch was successful
if [ $? -eq 0 ]; then
    echo "Patch applied successfully"
    echo "Backup saved as $TARGET_FILE.bak"
else
    echo "Patch failed, restoring from backup"
    mv "$TARGET_FILE.bak" "$TARGET_FILE"
    exit 1
fi

python -m sharktank.examples.paged_llm_v1 --gguf-file=$WD/open-llama-3b-v2-f16.gguf "1 2 3 4 5 " --tokenizer-config-json $WD/tokenizer_config.json


# restore the original file
mv "$TARGET_FILE.bak" "$TARGET_FILE"