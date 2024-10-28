#!/bin/bash
# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#BS="1,4"
BS="1,4"

# Create logs directory
LOGDIR="export_and_serve_logs"
mkdir -p $LOGDIR

echo "Exporting with batch size $BS"
set -xeuo pipefail

WD=/tmp/export_and_serve
rm -rf $WD

pkill -9 -f "python -m shortfin_apps.llm.server" || true

mkdir -p $WD

huggingface-cli download --local-dir $WD SlyEcho/open_llama_3b_v2_gguf open-llama-3b-v2-f16.gguf

HUGGING_FACE_TOKENIZER="openlm-research/open_llama_3b_v2"

python - <<EOF
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("${HUGGING_FACE_TOKENIZER}")
tokenizer.save_pretrained("$WD")
EOF

python -m sharktank.examples.export_paged_llm_v1 \
  --gguf-file="$WD/open-llama-3b-v2-f16.gguf" \
  --output-mlir="$WD/model.mlir" \
  --output-config="$WD/config.json" \
  --bs=$BS

iree-compile "$WD/model.mlir" \
  --iree-hal-target-backends=rocm \
  --iree-hip-target=gfx1100 \
  -o $WD/model.vmfb

# Write the JSON configuration to edited_config.json
cat > $WD/edited_config.json << EOF
{
    "module_name": "module",
    "module_abi_version": 1,
    "max_seq_len": 2048,
    "attn_head_count": 32,
    "attn_head_dim": 100,
    "prefill_batch_sizes": [
        $BS
    ],
    "decode_batch_sizes": [
        $BS
    ],
    "transformer_block_count": 26,
    "paged_kv_cache": {
        "block_seq_stride": 16,
        "device_block_count": 256
    }
}
EOF

echo "Serving with batch size $BS"

# Start the server in the background and redirect output to log file
python -m shortfin_apps.llm.server \
  --tokenizer=$WD/tokenizer.json \
  --model_config=$WD/edited_config.json \
  --vmfb=$WD/model.vmfb \
  --parameters=$WD/open-llama-3b-v2-f16.gguf \
  --device=hip > "$LOGDIR/server.log" 2>&1 &

SERVER_PID=$!

# Wait a bit for the server to start up
sleep 5

# Run the client and redirect output to log file
python ~/SHARK-Platform/shortfin/python/shortfin_apps/llm/client.py > "$LOGDIR/client.log" 2>&1

# Kill the server
kill $SERVER_PID

# Wait for the server to shut down
wait $SERVER_PID