#!/bin/bash
# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#BS="1,4"
BS="1,4"

echo "Exporting with batch size $BS"
set -xeuo pipefail

WD=$HOME/xshortfin/goldens/exported_llama_model
rm -rf $WD

mkdir -p $WD

huggingface-cli download --local-dir $WD SlyEcho/open_llama_3b_v2_gguf open-llama-3b-v2-f16.gguf

# save an irpa copy
python -m sharktank.tools.dump_gguf --gguf-file $WD/open-llama-3b-v2-f16.gguf --save $WD/open-llama-3b-v2-f16.irpa

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

