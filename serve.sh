#!/bin/bash
# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Create logs directory if it doesn't exist
LOGDIR="serve_logs"
mkdir -p $LOGDIR

# Kill any existing server instances
pkill -9 -f "python -m shortfin_apps.llm.server" || true

# Start the server with the exported model
python -m shortfin_apps.llm.server \
  --tokenizer=/tmp/export_and_serve/tokenizer.json \
  --model_config=/tmp/export_and_serve/edited_config.json \
  --vmfb=/tmp/export_and_serve/model.vmfb \
  --parameters=/tmp/export_and_serve/open-llama-3b-v2-f16.gguf \
  --device=hip |& tee "$LOGDIR/server.log"&

SERVER_PID=$!

# Wait for server startup
sleep 5

# Run the client
python ~/SHARK-Platform/shortfin/python/shortfin_apps/llm/client.py |& tee "$LOGDIR/client.log"

# Cleanup
kill $SERVER_PID
wait $SERVER_PID