#!/bin/bash
# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#BS="1,4"
BS="1,4"

echo "Exporting with batch size $BS"



pkill -9 -f "python -m shortfin_apps.llm.server" || true


echo "Serving with batch size $BS"

WD=/tmp/export_and_serve
# Start the server in the background and save its PID
python -m shortfin_apps.llm.server \
  --tokenizer=$WD/tokenizer.json \
  --model_config=$WD/edited_config.json \
  --vmfb=$WD/model.vmfb \
  --parameters=$WD/open-llama-3b-v2-f16.gguf \
  --device=hip &

SERVER_PID=$!

# Wait a bit for the server to start up
sleep 5

# Run the client
python ~/SHARK-Platform/shortfin/python/shortfin_apps/llm/client.py

# Kill the server
kill $SERVER_PID

# Wait for the server to shut down
wait $SERVER_PID
