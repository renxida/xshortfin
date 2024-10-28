# Prefill
iree-run-module \
  --module=/tmp/sharktank/llama/model.vmfb \
  --function=prefill_bs4 \
  --device=hip \
  --input=4x1xi64=0 \
  --input=4xi64=1 \
  --input=4x1xi64=0,0,0,0 \
  --input=1x2662400xf16 \
  --parameters=model=/tmp/sharktank/llama/open-llama-3b-v2-f16.gguf
iree-run-module   --module=/tmp/sharktank/llama/model.vmfb   --function=prefill_bs4   --device=hip   --input=4x1xi64=0   --input=4xi64=1   --input=4x1xi64=0,0,0,0   --input=1x2662400xf16   --parameters=model=/tmp/sharktank/llama/open-llama-3b-v2-f16.gguf --output=+a.npy

iree-compile model_edited.mlir --iree-hal-target-backends=rocm --iree-hip-target=gfx1100 -o model_edited.vmfb
iree-run-module   --module=/tmp/sharktank/llama/model.vmfb   --function=prefill_bs4   --device=hip   --input=4x1xi64=0   --input=4xi64=1   --input=4x1xi64=0,0,0,0   --input=1x2662400xf16   --parameters=model=/tmp/sharktank/llama/open-llama-3b-v2-f16.gguf --output=+a.npy