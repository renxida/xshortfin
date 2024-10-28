
#shortfin tokens
# [    1 29500 29532 29500 29536 29500 29556 29500 29562 29500 29561 29500     0     0     0     0]
# 1 is eos token

iree-run-module \
     --module=/tmp/sharktank/llama/model.vmfb \
     --parameters=model=/tmp/sharktank/llama/open-llama-3b-v2-f16.gguf \
     --function=prefill_bs1 \
     --device=hip \
     --input=1x12xi64=1,29500,29532,29500,29536,29500,29556,29500,29562,29500,29561,29500\
     --input=1xi64=12 \
     --input=1x1xi64=255 \
     --input=256x2662400xf16=0 \
     --output=@/tmp/logits.npy


iree-run-module \
     --module=/tmp/sharktank/llama/model.vmfb \
     --parameters=model=/tmp/sharktank/llama/open-llama-3b-v2-f16.gguf \
     --function=prefill_bs1 \
     --device=hip \
     --input=1x16xi64=1,29500,29532,29500,29536,29500,29556,29500,29562,29500,29561,29500,0,0,0,0\
     --input=1xi64=16 \
     --input=1x1xi64=255 \
     --input=256x2662400xf16=0 \
     --output=@/tmp/logits.npy



# analyze iree logits
import numpy as np
logits = np.load('/tmp/logits.npy')
import pandas as pd

def extract_tokens_from_logits(
    logits: np.ndarray, seq_lens: list[int]
) -> list[int]:
    """Extracts tokens from a batch of logits (B, S, D).
    
    Args:
        logits: NumPy array of shape (batch_size, sequence_length, dimension)
        seq_lens: List of sequence lengths for each batch
        
    Returns:
        List of token indices extracted from the logits
        
    The length of seq_lens must be equal to the batch size.
    Note that there are ways to code the indexing as array operations
    but it just creates a bunch of weirdly shaped little work.
    Statically looping like this is more efficient.
    """
    bs, *_ = logits.shape
    assert len(seq_lens) == bs
    results = []
    for batch, seq_len in enumerate(seq_lens):
        step_logits = logits[batch, seq_len - 1]
        results.append(np.argmax(step_logits))
    return results

extract_tokens_from_logits(logits, [12])
