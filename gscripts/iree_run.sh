#!/bin/bash
set -e  # Exit on any error

# Function to convert cache state to fp16 and copy
convert_and_copy_cache() {
    local src_path=$1
    local dst_path=$2
    
    echo "Converting and copying cache state from $src_path to $dst_path"
    python3 -c "
import numpy as np
cache = np.load('$src_path')
cache_fp16 = cache.astype(np.float16)
np.save('$dst_path', cache_fp16)
"
}

# Clean and create iree_run directory
rm -rf /home/xidaren2/xshortfin/goldens/iree_run
mkdir -p /home/xidaren2/xshortfin/goldens/iree_run/prefill_inputs
mkdir -p /home/xidaren2/xshortfin/goldens/iree_run/prefill_outputs

# Copy VMFB and config and parameters
cp /home/xidaren2/xshortfin/export/model.vmfb /home/xidaren2/xshortfin/goldens/iree_run/
cp /home/xidaren2/xshortfin/export/config.json /home/xidaren2/xshortfin/goldens/iree_run/
cp /home/xidaren2/xshortfin/export/open-llama-3b-v2-f16.gguf /home/xidaren2/xshortfin/goldens/iree_run/

echo "=== Running Prefill ==="
# Copy prefill inputs
cp /home/xidaren2/xshortfin/goldens/sharktank/prefill_inputs/tokens.npy /home/xidaren2/xshortfin/goldens/iree_run/prefill_inputs/
cp /home/xidaren2/xshortfin/goldens/sharktank/prefill_inputs/seq_lens.npy /home/xidaren2/xshortfin/goldens/iree_run/prefill_inputs/
cp /home/xidaren2/xshortfin/goldens/sharktank/prefill_inputs/seq_block_ids.npy /home/xidaren2/xshortfin/goldens/iree_run/prefill_inputs/

# Convert and copy cache state for prefill
convert_and_copy_cache \
    "/home/xidaren2/xshortfin/goldens/sharktank/prefill_inputs/cache_state_0.npy" \
    "/home/xidaren2/xshortfin/goldens/iree_run/prefill_inputs/cache_state_0.npy"

# Run prefill
iree-run-module \
    --module=/home/xidaren2/xshortfin/goldens/iree_run/model.vmfb \
    --parameters=model=/home/xidaren2/xshortfin/goldens/iree_run/open-llama-3b-v2-f16.gguf \
    --function=prefill_bs1 \
    --device=hip \
    --input=@/home/xidaren2/xshortfin/goldens/iree_run/prefill_inputs/tokens.npy \
    --input=@/home/xidaren2/xshortfin/goldens/iree_run/prefill_inputs/seq_lens.npy \
    --input=@/home/xidaren2/xshortfin/goldens/iree_run/prefill_inputs/seq_block_ids.npy \
    --input=@/home/xidaren2/xshortfin/goldens/iree_run/prefill_inputs/cache_state_0.npy \
    --output=@/home/xidaren2/xshortfin/goldens/iree_run/prefill_outputs/logits.npy

# Compare prefill outputs
echo "Comparing prefill outputs..."
python3 -c "
import numpy as np
from scipy.special import softmax, log_softmax

def compute_metrics(logits1, logits2):
    # Original top-k comparison
    top_k1 = np.argsort(logits1)[-10:][::-1]
    top_k2 = np.argsort(logits2)[-10:][::-1]
    top_1_different = top_k1[0] != top_k2[0]
    different_tokens = len(set(top_k1) ^ set(top_k2))
    
    # Original relative differences
    abs_diff = np.abs(logits1 - logits2)
    relative_diff = abs_diff / (np.abs(logits2) + 1e-6) * 100
    mean_diff = np.mean(relative_diff)
    max_diff = np.max(relative_diff)
    
    # Add cross entropy in both directions
    probs1 = softmax(logits1)
    probs2 = softmax(logits2)
    log_probs1 = log_softmax(logits1)
    log_probs2 = log_softmax(logits2)
    
    ce_1_to_2 = -np.sum(probs2 * log_probs1)  # Using 1's logprobs with 2's probs
    ce_2_to_1 = -np.sum(probs1 * log_probs2)  # Using 2's logprobs with 1's probs
    symmetric_ce = (ce_1_to_2 + ce_2_to_1) / 2
    
    return (top_1_different, different_tokens, top_k1[0], top_k2[0], 
            mean_diff, max_diff, ce_1_to_2, ce_2_to_1, symmetric_ce)

iree_logits = np.load('/home/xidaren2/xshortfin/goldens/iree_run/prefill_outputs/logits.npy')
shark_logits = np.load('/home/xidaren2/xshortfin/goldens/sharktank/prefill_outputs/logits.npy')
seq_lens = np.load('/home/xidaren2/xshortfin/goldens/iree_run/prefill_inputs/seq_lens.npy')
seq_len = seq_lens[0]

# Compare at sequence length position
(top_1_diff, num_diff_top_k, iree_top, shark_top, 
 mean_diff, max_diff, ce_iree_to_shark, ce_shark_to_iree, symmetric_ce) = compute_metrics(
    iree_logits[0, seq_len - 1, :], shark_logits[0, seq_len - 1, :]
)

print(f'Prefill comparison:')
print(f'Top-1 token different: {top_1_diff} (IREE: {iree_top}, Shark: {shark_top})')
print(f'Number of different tokens in top-10: {num_diff_top_k}')
print(f'Logits mean relative difference: {mean_diff:.2f}%')
print(f'Logits max relative difference: {max_diff:.2f}%')
print(f'Cross entropy (IREE→Shark): {ce_iree_to_shark:.6f}')
print(f'Cross entropy (Shark→IREE): {ce_shark_to_iree:.6f}')
print(f'Symmetric cross entropy: {symmetric_ce:.6f}')
"

# Find total number of decode steps
NUM_DECODE_STEPS=$(ls -d /home/xidaren2/xshortfin/goldens/sharktank/decode_invocation*_inputs | wc -l)
echo "Found $NUM_DECODE_STEPS decode steps to process"

# Process each decode step
for ((step=0; step<NUM_DECODE_STEPS; step++)); do
    echo "=== Running Decode Step $step ==="
    
    # Create directories for this step
    mkdir -p /home/xidaren2/xshortfin/goldens/iree_run/decode_invocation${step}_inputs
    mkdir -p /home/xidaren2/xshortfin/goldens/iree_run/decode_invocation${step}_outputs
    
    # Copy inputs from sharktank
    cp /home/xidaren2/xshortfin/goldens/sharktank/decode_invocation${step}_inputs/tokens.npy /home/xidaren2/xshortfin/goldens/iree_run/decode_invocation${step}_inputs/
    cp /home/xidaren2/xshortfin/goldens/sharktank/decode_invocation${step}_inputs/seq_lens.npy /home/xidaren2/xshortfin/goldens/iree_run/decode_invocation${step}_inputs/
    cp /home/xidaren2/xshortfin/goldens/sharktank/decode_invocation${step}_inputs/start_positions.npy /home/xidaren2/xshortfin/goldens/iree_run/decode_invocation${step}_inputs/
    cp /home/xidaren2/xshortfin/goldens/sharktank/decode_invocation${step}_inputs/seq_block_ids.npy /home/xidaren2/xshortfin/goldens/iree_run/decode_invocation${step}_inputs/

    # Convert and copy cache state for decode
    convert_and_copy_cache \
        "/home/xidaren2/xshortfin/goldens/sharktank/decode_invocation${step}_inputs/cache_state_0.npy" \
        "/home/xidaren2/xshortfin/goldens/iree_run/decode_invocation${step}_inputs/cache_state_0.npy"

    # Run decode step
    iree-run-module \
        --module=/home/xidaren2/xshortfin/goldens/iree_run/model.vmfb \
        --parameters=model=/home/xidaren2/xshortfin/goldens/iree_run/open-llama-3b-v2-f16.gguf \
        --function=decode_bs1 \
        --device=hip \
        --input=@/home/xidaren2/xshortfin/goldens/iree_run/decode_invocation${step}_inputs/tokens.npy \
        --input=@/home/xidaren2/xshortfin/goldens/iree_run/decode_invocation${step}_inputs/seq_lens.npy \
        --input=@/home/xidaren2/xshortfin/goldens/iree_run/decode_invocation${step}_inputs/start_positions.npy \
        --input=@/home/xidaren2/xshortfin/goldens/iree_run/decode_invocation${step}_inputs/seq_block_ids.npy \
        --input=@/home/xidaren2/xshortfin/goldens/iree_run/decode_invocation${step}_inputs/cache_state_0.npy \
        --output=@/home/xidaren2/xshortfin/goldens/iree_run/decode_invocation${step}_outputs/logits.npy

    # Compare outputs
    echo "Comparing outputs for step $step..."
    python3 -c "
import numpy as np
from scipy.special import softmax, log_softmax

def compute_metrics(logits1, logits2):
    # Original top-k comparison
    top_k1 = np.argsort(logits1)[-10:][::-1]
    top_k2 = np.argsort(logits2)[-10:][::-1]
    top_1_different = top_k1[0] != top_k2[0]
    different_tokens = len(set(top_k1) ^ set(top_k2))
    
    # Original relative differences
    abs_diff = np.abs(logits1 - logits2)
    relative_diff = abs_diff / (np.abs(logits2) + 1e-6) * 100
    mean_diff = np.mean(relative_diff)
    max_diff = np.max(relative_diff)
    
    # Add cross entropy in both directions
    probs1 = softmax(logits1)
    probs2 = softmax(logits2)
    log_probs1 = log_softmax(logits1)
    log_probs2 = log_softmax(logits2)
    
    ce_1_to_2 = -np.sum(probs2 * log_probs1)  # Using 1's logprobs with 2's probs
    ce_2_to_1 = -np.sum(probs1 * log_probs2)  # Using 2's logprobs with 1's probs
    symmetric_ce = (ce_1_to_2 + ce_2_to_1) / 2
    
    return (top_1_different, different_tokens, top_k1[0], top_k2[0], 
            mean_diff, max_diff, ce_1_to_2, ce_2_to_1, symmetric_ce)

iree_logits = np.load('/home/xidaren2/xshortfin/goldens/iree_run/decode_invocation${step}_outputs/logits.npy')
shark_logits = np.load('/home/xidaren2/xshortfin/goldens/sharktank/decode_invocation${step}_outputs/logits.npy')

(top_1_diff, num_diff_top_k, iree_top, shark_top, 
 mean_diff, max_diff, ce_iree_to_shark, ce_shark_to_iree, symmetric_ce) = compute_metrics(
    iree_logits[0, 0, :], shark_logits[0, 0, :]
)

print(f'Decode step ${step} comparison:')
print(f'Top-1 token different: {top_1_diff} (IREE: {iree_top}, Shark: {shark_top})')
print(f'Number of different tokens in top-10: {num_diff_top_k}')
print(f'Logits mean relative difference: {mean_diff:.2f}%')
print(f'Logits max relative difference: {max_diff:.2f}%')
print(f'Cross entropy (IREE→Shark): {ce_iree_to_shark:.6f}')
print(f'Cross entropy (Shark→IREE): {ce_shark_to_iree:.6f}')
print(f'Symmetric cross entropy: {symmetric_ce:.6f}')
"
done

echo "=== Execution Complete ==="
# Print directory structure at the end
# tree /home/xidaren2/xshortfin/goldens/iree_run