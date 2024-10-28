import numpy as np

def analyze_npy(path: str) -> None:
    print(f"\n=== {path} ===")
    arr = np.load(path)
    print(f"Shape: {arr.shape}")
    print(f"Dtype: {arr.dtype}")

base = "/home/xidaren2/xshortfin/goldens/sharktank"

# Prefill inputs
print("\n=== Prefill Inputs ===")
for file in ['tokens.npy', 'cache_state_0.npy', 'seq_block_ids.npy', 'seq_lens.npy']:
    path = f"{base}/prefill_inputs/{file}"
    analyze_npy(path)

# Prefill outputs
print("\n=== Prefill Outputs ===")
path = f"{base}/prefill_outputs/logits.npy"
analyze_npy(path)

# Decode 0 inputs
print("\n=== Decode 0 Inputs ===")
for file in ['cache_state_0.npy', 'seq_lens.npy', 'seq_block_ids.npy', 'start_positions.npy', 'tokens.npy']:
    path = f"{base}/decode_invocation0_inputs/{file}"
    analyze_npy(path)

# Decode 0 outputs
print("\n=== Decode 0 Outputs ===")
path = f"{base}/decode_invocation0_outputs/logits.npy"
analyze_npy(path)