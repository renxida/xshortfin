import subprocess
import json
import numpy as np
from pathlib import Path

def run_iree_module(function, inputs, model_path, vmfb_path, output_file):
    cmd = [
        "iree-run-module",
        f"--module={vmfb_path}",
        f"--function={function}",
        "--device=local-task",
        f"--parameters=model={model_path}",
        f"--output={output_file}"
    ]
    
    for inp in inputs:
        cmd.append(f"--input={inp}")
    
    subprocess.run(cmd, check=True)

def load_bin_file(file_path):
    return np.fromfile(file_path, dtype=np.int64)

def main():
    # Paths
    config_path = "/tmp/sharktank/llama/config.json"
    model_path = "/tmp/sharktank/llama/open-llama-3b-v2-f16.gguf"
    vmfb_path = "/tmp/sharktank/llama/model.vmfb"
    input_dir = Path("/tmp/inputs")
    output_dir = Path("/tmp/outputs")
    output_dir.mkdir(exist_ok=True)

    # Load config
    with open(config_path) as f:
        config = json.load(f)

    # Load generated input data
    arg0 = load_bin_file(input_dir / "arg0.bin")
    arg1 = load_bin_file(input_dir / "arg1.bin")
    
    # Reshape arg0 based on config
    max_seq_len = config["max_seq_len"]
    prefill_batch_size = config["prefill_batch_sizes"][0]
    arg0 = arg0.reshape(prefill_batch_size, max_seq_len)

    config["kv_cache_size"] = 2662400

    # Prefill
    prefill_inputs = [
        f"{prefill_batch_size}x{max_seq_len}xi64=@{input_dir}/arg0.bin",
        f"{prefill_batch_size}xi64=@{input_dir}/arg1.bin",
        f"{prefill_batch_size}x{max_seq_len}xi64={','.join(map(str, range(max_seq_len)))}",
        f"1x{config['kv_cache_size']}xf16"
    ]
    
    prefill_output_file = output_dir / "prefill_output.npy"
    run_iree_module("prefill_bs1", prefill_inputs, model_path, vmfb_path, prefill_output_file)
    
    # Load and inspect prefill output
    prefill_output = np.load(prefill_output_file)
    print("Prefill output shape:", prefill_output.shape)
    print("Prefill output sample:", prefill_output[:5])
    
    # In a real scenario, you would use the prefill output to select the next token.
    # For this example, we'll just choose a random token.
    next_token = np.random.randint(0, 32000)
    
    # Decode
    decode_inputs = [
        f"1x1xi64={next_token}",
        f"1xi64={prefill_batch_size}",
        f"1xi64={arg1[0]}",  # Use the actual sequence length from arg1
        f"1x{max_seq_len}xi64={','.join(map(str, range(max_seq_len)))}",
        f"1x{config['kv_cache_size']}xf16"
    ]
    
    decode_output_file = output_dir / "decode_output.npy"
    run_iree_module("decode_bs1", decode_inputs, model_path, vmfb_path, decode_output_file)
    
    # Load and inspect decode output
    decode_output = np.load(decode_output_file)
    print("Decode output shape:", decode_output.shape)
    print("Decode output sample:", decode_output[:5])
    
    print("Chat completed.")

if __name__ == "__main__":
    
    main()
