#!/usr/bin/env python3
import asyncio
import shutil
import numpy as np
from pathlib import Path
import shortfin.array as sfnp
from scipy.special import softmax, log_softmax
from shortfin_apps.llm.components.messages import InferenceExecRequest, InferencePhase
from shortfin_apps.llm.components.service import GenerateService
from shortfin_apps.llm.components.manager import SystemManager
from shortfin_apps.llm.components.tokenizer import Tokenizer
from shortfin_apps.llm.components.config_struct import ModelParams

# Path configurations
SCRIPT_DIR = Path.home() / "xshortfin/gscripts"
GOLDEN_DIR = Path.home() / "xshortfin/goldens/sharktank"
SHORTFIN_DIR = Path.home() / "xshortfin/goldens/shortfin_run"
EXPORT_DIR = Path.home() / "xshortfin/goldens/exported_llama_model"

def compute_metrics(logits1, logits2):
    """Compute comparison metrics between two sets of logits."""
    # Top-k comparison
    top_k1 = np.argsort(logits1)[-10:][::-1]
    top_k2 = np.argsort(logits2)[-10:][::-1]
    top_1_different = top_k1[0] != top_k2[0]
    different_tokens = len(set(top_k1) ^ set(top_k2))
    
    # Relative differences
    abs_diff = np.abs(logits1 - logits2)
    relative_diff = abs_diff / (np.abs(logits2) + 1e-6) * 100
    mean_diff = np.mean(relative_diff)
    max_diff = np.max(relative_diff)
    
    # Cross entropy in both directions
    probs1 = softmax(logits1)
    probs2 = softmax(logits2)
    log_probs1 = log_softmax(logits1)
    log_probs2 = log_softmax(logits2)
    
    ce_1_to_2 = -np.sum(probs2 * log_probs1)
    ce_2_to_1 = -np.sum(probs1 * log_probs2)
    symmetric_ce = (ce_1_to_2 + ce_2_to_1) / 2
    
    return (top_1_different, different_tokens, top_k1[0], top_k2[0],
            mean_diff, max_diff, ce_1_to_2, ce_2_to_1, symmetric_ce)

async def to_np(tensor: sfnp.device_array):
    """Convert a device array to numpy array."""
    tensor_shape = tensor.shape
    host_tensor = tensor.for_transfer()
    host_tensor.copy_from(tensor)
    await tensor.device
    tensor = host_tensor
    is_float16 = tensor.dtype == sfnp.float16
    tensor = tensor.items
    dtype = np.float16 if is_float16 else tensor.typecode
    np_array = np.frombuffer(tensor, dtype=dtype)
    return np_array.reshape(*tensor_shape)

def setup_directories():
    """Create and prepare output directories."""
    # Clean existing directory
    if SHORTFIN_DIR.exists():
        shutil.rmtree(SHORTFIN_DIR)
    
    # Create directories
    for dir_name in ["prefill_inputs", "prefill_outputs", 
                    "decode_invocation0_inputs", "decode_invocation0_outputs"]:
        (SHORTFIN_DIR / dir_name).mkdir(parents=True, exist_ok=True)
    
    # Copy input files from golden directory
    for dir_name in ["prefill_inputs", "decode_invocation0_inputs"]:
        src_dir = GOLDEN_DIR / dir_name
        dst_dir = SHORTFIN_DIR / dir_name
        if dst_dir.exists():
            shutil.rmtree(dst_dir)
        shutil.copytree(src_dir, dst_dir)

async def initialize_service():
    """Initialize the Shortfin service."""
    sysman = SystemManager(device="hip")
    tokenizer = Tokenizer.from_tokenizer_json_file(EXPORT_DIR / "tokenizer.json")
    model_params = ModelParams.load_json(EXPORT_DIR / "edited_config.json")

    service = GenerateService(
        name="default",
        sysman=sysman,
        tokenizer=tokenizer,
        model_params=model_params,
    )

    service.load_inference_module(EXPORT_DIR / "model.vmfb")
    service.load_inference_parameters(
        EXPORT_DIR / "open-llama-3b-v2-f16.irpa",
        parameter_scope="model"
    )
    service.start()
    return service

def compare_outputs(shortfin_outputs, golden_outputs, phase, seq_len=None):
    """Compare outputs and print metrics."""
    print(f"\n=== {phase.upper()} COMPARISON ===")
    print(f"Shapes - Shortfin: {shortfin_outputs.shape}, Golden: {golden_outputs.shape}")
    
    if phase == "prefill":
        # Shortfin only outputs the last token's logits [batch_size, 1, vocab]
        # while golden outputs all tokens [batch_size, seq_len, vocab]
        shortfin_logits = shortfin_outputs[0, 0, :]  # Just take the only position we have
        golden_logits = golden_outputs[0, seq_len - 1, :]  # Take the last position from golden
    else:
        # For decode, both should be [batch_size, 1, vocab]
        shortfin_logits = shortfin_outputs[0, 0, :]
        golden_logits = golden_outputs[0, 0, :]
    
    print(f"Comparing logits - Shortfin slice shape: {shortfin_logits.shape}, "
          f"Golden slice shape: {golden_logits.shape}")
    
    (top_1_diff, num_diff_top_k, shortfin_top, golden_top,
     mean_diff, max_diff, ce_shortfin_to_golden, ce_golden_to_shortfin, 
     symmetric_ce) = compute_metrics(shortfin_logits, golden_logits)
    
    print(f'{phase.capitalize()} comparison:')
    print(f'Top-1 token different: {top_1_diff} (Shortfin: {shortfin_top}, Golden: {golden_top})')
    print(f'Number of different tokens in top-10: {num_diff_top_k}')
    print(f'Logits mean relative difference: {mean_diff:.2f}%')
    print(f'Logits max relative difference: {max_diff:.2f}%')
    print(f'Cross entropy (Shortfin→Golden): {ce_shortfin_to_golden:.6f}')
    print(f'Cross entropy (Golden→Shortfin): {ce_golden_to_shortfin:.6f}')
    print(f'Symmetric cross entropy: {symmetric_ce:.6f}')

async def run_inference():
    """Run inference and compare with golden outputs."""
    print("=== Running Prefill ===")
    
    # Load prefill inputs
    prefill_tokens = np.load(SHORTFIN_DIR / "prefill_inputs/tokens.npy")
    prefill_cache = np.load(SHORTFIN_DIR / "prefill_inputs/cache_state_0.npy").astype(np.float16)
    prefill_block_ids = np.load(SHORTFIN_DIR / "prefill_inputs/seq_block_ids.npy")
    prefill_seq_lens = np.load(SHORTFIN_DIR / "prefill_inputs/seq_lens.npy")
    
    # Initialize service
    service = await initialize_service()
    
    # Run prefill
    exec_request = InferenceExecRequest(
        InferencePhase.PREFILL,
        prefill_tokens.tolist()[0]
    )
    exec_request.cache_state = prefill_cache
    exec_request.seq_block_ids = prefill_block_ids
    exec_request.seq_lens = prefill_seq_lens
    
    service.batcher.submit(exec_request)
    await exec_request.done
    
    # Save and compare prefill outputs
    prefill_logits = await to_np(exec_request.result_logits)
    np.save(SHORTFIN_DIR / "prefill_outputs/logits.npy", prefill_logits)
    
    golden_prefill = np.load(GOLDEN_DIR / "prefill_outputs/logits.npy")
    compare_outputs(prefill_logits, golden_prefill, "prefill", prefill_seq_lens[0])
    
    print("\n=== Running Decode ===")
    
    # Load decode inputs
    decode_tokens = np.load(SHORTFIN_DIR / "decode_invocation0_inputs/tokens.npy")
    decode_start_pos = np.load(SHORTFIN_DIR / "decode_invocation0_inputs/start_positions.npy")
    decode_block_ids = np.load(SHORTFIN_DIR / "decode_invocation0_inputs/seq_block_ids.npy")
    decode_seq_lens = np.load(SHORTFIN_DIR / "decode_invocation0_inputs/seq_lens.npy")
    
    # Run decode
    exec_request.reset(InferencePhase.DECODE)
    exec_request.input_token_ids = decode_tokens.tolist()[0]
    exec_request.start_position = decode_start_pos[0]
    exec_request.seq_block_ids = decode_block_ids
    exec_request.seq_lens = decode_seq_lens
    
    service.batcher.submit(exec_request)
    await exec_request.done
    
    # Save and compare decode outputs
    decode_logits = await to_np(exec_request.result_logits)
    np.save(SHORTFIN_DIR / "decode_invocation0_outputs/logits.npy", decode_logits)
    
    golden_decode = np.load(GOLDEN_DIR / "decode_invocation0_outputs/logits.npy")
    compare_outputs(decode_logits, golden_decode, "decode")

async def main():
    try:
        print("=== Starting Shortfin Run ===")
        setup_directories()
        await run_inference()
        print("\n=== Execution Complete ===")
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    import shortfin
    ls = shortfin.amdgpu.SystemBuilder().create_system()
    ls.run(main())