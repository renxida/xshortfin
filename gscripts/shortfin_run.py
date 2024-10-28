#!/usr/bin/env python3
import shutil
import asyncio
import numpy as np
from pathlib import Path
import shortfin.array as sfnp
import shortfin
from shortfin_apps.llm.components.messages import InferenceExecRequest, InferencePhase
from shortfin_apps.llm.components.service import (
    InferenceExecutorProcess,
    GenerateService,
)
from shortfin_apps.llm.components.manager import SystemManager
from shortfin_apps.llm.components.tokenizer import Tokenizer
from shortfin_apps.llm.components.config_struct import ModelParams
from shortfin_apps.llm.components.io_struct import GenerateReqInput
from typing import List

# Set up paths
SCRIPT_DIR = Path.home() / "xshortfin/gscripts"
GOLDEN_DIR = Path.home() / "xshortfin/goldens/sharktank"
SHORTFIN_DIR = Path.home() / "xshortfin/goldens/shortfin_run"
EXPORT_DIR = Path("/tmp/export_and_serve/")

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
    """Create output directories mirroring the input structure."""
    directories = [
        "prefill_outputs",
        "decode_invocation0_outputs"
    ]
    
    for dir_name in directories:
        full_path = SHORTFIN_DIR / dir_name
        full_path.mkdir(parents=True, exist_ok=True)
        
    # Copy input files
    input_dirs = [
        "prefill_inputs",
        "decode_invocation0_inputs"
    ]
    
    for dir_name in input_dirs:
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
        EXPORT_DIR / "open-llama-3b-v2-f16.gguf", 
        parameter_scope="model"
    )
    service.start()
    return service

async def run_inference():
    """Run inference using the same inputs as the golden test cases."""
    # Load input tensors
    prefill_tokens = np.load(SHORTFIN_DIR / "prefill_inputs/tokens.npy")
    prefill_cache = np.load(SHORTFIN_DIR / "prefill_inputs/cache_state_0.npy")
    prefill_block_ids = np.load(SHORTFIN_DIR / "prefill_inputs/seq_block_ids.npy")
    prefill_seq_lens = np.load(SHORTFIN_DIR / "prefill_inputs/seq_lens.npy")

    print("Loaded input shapes:")
    print(f"Tokens: {prefill_tokens.shape}")
    print(f"Cache: {prefill_cache.shape}")
    print(f"Block IDs: {prefill_block_ids.shape}")
    print(f"Seq lens: {prefill_seq_lens.shape}")

    # Initialize service
    service = await initialize_service()

    # Run prefill
    exec_request = InferenceExecRequest(
        InferencePhase.PREFILL, 
        prefill_tokens.tolist()[0]
    )
    
    # Convert cache state to fp16 if needed
    if prefill_cache.dtype != np.float16:
        prefill_cache = prefill_cache.astype(np.float16)
    
    # Set inputs directly as numpy arrays
    exec_request.cache_state = prefill_cache
    exec_request.seq_block_ids = prefill_block_ids
    exec_request.seq_lens = prefill_seq_lens
    
    service.batcher.submit(exec_request)
    await exec_request.done
    
    # Save prefill outputs
    prefill_logits = await to_np(exec_request.result_logits)
    print(f"Prefill output shape: {prefill_logits.shape}")
    np.save(SHORTFIN_DIR / "prefill_outputs/logits.npy", prefill_logits)

    # Load decode inputs
    decode_tokens = np.load(SHORTFIN_DIR / "decode_invocation0_inputs/tokens.npy")
    decode_start_pos = np.load(SHORTFIN_DIR / "decode_invocation0_inputs/start_positions.npy")
    decode_block_ids = np.load(SHORTFIN_DIR / "decode_invocation0_inputs/seq_block_ids.npy")
    decode_seq_lens = np.load(SHORTFIN_DIR / "decode_invocation0_inputs/seq_lens.npy")
    
    print("\nDecode input shapes:")
    print(f"Tokens: {decode_tokens.shape}")
    print(f"Start positions: {decode_start_pos.shape}")
    print(f"Block IDs: {decode_block_ids.shape}")
    print(f"Seq lens: {decode_seq_lens.shape}")
    
    # Run decode
    exec_request.reset(InferencePhase.DECODE)
    exec_request.input_token_ids = decode_tokens.tolist()[0]
    exec_request.start_position = decode_start_pos[0]
    
    # Update inputs for decode phase
    exec_request.seq_block_ids = decode_block_ids
    exec_request.seq_lens = decode_seq_lens
    
    service.batcher.submit(exec_request)
    await exec_request.done
    
    # Save decode outputs
    decode_logits = await to_np(exec_request.result_logits)
    print(f"Decode output shape: {decode_logits.shape}")
    np.save(SHORTFIN_DIR / "decode_invocation0_outputs/logits.npy", decode_logits)

async def main():
    try:
        setup_directories()
        await run_inference()
        print("Successfully ran Shortfin inference and saved outputs")
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    import shortfin
    ls = shortfin.amdgpu.SystemBuilder().create_system()
    ls.run(main())