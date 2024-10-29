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

async def main():
    try:
        print("=== Starting Shortfin Run ===")
        
        # Setup directories
        if SHORTFIN_DIR.exists():
            shutil.rmtree(SHORTFIN_DIR)
        
        for dir_name in ["prefill_inputs", "prefill_outputs", 
                        "decode_invocation0_inputs", "decode_invocation0_outputs"]:
            (SHORTFIN_DIR / dir_name).mkdir(parents=True, exist_ok=True)
        
        for dir_name in ["prefill_inputs", "decode_invocation0_inputs"]:
            src_dir = GOLDEN_DIR / dir_name
            dst_dir = SHORTFIN_DIR / dir_name
            if dst_dir.exists():
                shutil.rmtree(dst_dir)
            shutil.copytree(src_dir, dst_dir)

        # Initialize service
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

        # Run prefill
        print("=== Running Prefill ===")
        prefill_tokens = np.load(SHORTFIN_DIR / "prefill_inputs/tokens.npy")
        prefill_cache = np.load(SHORTFIN_DIR / "prefill_inputs/cache_state_0.npy").astype(np.float16)
        prefill_block_ids = np.load(SHORTFIN_DIR / "prefill_inputs/seq_block_ids.npy")
        prefill_seq_lens = np.load(SHORTFIN_DIR / "prefill_inputs/seq_lens.npy")

        # print the golden inputs
        print("=== Golden Inputs ===")
        print(f"Tokens: {prefill_tokens}")
        print(f"Block IDs: {prefill_block_ids}")
        print(f"Sequence Lengths: {prefill_seq_lens}")
        # only print summary of cache state
        print(f"Cache State shape: {prefill_cache.shape}")
        # min
        print(f"Cache State min: {np.min(prefill_cache)}")
        # max
        print(f"Cache State max: {np.max(prefill_cache)}")
        # mean
        print(f"Cache State mean: {np.mean(prefill_cache)}")

        # remove 0 padding behind prefill_tokens
        prefill_tokens = prefill_tokens[:, :prefill_seq_lens[0]]
        



        exec_request = InferenceExecRequest(
            InferencePhase.PREFILL,
            prefill_tokens.tolist()[0]
        )
        exec_request.seq_block_ids = prefill_block_ids
        exec_request.seq_lens = prefill_seq_lens
        print(exec_request)

        service.batcher.submit(exec_request)
        await exec_request.done

        prefill_logits = await to_np(exec_request.result_logits)
        np.save(SHORTFIN_DIR / "prefill_outputs/logits.npy", prefill_logits)

        # Compare prefill outputs
        golden_prefill = np.load(GOLDEN_DIR / "prefill_outputs/logits.npy")
        print("\n=== PREFILL COMPARISON ===")
        print(f"Shapes - Shortfin: {prefill_logits.shape}, Golden: {golden_prefill.shape}")

        shortfin_logits = prefill_logits[0, 0, :]
        golden_logits = golden_prefill[0, prefill_seq_lens[0] - 1, :]

        # Compute prefill metrics
        top_k1 = np.argsort(shortfin_logits)[-10:][::-1]
        top_k2 = np.argsort(golden_logits)[-10:][::-1]
        top_1_different = top_k1[0] != top_k2[0]
        different_tokens = len(set(top_k1) ^ set(top_k2))

        abs_diff = np.abs(shortfin_logits - golden_logits)
        relative_diff = abs_diff / (np.abs(golden_logits) + 1e-6) * 100
        mean_diff = np.mean(relative_diff)
        max_diff = np.max(relative_diff)

        probs1 = softmax(shortfin_logits)
        probs2 = softmax(golden_logits)
        log_probs1 = log_softmax(shortfin_logits)
        log_probs2 = log_softmax(golden_logits)

        ce_1_to_2 = -np.sum(probs2 * log_probs1)
        ce_2_to_1 = -np.sum(probs1 * log_probs2)
        symmetric_ce = (ce_1_to_2 + ce_2_to_1) / 2

        print('Prefill comparison:')
        print(f'Top-1 token different: {top_1_different} (Shortfin: {top_k1[0]}, Golden: {top_k2[0]})')
        print(f'Number of different tokens in top-10: {different_tokens}')
        print(f'Logits mean relative difference: {mean_diff:.2f}%')
        print(f'Logits max relative difference: {max_diff:.2f}%')
        print(f'Cross entropy (Shortfin→Golden): {ce_1_to_2:.6f}')
        print(f'Cross entropy (Golden→Shortfin): {ce_2_to_1:.6f}')
        print(f'Symmetric cross entropy: {symmetric_ce:.6f}')

        # return # skip decode

        # Run decode
        print("\n=== Running Decode ===")
        decode_tokens = np.load(SHORTFIN_DIR / "decode_invocation0_inputs/tokens.npy")
        decode_start_pos = np.load(SHORTFIN_DIR / "decode_invocation0_inputs/start_positions.npy")
        decode_block_ids = np.load(SHORTFIN_DIR / "decode_invocation0_inputs/seq_block_ids.npy")
        decode_seq_lens = np.load(SHORTFIN_DIR / "decode_invocation0_inputs/seq_lens.npy")
        cache_state = np.load(SHORTFIN_DIR / "decode_invocation0_inputs/cache_state_0.npy").astype(np.float16)

        # reset kvcache state
        cache_devicearray = service.page_cache.page_tables[0]
        print(f"Cache DeviceArray shape: {cache_devicearray.shape}")
        print(f"Cache State shape: {cache_state.shape}")
        assert tuple(cache_devicearray.shape) == tuple(cache_state.shape)

        cache_host = cache_devicearray.for_transfer()

        with cache_host.map(write=True, discard=True) as m:
            cache_state_as_array = np.asarray(cache_state, dtype='float16')
            m.fill(cache_state_as_array)
        cache_devicearray.copy_from(cache_host)

        

        exec_request.reset(InferencePhase.DECODE)
        exec_request.input_token_ids = decode_tokens.tolist()[0]
        exec_request.start_position = decode_start_pos[0]
        exec_request.seq_block_ids = decode_block_ids
        exec_request.seq_lens = decode_seq_lens


        # print the golden inputs
        print("=== Decode Golden Inputs ===")
        print(f"Tokens: {decode_tokens}")
        print(f"Start Position: {decode_start_pos}")
        print(f"Block IDs: {decode_block_ids}")
        print(f"Sequence Lengths: {decode_seq_lens}")
        # only print summary of cache state
        print(f"Cache State shape: {cache_state.shape}")
        # min
        print(f"Cache State min: {np.min(cache_state)}")
        # max
        print(f"Cache State max: {np.max(cache_state)}")
        # mean
        print(f"Cache State mean: {np.mean(cache_state)}")

        print(exec_request)


        service.batcher.submit(exec_request)
        await exec_request.done

        decode_logits = await to_np(exec_request.result_logits)
        np.save(SHORTFIN_DIR / "decode_invocation0_outputs/logits.npy", decode_logits)

        # Compare decode outputs
        golden_decode = np.load(GOLDEN_DIR / "decode_invocation0_outputs/logits.npy")
        print("\n=== DECODE COMPARISON ===")
        print(f"Shapes - Shortfin: {decode_logits.shape}, Golden: {golden_decode.shape}")

        shortfin_logits = decode_logits[0, 0, :]
        golden_logits = golden_decode[0, 0, :]

        # Compute decode metrics
        top_k1 = np.argsort(shortfin_logits)[-10:][::-1]
        top_k2 = np.argsort(golden_logits)[-10:][::-1]
        top_1_different = top_k1[0] != top_k2[0]
        different_tokens = len(set(top_k1) ^ set(top_k2))

        abs_diff = np.abs(shortfin_logits - golden_logits)
        relative_diff = abs_diff / (np.abs(golden_logits) + 1e-6) * 100
        mean_diff = np.mean(relative_diff)
        max_diff = np.max(relative_diff)

        probs1 = softmax(shortfin_logits)
        probs2 = softmax(golden_logits)
        log_probs1 = log_softmax(shortfin_logits)
        log_probs2 = log_softmax(golden_logits)

        ce_1_to_2 = -np.sum(probs2 * log_probs1)
        ce_2_to_1 = -np.sum(probs1 * log_probs2)
        symmetric_ce = (ce_1_to_2 + ce_2_to_1) / 2

        print('Decode comparison:')
        print(f'Top-1 token different: {top_1_different} (Shortfin: {top_k1[0]}, Golden: {top_k2[0]})')
        print(f'Number of different tokens in top-10: {different_tokens}')
        print(f'Logits mean relative difference: {mean_diff:.2f}%')
        print(f'Logits max relative difference: {max_diff:.2f}%')
        print(f'Cross entropy (Shortfin→Golden): {ce_1_to_2:.6f}')
        print(f'Cross entropy (Golden→Shortfin): {ce_2_to_1:.6f}')
        print(f'Symmetric cross entropy: {symmetric_ce:.6f}')

        print("\n=== Execution Complete ===")
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    import shortfin
    ls = shortfin.amdgpu.SystemBuilder().create_system()
    ls.run(main())