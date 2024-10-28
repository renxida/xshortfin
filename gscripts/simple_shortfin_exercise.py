#!/usr/bin/env python3

import asyncio
import numpy as np
from pathlib import Path
import shortfin.array as sfnp
import safetensors.numpy as sfnp_safe
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


export_dir = Path("/tmp/export_and_serve/")


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


async def initialize_service(export_dir: Path) -> GenerateService:
    """Initialize and start the generation service with the given exported model directory."""
    try:
        sysman = SystemManager(device="hip")
        tokenizer = Tokenizer.from_tokenizer_json_file(export_dir / "tokenizer.json")
        model_params = ModelParams.load_json(export_dir / "edited_config.json")

        service = GenerateService(
            name="default",
            sysman=sysman,
            tokenizer=tokenizer,
            model_params=model_params,
        )

        # Load model artifacts
        service.load_inference_module(export_dir / "model.vmfb")
        service.load_inference_parameters(
            export_dir / "open-llama-3b-v2-f16.gguf", parameter_scope="model"
        )
        service.start()

        return service
    except Exception as e:
        raise RuntimeError(f"Failed to initialize service: {str(e)}") from e


from transformers import AutoTokenizer

tokenizer_path = export_dir / "tokenizer.json"
try:
    tokenizer = AutoTokenizer.from_pretrained(str(export_dir))
except Exception as e:
    print(f"Warning: Failed to load tokenizer using AutoTokenizer: {e}")
    # Fallback: Try loading directly from json
    try:
        with open(tokenizer_path, "r") as f:
            tokenizer_json = json.load(f)
        from transformers import PreTrainedTokenizerFast

        tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_path))
    except Exception as e:
        raise RuntimeError(f"Failed to load tokenizer from {tokenizer_path}: {e}")


def tok(text: str):
    """Tokenize a text."""
    return tokenizer.encode(text)


def detok(token_ids: List[int]):
    """Decode a single token ID."""
    return tokenizer.decode(token_ids)


async def analyze_logits(logits: np.ndarray):
    """Analyze logits and print top 10 tokens with their probabilities."""
    tokenizer_path = export_dir / "tokenizer.json"

    try:
        # Load the tokenizer

        # Basic diagnostics
        print("\nLogits Statistics:")
        print(f"Shape: {logits.shape}")
        print(f"dtype: {logits.dtype}")
        print(f"min: {np.min(logits):.4f}")
        print(f"max: {np.max(logits):.4f}")
        print(f"mean: {np.mean(logits):.4f}")
        print(f"std: {np.std(logits):.4f}")

        # Get top 10 tokens
        top_k = 10
        top_indices = np.argsort(logits)[-top_k:][::-1]

        # Compute softmax for top tokens
        top_logits = logits[top_indices]
        top_logits_shifted = top_logits - np.max(top_logits)  # For numerical stability
        top_probs = np.exp(top_logits_shifted) / np.sum(np.exp(top_logits_shifted))

        print("\nTop 10 tokens:")
        print("Token ID | Token | Probability")
        print("-" * 40)
        for idx, prob in zip(top_indices, top_probs):
            try:
                # Decode single token
                (token_text,) = detok([int(idx)])
                # Replace newlines and tabs for display
                token_text = token_text.replace("\n", "\\n").replace("\t", "\\t")
                print(f"{idx:7d} | {token_text:20s} | {prob:.4f}")
            except Exception as e:
                print(f"{idx:7d} | <decode error: {str(e)}> | {prob:.4f}")

    except Exception as e:
        raise RuntimeError(f"Analysis failed: {str(e)}") from e


async def main():
    # Setup export directory

    if not export_dir.exists():
        raise FileNotFoundError(f"Export directory not found: {export_dir}")

    try:
        # Initialize service
        service = await initialize_service(export_dir)

        # Example input tokens
        input_tokens = [
            1,
            29532,
            29500,
            29536,
            29500,
            29556,
            29500,
            29562,
            29500,
            29561,
            29500,
        ]
        print(f"Input tokens: {input_tokens}")
        print(f"Input text: {detok(input_tokens)}")

        # prefill
        # output should be of shape [batch_size, seq_len, vocab_size]
        try:
            exec_request = InferenceExecRequest(InferencePhase.PREFILL, input_tokens)

            service.batcher.submit(exec_request)
            await exec_request.done

            result_logits = exec_request.result_logits
        except Exception as e:
            raise RuntimeError(f"Inference failed: {str(e)}") from e
        print(result_logits.shape)
        assert tuple(result_logits.shape) == (1, 1, 32000)

        token = sfnp.argmax(result_logits)
        token_int = token.items[0]

        # post prefil analysis
        print(f"First token: {token_int}")
        print(f"First token text: {detok([token_int])}")

        result_logits = await to_np(result_logits)
        last_token = result_logits[0, -1, :]

        await analyze_logits(last_token)

        tokens_history = [] + [token_int]

        # decode
        to_decode = 3
        exec_request.start_position = len(input_tokens) - 1
        for i in range(to_decode):
            exec_request.reset(InferencePhase.DECODE)
            exec_request.input_token_ids = [token_int]
            exec_request.start_position += 1
            service.batcher.submit(exec_request)
            await exec_request.done

            result_logits = exec_request.result_logits
            token = sfnp.argmax(exec_request.result_logits)
            token_int = token.items[0]
            tokens_history.append(token_int)

            print(f"Token {i+1}: {token_int}")
            result_logits = await to_np(result_logits)
            print(f"Printing decoded token # {i+1} logit stats:")
            await analyze_logits(result_logits[0, 0, :])

        print(f"Generated tokens: {tokens_history}")
        print(f"Generated text: {detok(tokens_history)}")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise
    finally:
        # Cleanup
        pass


if __name__ == "__main__":
    ls = shortfin.amdgpu.SystemBuilder().create_system()
    ls.run(main())
"""
# i need the correct kvcache to test pz/'oaxxxxxiiiirefill
python -m sharktank.examples.paged_llm_v1 --gguf-file=/tmp/sharktank/llama/open-llama-3b-v2-f16.gguf "1 2 3 4 5" --tokenizer-config-json /tmp/sharktank/llama/tokenizer_config.json
# run this, dump all inputs to decode before 1st decode, incl. kvcache

If doesn't work investigate.

If it works, try the inputs with iree-run-module. Cache outputs

If that works, try using my bare bones shortfin code to run with the same inputs. Cache outputs.

If that works, try shortfin serve with overrided input. Cache outputs.

Make sure the numbers going in & the numbers coming out are the EXACT same thing.
iree-run-module and shortfin should be bit-identical.

Avoid: messing with shortfin code trying to fix things when iree module is screwed.

Generate goldens to work with & validate that we're doing the EXACT same thing.

Always do it based on the INPUT to the function.
"""

