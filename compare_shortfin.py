import numpy as np
import os
from json import load
from pathlib import Path

def extract_tokens_from_logits(logits: np.ndarray, seq_lens: list[int]) -> list[int]:
    """Extracts tokens from a batch of logits (B, S, D).
    
    Args:
        logits: NumPy array of shape (batch_size, sequence_length, dimension)
        seq_lens: List of sequence lengths for each batch
        
    Returns:
        List of token indices extracted from the logits
    """
    results = []
    for batch, seq_len in enumerate(seq_lens):
        step_logits = logits[batch, seq_len - 1]
        results.append(np.argmax(step_logits))
    return results

def load_tokenizer(tokenizer_path):
    """Load and invert the tokenizer vocabulary."""
    with open(tokenizer_path) as f:
        tokenizer = load(f)
    vocab = tokenizer['model']['vocab']
    return {v: k for k, v in vocab.items()}

def process_logits_files(shortfin_dir, tokenizer_path):
    """Process all .npy files in the directory and print tokens in a formatted table."""
    # Load tokenizer
    token_dict = load_tokenizer(tokenizer_path)
    
    # Print table header
    header = "| {:<20} | {:<10} | {:<30} |".format("Token", "Token ID", "Filename")
    separator = "|" + "-"*22 + "|" + "-"*12 + "|" + "-"*32 + "|"
    print(separator)
    print(header)
    print(separator)
    
    # Get all .npy files
    npy_files = Path(shortfin_dir).glob('logits*.npy')
    
    for npy_path in npy_files:
        try:
            # Load and reshape logits
            logits = np.load(npy_path)
            orig_shape = logits.shape
            logits = logits.reshape(1, 1, 32000)
            
            # Extract token indices
            token_indices = extract_tokens_from_logits(logits, [1])
            
            # Get token for each index
            token = token_dict[token_indices[0]]
            # Escape special characters and limit length
            token_repr = repr(token)[:18] + ".." if len(repr(token)) > 20 else repr(token)
            
            # Print formatted row
            print("| {:<20} | {:<10} | {:<30} |".format(
                token_repr,
                token_indices[0],
                npy_path.name
            ))
        except Exception as e:
            # Print error row
            print("| {:<20} | {:<10} | {:<30} |".format(
                "FAILED",
                str(logits.shape),
                npy_path.name
            ))
    
    # Print bottom border
    print(separator)

if __name__ == "__main__":
    SHORTFIN_DIR = "/tmp/sharktank/shortfin_llm"
    TOKENIZER_PATH = "/tmp/sharktank/llama/tokenizer.json"
    
    process_logits_files(SHORTFIN_DIR, TOKENIZER_PATH)