import torch
from typing import Optional, Tuple
from iree.turbine.aot import *

class KVCacheOps(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def read_kvcache(self, kvcache: torch.Tensor, page_index: torch.Tensor, 
                    layer_index: torch.Tensor) -> torch.Tensor:
        """
        Read operation from the KV cache.
        
        Args:
            kvcache: [num_pages, 2662400] cache tensor
            page_index: page to read from
            layer_index: layer to read from
            
        Returns:
            torch.Tensor: [16, 32, 100] data read from cache
        """
        # Get number of pages from first dimension
        num_pages = kvcache.size(0)
        
        # Reshape kvcache from [num_pages, 2662400] to [num_pages, 26, 2, 16, 32, 100]
        shaped_cache = kvcache.view(num_pages, 26, 2, 16, 32, 100)
        
        # Create index tensors with consistent types
        page_idx = page_index.unsqueeze(0).to(torch.int64)
        layer_idx = layer_index.unsqueeze(0).to(torch.int64)
        
        # Create index list for advanced indexing
        indices = [page_idx, layer_idx]
        for _ in range(4):  # Add slice(None) for remaining dimensions
            indices.append(slice(None))
            
        # Read from cache at specified indices and squeeze extra dimensions
        return shaped_cache[tuple(indices)].squeeze(0)

    def write_kvcache(self, kvcache: torch.Tensor, new_data: torch.Tensor,
                     page_index: torch.Tensor, layer_index: torch.Tensor) -> torch.Tensor:
        """
        Write operation to the KV cache.
        
        Args:
            kvcache: [num_pages, 2662400] cache tensor
            new_data: [16, 32, 100] new data tensor to write
            page_index: page to write to
            layer_index: layer to write to
            
        Returns:
            torch.Tensor: Updated cache tensor [num_pages, 2662400]
        """
        # Get number of pages from first dimension
        num_pages = kvcache.size(0)
        
        # Reshape kvcache from [num_pages, 2662400] to [num_pages, 26, 2, 16, 32, 100]
        shaped_cache = kvcache.view(num_pages, 26, 2, 16, 32, 100)
        
        # Create index tensors with consistent types
        page_idx = page_index.unsqueeze(0).to(torch.int64)
        layer_idx = layer_index.unsqueeze(0).to(torch.int64)
        
        # Create index list for advanced indexing
        indices = [page_idx, layer_idx]
        for _ in range(4):  # Add slice(None) for remaining dimensions
            indices.append(slice(None))
            
        # Update the cache at the specified indices
        shaped_cache[tuple(indices)] = new_data
        
        # Reshape back to original flattened form
        return shaped_cache.view(num_pages, -1)

    def forward(self, kvcache: torch.Tensor, new_data: torch.Tensor,
                page_index: torch.Tensor, layer_index: torch.Tensor) -> torch.Tensor:
        """
        Perform read and write operations in sequence.
        
        Args:
            kvcache: [num_pages, 2662400] cache tensor
            new_data: [16, 32, 100] new data tensor to write
            page_index: page to access
            layer_index: layer to access
            
        Returns:
            torch.Tensor: Retrieved data from cache
        """
        # Perform write operation
        updated_cache = self.write_kvcache(kvcache, new_data, page_index, layer_index)
        
        # Perform read operation
        retrieved_data = self.read_kvcache(updated_cache, page_index, layer_index)
        
        return retrieved_data

def export_ops():
    """Export the read and write operations to MLIR"""
    # Create model instance
    model = KVCacheOps()
    
    # Create sample inputs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kvcache = torch.randn(2, 2662400, dtype=torch.float16, device=device)
    new_data = torch.randn(16, 32, 100, dtype=torch.float16, device=device)
    page_index = torch.tensor(0, dtype=torch.int64, device=device)  # Changed to int64
    layer_index = torch.tensor(0, dtype=torch.int64, device=device)  # Changed to int64

    # Create FxProgramsBuilder for the model
    fxb = FxProgramsBuilder(model)

    device_block_count = torch.export.Dim("device_block_count")
    
    # Create dynamic shapes dictionary
    dynamic_shapes = {
        "kvcache": {0: device_block_count},  # First dimension is dynamic
        "new_data": None,     # Static shape
        "page_index": None,   # Static shape
        "layer_index": None   # Static shape
    }

    # Define and export the program
    @fxb.export_program(
        name="kvcache_ops",
        args=(kvcache, new_data, page_index, layer_index),
        dynamic_shapes=dynamic_shapes
    )
    def _(model, kvcache, new_data, page_index, layer_index):
        return model(kvcache, new_data, page_index, layer_index)

    # Export to MLIR
    output = export(fxb)
    
    # Save to file
    output.save_mlir("kvcache.mlir")

if __name__ == "__main__":
    # Create model instance
    model = KVCacheOps()
    
    # Create test inputs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kvcache = torch.randn(2, 2662400, dtype=torch.float16, device=device)
    new_data = torch.randn(16, 32, 100, dtype=torch.float16, device=device)
    page_index = torch.tensor(0, dtype=torch.int64, device=device)  # Changed to int64
    layer_index = torch.tensor(0, dtype=torch.int64, device=device)  # Changed to int64
    
    # Run forward pass
    retrieved_data = model(kvcache, new_data, page_index, layer_index)
    
    # Perform tolerance check
    data_matches = torch.allclose(new_data, retrieved_data, rtol=1e-3)
    
    # Print test results
    print(f"Test Results:")
    print(f"Retrieved data shape: {retrieved_data.shape}")
    print(f"Data matches: {data_matches}")
    
    # Export if tests pass
    if data_matches:
        export_ops()