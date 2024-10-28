set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# Create working directory
WORK_DIR="/tmp/sharktank/replicate_weird_decode"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

# Function to test a specific batch size configuration
test_batch_config() {
    local batch_sizes="$1"
    echo -e "${GREEN}Testing batch sizes: $batch_sizes${NC}"

    # Clean up previous MLIR file
    rm -f model.mlir
    
    # Download model if it doesn't exist
    if [ ! -f "open-llama-3b-v2-f16.gguf" ]; then
        echo "Downloading model..."
        huggingface-cli download --local-dir . SlyEcho/open_llama_3b_v2_gguf open-llama-3b-v2-f16.gguf
    fi
    
    # Export model
    echo "Exporting model with batch sizes: $batch_sizes"
    python -m sharktank.examples.export_paged_llm_v1 \
        --gguf-file="$WORK_DIR/open-llama-3b-v2-f16.gguf" \
        --output-mlir="$WORK_DIR/model.mlir" \
        --output-config="$WORK_DIR/config.json" \
        --bs="$batch_sizes"
    
    # Check MLIR output
    echo -e "${GREEN}Checking MLIR output for batch size $batch_sizes:${NC}"
    echo "Lines containing '_bs':"
    cat model.mlir | grep "_bs" || true
    echo "----------------------------------------"
}

# Test different batch size configurations
echo "Starting batch size tests..."

echo -e "${GREEN}Test 1: Single batch size = 1${NC}"
test_batch_config "1"

echo -e "${GREEN}Test 2: Single batch size = 4${NC}"
test_batch_config "4"

echo -e "${GREEN}Test 3: Multiple batch sizes = 1,4${NC}"
test_batch_config "1,4"

echo -e "${GREEN}Done with all tests${NC}"
