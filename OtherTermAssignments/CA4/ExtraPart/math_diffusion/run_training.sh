#!/bin/bash
# run_training.sh - Launch distributed training on 2x T4 GPUs

# Configuration
NUM_GPUS=2
CONFIG="default"
DATASET="gsm8k"
MAX_STEPS=100000
OUTPUT_DIR="./outputs"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --max_steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Print configuration
echo "============================================"
echo "Math Reasoning Diffusion Model - Training"
echo "============================================"
echo "GPUs: $NUM_GPUS"
echo "Config: $CONFIG"
echo "Dataset: $DATASET"
echo "Max Steps: $MAX_STEPS"
echo "Output: $OUTPUT_DIR"
echo "============================================"

# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

# Launch training
if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Launching distributed training..."
    torchrun --nproc_per_node=$NUM_GPUS \
        train.py \
        --config "$CONFIG" \
        --dataset "$DATASET" \
        --max_steps "$MAX_STEPS" \
        --output_dir "$OUTPUT_DIR" \
        2>&1 | tee "$OUTPUT_DIR/training.log"
else
    echo "Launching single GPU training..."
    python train.py \
        --config "$CONFIG" \
        --dataset "$DATASET" \
        --max_steps "$MAX_STEPS" \
        --output_dir "$OUTPUT_DIR" \
        2>&1 | tee "$OUTPUT_DIR/training.log"
fi

echo "Training complete! Checkpoints saved to $OUTPUT_DIR"
