#!/bin/bash

# Default parameters
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
CHECKPOINT_PATH="/path/to/checkpoint"
DATA_FILE="/path/to/video_info.meta.jsonl"
OUTPUT_FILE="motionchat_stage1.jsonl"
RESTORE_NAME="restore"
RESUME=0
NUM_SEGS=16

# Help information
print_help() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -g, --gpus NUM        Number of GPUs to use (default: all available GPUs)"
    echo "  -c, --checkpoint PATH Path to model checkpoint (default: $CHECKPOINT_PATH)"
    echo "  -d, --data PATH       Path to data file (default: $DATA_FILE)"
    echo "  -o, --output FILE     Output filename (default: $OUTPUT_FILE)"
    echo "  -r, --restore NAME    Directory name to save video frames (default: $RESTORE_NAME)"
    echo "  -s, --resume IDX      Index to resume evaluation from (default: 0)"
    echo "  -n, --num-segs NUM    Number of video segments (default: 16)"
    echo "  -h, --help            Display help information"
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -g|--gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        -c|--checkpoint)
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        -d|--data)
            DATA_FILE="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        -r|--restore)
            RESTORE_NAME="$2"
            shift 2
            ;;
        -s|--resume)
            RESUME="$2"
            shift 2
            ;;
        -n|--num-segs)
            NUM_SEGS="$2"
            shift 2
            ;;
        -h|--help)
            print_help
            ;;
        *)
            echo "Unknown option: $1"
            print_help
            ;;
    esac
done

echo "======== Parallel Evaluation Configuration ========"
echo "Number of GPUs: $NUM_GPUS"
echo "Checkpoint Path: $CHECKPOINT_PATH"
echo "Data File: $DATA_FILE"
echo "Output File: $OUTPUT_FILE"
echo "Restore Directory: $RESTORE_NAME"
echo "Resume Index: $RESUME"
echo "Number of Segments: $NUM_SEGS"
echo "==================================================="

# Run evaluation script
python -m eval.motionchat.motionbench.motionchat_motionbench \
    --num_gpus $NUM_GPUS \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --data_file "$DATA_FILE" \
    --output_filename "$OUTPUT_FILE" \
    --restore_name "$RESTORE_NAME" \
    --resume $RESUME \
    --num_segs $NUM_SEGS

echo "Evaluation completed, results saved to $OUTPUT_FILE"