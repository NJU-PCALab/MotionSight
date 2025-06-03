#!/bin/bash

# Default parameters
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
CHECKPOINT_PATH="/path/to/checkpoint"
DATA_FILE="/path/to/video_perspective.json"
VIDEO_DIR="/path/to/FAVOR-Bench"
OUTPUT_FILE="motionchat_favor_dpo.jsonl"
RESTORE_NAME="restore_qwenfavor_parallel"
RESUME=0
NUM_SEGS=16

# Help information
print_help() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -g, --gpus NUM        Number of GPUs to use (default: all available GPUs)"
    echo "  -c, --checkpoint PATH Path to model checkpoint (default: $CHECKPOINT_PATH)"
    echo "  -d, --data PATH       Path to data file (default: $DATA_FILE)"
    echo "  -v, --video-dir DIR   Directory of video files (default: $VIDEO_DIR)"
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
        -v|--video-dir)
            VIDEO_DIR="$2"
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

echo "======== FAVOR Parallel Evaluation Configuration ========"
echo "Number of GPUs: $NUM_GPUS"
echo "Model checkpoint: $CHECKPOINT_PATH"
echo "Data file: $DATA_FILE"
echo "Video directory: $VIDEO_DIR"
echo "Output file: $OUTPUT_FILE"
echo "Restore directory: $RESTORE_NAME"
echo "Resume index: $RESUME"
echo "Number of video segments: $NUM_SEGS"
echo "========================================================"

# Ensure the script is run from the correct directory
CURRENT_DIR=$(pwd)
SCRIPT_DIR=$(dirname "$0")
cd "$SCRIPT_DIR/../.." || { echo "Unable to switch to the correct directory"; exit 1; }

# Run the evaluation script
python -m eval.motionchat.favor.motionchat_favor \
    --num_gpus $NUM_GPUS \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --data_file "$DATA_FILE" \
    --video_dir "$VIDEO_DIR" \
    --output_filename "$OUTPUT_FILE" \
    --restore_name "$RESTORE_NAME" \
    --resume $RESUME \
    --num_segs $NUM_SEGS

# Return to the original directory
cd "$CURRENT_DIR"

echo "Evaluation complete, results saved to $OUTPUT_FILE"