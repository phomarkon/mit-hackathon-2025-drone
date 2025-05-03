#!/bin/bash
# Convenience script for running the thermal drone anomaly detection tests

# Set default values
MODE=""
INPUT_DIR=""
OUTPUT_DIR=""
SAVE_IMAGES=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dashboard)
      MODE="dashboard"
      shift
      ;;
    --process)
      MODE="process"
      shift
      ;;
    --input)
      INPUT_DIR="$2"
      shift 2
      ;;
    --output)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --save)
      SAVE_IMAGES=true
      shift
      ;;
    --help)
      echo "Usage: $0 [--dashboard | --process] [--input dir] [--output dir] [--save]"
      echo ""
      echo "Options:"
      echo "  --dashboard         Launch the Streamlit dashboard interface"
      echo "  --process           Process a directory of images with both models"
      echo "  --input dir         Specify input directory containing thermal images"
      echo "  --output dir        Specify output directory for results"
      echo "  --save              Save processed images with annotations"
      echo "  --help              Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Check that a mode was specified
if [[ -z "$MODE" ]]; then
  echo "Error: You must specify either --dashboard or --process mode"
  echo "Use --help for usage information"
  exit 1
fi

# Build the command
CMD="python run_test.py --mode $MODE"

# Add additional arguments if needed
if [[ "$MODE" == "process" ]]; then
  if [[ -z "$INPUT_DIR" ]]; then
    echo "Error: You must specify an input directory with --input when using --process mode"
    exit 1
  fi
  
  CMD="$CMD --input \"$INPUT_DIR\""
  
  if [[ -n "$OUTPUT_DIR" ]]; then
    CMD="$CMD --output \"$OUTPUT_DIR\""
  fi
  
  if [[ "$SAVE_IMAGES" == true ]]; then
    CMD="$CMD --save"
  fi
fi

# Execute the command
echo "Running: $CMD"
eval $CMD 