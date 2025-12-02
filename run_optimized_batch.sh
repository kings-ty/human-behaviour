#!/bin/bash
# Optimized batch processing runner

echo "=================================================================="
echo "OPTIMIZED POSE EXTRACTION - BATCH PROCESSING"
echo "=================================================================="

# Check dependencies
echo "Checking dependencies..."
python3 -c "
import scipy.signal
import cv2
import numpy as np
from ultralytics import YOLO
print('✅ All dependencies available')
" 2>/dev/null || {
    echo "❌ Missing dependencies. Installing..."
    pip install scipy
}

# Check GPU and recommend batch size
echo ""
echo "Checking GPU capabilities..."
GPU_INFO=$(python3 -c "
import torch
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'GPU: {gpu_name}')
    print(f'Memory: {gpu_memory:.1f}GB')
    
    # Recommend batch size
    if 'Xavier' in gpu_name:
        print('Recommended batch size: 30-40')
    elif 'MX450' in gpu_name or 'GTX' in gpu_name:
        print('Recommended batch size: 25-35')
    elif gpu_memory > 8:
        print('Recommended batch size: 50-100')
    else:
        print('Recommended batch size: 20-30')
else:
    print('No CUDA available - using CPU')
    print('Recommended batch size: 10-20')
")

echo "$GPU_INFO"

echo ""
echo "Configuration options:"
echo "1. Small batches (20 videos) - Conservative, less memory"
echo "2. Medium batches (50 videos) - Balanced (recommended)"  
echo "3. Large batches (100 videos) - Fast, needs more memory"
echo "4. Custom batch size"

read -p "Choose batch size option (1/2/3/4): " batch_choice

case $batch_choice in
    1)
        BATCH_SIZE=20
        echo "Using small batches (20 videos)"
        ;;
    2)
        BATCH_SIZE=50
        echo "Using medium batches (50 videos)"
        ;;
    3)
        BATCH_SIZE=100
        echo "Using large batches (100 videos)"
        ;;
    4)
        read -p "Enter custom batch size: " BATCH_SIZE
        echo "Using custom batch size ($BATCH_SIZE videos)"
        ;;
    *)
        BATCH_SIZE=50
        echo "Default: medium batches (50 videos)"
        ;;
esac

echo ""
echo "Processing options:"
echo "1. Fresh start (delete existing batches)"
echo "2. Resume from existing batches"

read -p "Choose processing option (1/2): " process_choice

case $process_choice in
    1)
        echo "Starting fresh processing..."
        rm -rf pose_features_optimized/batches/
        RESUME=""
        ;;
    2)
        echo "Resuming from existing batches..."
        RESUME="--resume"
        ;;
    *)
        echo "Default: fresh start"
        rm -rf pose_features_optimized/batches/
        RESUME=""
        ;;
esac

echo ""
echo "Final configuration:"
echo "  Batch size: $BATCH_SIZE videos"
echo "  Input: train_set/ (2099 videos)"
echo "  Output: pose_features_optimized/"
echo "  Resume: $([ -n "$RESUME" ] && echo "Yes" || echo "No")"

read -p "Proceed with processing? (y/n): " confirm

if [[ $confirm == [yY] || $confirm == [yY][eE][sS] ]]; then
    echo ""
    echo "Starting optimized batch processing..."
    
    # Create log file
    LOG_FILE="optimized_batch_$(date +%Y%m%d_%H%M%S).log"
    
    # Run processing
    python3 preprocessing_optimized_batch.py \
        --data_dir train_set \
        --output_dir pose_features_optimized \
        --batch_size $BATCH_SIZE \
        --device cuda \
        $RESUME \
        2>&1 | tee "$LOG_FILE"
    
    echo ""
    echo "Processing completed!"
    echo "Log saved to: $LOG_FILE"
    
    # Quick validation
    if [ -f "pose_features_optimized/train_sequences_optimized.npy" ]; then
        echo ""
        echo "=================================================================="
        echo "VALIDATION RESULTS"
        echo "=================================================================="
        
        python3 -c "
import numpy as np
import json
import os

try:
    # Load optimized data
    sequences = np.load('pose_features_optimized/train_sequences_optimized.npy')
    labels = np.load('pose_features_optimized/train_labels_optimized.npy')
    with open('pose_features_optimized/train_filenames_optimized.json') as f:
        filenames = json.load(f)
    
    print(f'✅ Successfully processed {len(sequences)} videos')
    print(f'   Sequence shape: {sequences.shape}')
    print(f'   Labels shape: {labels.shape}')
    
    # Check quality
    coords = sequences.reshape(-1, 2)
    zero_pct = np.sum(np.all(coords == 0, axis=1)) / len(coords) * 100
    
    print(f'\\nQuality metrics:')
    print(f'   Coordinate range: [{np.min(coords):.4f}, {np.max(coords):.4f}]')
    print(f'   Normalized properly: {np.max(np.abs(coords)) <= 1.0}')
    print(f'   Zero coordinates: {zero_pct:.1f}%')
    
    # Compare with original if available
    if os.path.exists('pose_features/train_sequences_updated.npy'):
        orig_seq = np.load('pose_features/train_sequences_updated.npy')
        orig_coords = orig_seq.reshape(-1, 2)
        
        print(f'\\nComparison with original:')
        print(f'   Original range: [{np.min(orig_coords):.4f}, {np.max(orig_coords):.4f}]')
        print(f'   Optimized range: [{np.min(coords):.4f}, {np.max(coords):.4f}]')
        print(f'   Improvement: {np.max(orig_coords) > 10 and np.max(np.abs(coords)) <= 1.0}')
        
    print(f'\\n🎯 Ready for LSTM training!')
        
except Exception as e:
    print(f'❌ Validation failed: {e}')
"
    else
        echo "❌ Output file not found - processing may have failed"
    fi
    
else
    echo "Processing cancelled."
fi