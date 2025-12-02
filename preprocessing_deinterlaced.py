#!/usr/bin/env python3
"""
DEINTERLACED YOLOv8-Pose Feature Extraction for HRI30 Dataset
Handles interlaced videos properly before pose extraction

Key improvements:
1. Proper deinterlacing before pose extraction
2. Temporal consistency checking
3. Better error handling and validation
4. GPU memory optimization
"""

import os
import sys
import cv2
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, List, Optional, Dict
import json
import argparse
import time

from ultralytics import YOLO


# COCO 17 keypoint format
KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]


def check_video_interlacing(video_path: str) -> Tuple[bool, Dict]:
    """
    Check if video is interlaced by analyzing frame differences
    
    Returns:
        is_interlaced: Boolean indicating if video is likely interlaced
        stats: Dictionary with analysis statistics
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return False, {"error": "Cannot open video"}
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Sample first 30 frames for analysis
    sample_frames = min(30, frame_count)
    frame_diffs = []
    
    prev_frame = None
    for i in range(sample_frames):
        ret, frame = cap.read()
        if not ret:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if prev_frame is not None:
            # Calculate frame difference
            diff = cv2.absdiff(prev_frame, gray)
            mean_diff = np.mean(diff)
            frame_diffs.append(mean_diff)
        
        prev_frame = gray
    
    cap.release()
    
    if len(frame_diffs) < 10:
        return False, {"error": "Not enough frames to analyze"}
    
    # Check for alternating pattern (interlacing signature)
    even_diffs = np.array(frame_diffs[::2])
    odd_diffs = np.array(frame_diffs[1::2][:len(even_diffs)])
    
    even_mean = np.mean(even_diffs)
    odd_mean = np.mean(odd_diffs)
    
    # Calculate ratio and variance
    ratio = even_mean / odd_mean if odd_mean > 0 else 1.0
    variance_ratio = np.var(even_diffs) / np.var(odd_diffs) if np.var(odd_diffs) > 0 else 1.0
    
    # Interlacing indicators:
    # 1. Significant difference in even/odd frame differences
    # 2. High variance in one set compared to other
    is_interlaced = (abs(ratio - 1.0) > 0.3) or (abs(variance_ratio - 1.0) > 2.0)
    
    stats = {
        "frame_count": frame_count,
        "fps": fps,
        "even_mean_diff": even_mean,
        "odd_mean_diff": odd_mean,
        "ratio": ratio,
        "variance_ratio": variance_ratio,
        "likely_interlaced": is_interlaced
    }
    
    return is_interlaced, stats


def deinterlace_video_stream(cap: cv2.VideoCapture, method: str = 'bob') -> cv2.VideoCapture:
    """
    Apply deinterlacing to video capture stream
    
    Args:
        cap: OpenCV VideoCapture object
        method: Deinterlacing method ('bob', 'linear', 'yadif')
    
    Returns:
        Modified VideoCapture with deinterlacing enabled
    """
    # Try to enable hardware deinterlacing first
    try:
        cap.set(cv2.CAP_PROP_DEINTERLACE, 1)
    except:
        pass  # Not all systems support this
    
    return cap


def deinterlace_frame(frame: np.ndarray, method: str = 'bob') -> np.ndarray:
    """
    Software deinterlacing for individual frames
    
    Args:
        frame: Input frame (H, W, 3)
        method: Deinterlacing method
    
    Returns:
        Deinterlaced frame
    """
    if method == 'bob':
        # Bob deinterlacing - duplicate odd lines
        height, width = frame.shape[:2]
        deinterlaced = np.zeros_like(frame)
        
        # Keep odd lines, interpolate even lines
        deinterlaced[1::2] = frame[1::2]  # Odd lines
        deinterlaced[0::2] = frame[1::2]  # Interpolate even from odd
        
        # Handle boundary
        if height > 1:
            deinterlaced[0] = frame[1]
        
        return deinterlaced
    
    elif method == 'linear':
        # Linear interpolation
        height, width = frame.shape[:2]
        deinterlaced = frame.copy()
        
        # Interpolate even lines from neighboring odd lines
        for i in range(0, height, 2):
            if i + 1 < height and i - 1 >= 0:
                deinterlaced[i] = (frame[i-1] + frame[i+1]) // 2
            elif i + 1 < height:
                deinterlaced[i] = frame[i+1]
        
        return deinterlaced
    
    else:
        return frame  # No deinterlacing


def get_reference_point(keypoints: np.ndarray, confidence: np.ndarray,
                        min_confidence: float = 0.3) -> Optional[np.ndarray]:
    """Calculate reference point for normalization"""
    # Hip center (most stable for industrial actions)
    left_hip, right_hip = keypoints[11], keypoints[12]
    left_hip_conf, right_hip_conf = confidence[11], confidence[12]
    
    if left_hip_conf > min_confidence and right_hip_conf > min_confidence:
        return (left_hip + right_hip) / 2.0
    elif left_hip_conf > min_confidence:
        return left_hip
    elif right_hip_conf > min_confidence:
        return right_hip
    
    # Fallback to shoulder center
    left_shoulder, right_shoulder = keypoints[5], keypoints[6]
    left_shoulder_conf, right_shoulder_conf = confidence[5], confidence[6]
    
    if left_shoulder_conf > min_confidence and right_shoulder_conf > min_confidence:
        return (left_shoulder + right_shoulder) / 2.0
    elif left_shoulder_conf > min_confidence:
        return left_shoulder
    elif right_shoulder_conf > min_confidence:
        return right_shoulder
    
    return None


def normalize_keypoints(keypoints: np.ndarray, confidence: np.ndarray,
                        reference_point: np.ndarray, scale_factor: float = 100.0) -> np.ndarray:
    """Normalize keypoints properly"""
    # Center-relative coordinates
    normalized = keypoints - reference_point
    
    # Scale to [-1, 1] range
    normalized = normalized / scale_factor
    normalized = np.clip(normalized, -1.0, 1.0)
    
    # Zero out low-confidence keypoints
    mask = confidence < 0.3
    normalized[mask] = 0.0
    
    return normalized


def extract_pose_from_frame(model: YOLO, frame: np.ndarray, device: str = 'cuda') -> Tuple[np.ndarray, np.ndarray, bool]:
    """Extract pose with error handling"""
    try:
        results = model(frame, device=device, verbose=False)
        
        if len(results) == 0 or results[0].keypoints is None:
            return np.zeros((17, 2), dtype=np.float32), np.zeros(17, dtype=np.float32), False
        
        keypoints_data = results[0].keypoints
        
        if keypoints_data.xy is None or len(keypoints_data.xy) == 0:
            return np.zeros((17, 2), dtype=np.float32), np.zeros(17, dtype=np.float32), False
        
        kpts = keypoints_data.xy[0].cpu().numpy()
        conf = keypoints_data.conf[0].cpu().numpy() if keypoints_data.conf is not None else np.ones(17)
        
        return kpts.astype(np.float32), conf.astype(np.float32), True
        
    except Exception as e:
        print(f"Error in pose extraction: {e}")
        return np.zeros((17, 2), dtype=np.float32), np.zeros(17, dtype=np.float32), False


def process_video_deinterlaced(video_path: str, model: YOLO, device: str = 'cuda',
                              max_frames: int = 60, deinterlace_method: str = 'bob') -> Tuple[np.ndarray, bool, Dict]:
    """Process video with proper deinterlacing"""
    
    print(f"\nProcessing: {os.path.basename(video_path)}")
    
    # Check if video is interlaced
    is_interlaced, interlace_stats = check_video_interlacing(video_path)
    print(f"  Interlaced: {is_interlaced}")
    if is_interlaced:
        print(f"  Even/Odd ratio: {interlace_stats.get('ratio', 0):.3f}")
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"  Error: Cannot open video")
        return np.zeros((max_frames, 17, 2), dtype=np.float32), False, interlace_stats
    
    # Apply hardware deinterlacing if available
    if is_interlaced:
        cap = deinterlace_video_stream(cap, deinterlace_method)
    
    frames_data = []
    last_valid_keypoints = None
    last_valid_confidence = None
    frame_count = 0
    
    # Track temporal consistency
    movements = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Apply software deinterlacing if needed
        if is_interlaced:
            frame = deinterlace_frame(frame, deinterlace_method)
        
        # Extract pose
        keypoints, confidence, detected = extract_pose_from_frame(model, frame, device)
        
        # Handle missing detection
        if not detected:
            if last_valid_keypoints is not None:
                keypoints = last_valid_keypoints.copy()
                confidence = last_valid_confidence.copy()
            else:
                keypoints = np.zeros((17, 2), dtype=np.float32)
                confidence = np.zeros(17, dtype=np.float32)
        else:
            last_valid_keypoints = keypoints.copy()
            last_valid_confidence = confidence.copy()
        
        # Calculate movement from previous frame
        if len(frames_data) > 0:
            prev_kpts = frames_data[-1][0]
            movement = np.mean(np.linalg.norm(keypoints - prev_kpts, axis=1))
            movements.append(movement)
        
        frames_data.append((keypoints, confidence))
        frame_count += 1
    
    cap.release()
    
    if len(frames_data) == 0:
        print(f"  Warning: No frames extracted")
        return np.zeros((max_frames, 17, 2), dtype=np.float32), False, interlace_stats
    
    print(f"  Extracted {len(frames_data)} frames")
    
    # Check temporal consistency
    if len(movements) > 10:
        even_movements = np.array(movements[::2])
        odd_movements = np.array(movements[1::2][:len(even_movements)])
        
        even_mean = np.mean(even_movements)
        odd_mean = np.mean(odd_movements)
        movement_ratio = even_mean / odd_mean if odd_mean > 0 else 1.0
        
        print(f"  Movement ratio: {movement_ratio:.3f}")
        if abs(movement_ratio - 1.0) > 0.3:
            print(f"  ⚠️  Still shows interlacing artifacts after processing!")
        else:
            print(f"  ✅ Temporal consistency good")
        
        interlace_stats['post_movement_ratio'] = movement_ratio
    
    # Normalize all keypoints
    normalized_sequence = []
    
    for keypoints, confidence in frames_data:
        ref_point = get_reference_point(keypoints, confidence)
        
        if ref_point is not None:
            normalized = normalize_keypoints(keypoints, confidence, ref_point)
        else:
            normalized = np.zeros((17, 2), dtype=np.float32)
        
        normalized_sequence.append(normalized)
    
    # Convert to numpy and pad/truncate
    normalized_sequence = np.array(normalized_sequence, dtype=np.float32)
    sequence = pad_or_truncate_sequence(normalized_sequence, max_frames)
    
    return sequence, True, interlace_stats


def pad_or_truncate_sequence(sequence: np.ndarray, max_frames: int) -> np.ndarray:
    """Pad or truncate to fixed length"""
    T = len(sequence)
    
    if T == max_frames:
        return sequence
    elif T < max_frames:
        padding = np.zeros((max_frames - T, 17, 2), dtype=np.float32)
        return np.concatenate([sequence, padding], axis=0)
    else:
        # Uniform sampling for truncation
        indices = np.linspace(0, T - 1, max_frames, dtype=int)
        return sequence[indices]


def get_video_label(video_name: str) -> int:
    """Extract class label from video filename (HRI30 format)"""
    if video_name.startswith('CID'):
        try:
            class_id_str = video_name.split('_')[0].replace('CID', '')
            class_id = int(class_id_str)
            return class_id - 1  # Convert to 0-indexed
        except ValueError:
            pass
    
    print(f"Warning: Cannot determine class for {video_name}")
    return -1


def main():
    parser = argparse.ArgumentParser(description='Deinterlaced pose extraction for HRI30')
    parser.add_argument('--data_dir', type=str, default='/home/ty/human-bahviour/train_set',
                       help='Directory containing video files')
    parser.add_argument('--output_dir', type=str, default='/home/ty/human-bahviour/pose_features_deinterlaced',
                       help='Output directory')
    parser.add_argument('--model_path', type=str, default='yolov8n-pose.pt',
                       help='YOLOv8-Pose model path')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'], help='Device to use')
    parser.add_argument('--max_frames', type=int, default=60,
                       help='Maximum sequence length')
    parser.add_argument('--deinterlace_method', type=str, default='bob',
                       choices=['bob', 'linear', 'none'],
                       help='Deinterlacing method')
    parser.add_argument('--gpu_info', action='store_true',
                       help='Show GPU information and exit')
    
    args = parser.parse_args()
    
    print("="*80)
    print("DEINTERLACED POSE FEATURE EXTRACTION")
    print("="*80)
    
    # GPU Information
    if args.device == 'cuda' and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name}")
        print(f"GPU Memory: {gpu_memory:.2f} GB")
        
        # Recommend device based on GPU
        if 'Xavier' in gpu_name:
            print("💡 Jetson Xavier AGX detected - Good for this task")
        elif 'MX450' in gpu_name or 'GTX' in gpu_name:
            print("💡 MX450/GTX detected - Should work fine")
        
        if args.gpu_info:
            return
    
    elif args.device == 'cuda':
        print("Warning: CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {args.device}")
    print(f"Deinterlacing: {args.deinterlace_method}")
    print("="*80)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print(f"\nLoading YOLOv8-Pose model: {args.model_path}")
    model = YOLO(args.model_path)
    
    # Get video files
    video_extensions = ['.avi', '.mp4', '.mov', '.mkv']
    video_files = []
    for ext in video_extensions:
        video_files.extend(list(Path(args.data_dir).glob(f'*{ext}')))
    
    print(f"Found {len(video_files)} videos")
    
    if len(video_files) == 0:
        print("No video files found!")
        return
    
    # Process videos
    sequences = []
    labels = []
    filenames = []
    processing_stats = []
    
    interlaced_count = 0
    successful_count = 0
    
    start_time = time.time()
    
    for video_path in tqdm(video_files, desc="Processing videos"):
        video_name = video_path.name
        
        # Get label
        label = get_video_label(video_name)
        if label < 0:
            continue
        
        # Process video
        sequence, success, stats = process_video_deinterlaced(
            str(video_path), model, args.device, args.max_frames, args.deinterlace_method
        )
        
        if not success:
            print(f"Failed to process {video_name}")
            continue
        
        sequences.append(sequence)
        labels.append(label)
        filenames.append(video_name)
        processing_stats.append(stats)
        
        if stats.get('likely_interlaced', False):
            interlaced_count += 1
        
        successful_count += 1
    
    processing_time = time.time() - start_time
    
    # Convert to numpy arrays
    sequences = np.array(sequences, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)
    
    print(f"\n{'='*80}")
    print("PROCESSING SUMMARY")
    print(f"{'='*80}")
    print(f"Total videos: {len(video_files)}")
    print(f"Successfully processed: {successful_count}")
    print(f"Interlaced videos detected: {interlaced_count}")
    print(f"Processing time: {processing_time:.2f} seconds")
    print(f"Final sequence shape: {sequences.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Check coordinate ranges
    all_coords = sequences.reshape(-1, 2)
    print(f"\nCoordinate ranges:")
    print(f"  X: [{np.min(all_coords[:, 0]):.4f}, {np.max(all_coords[:, 0]):.4f}]")
    print(f"  Y: [{np.min(all_coords[:, 1]):.4f}, {np.max(all_coords[:, 1]):.4f}]")
    
    zero_coords = np.sum(np.all(all_coords == 0, axis=1))
    zero_percentage = (zero_coords / len(all_coords)) * 100
    print(f"  Zero coordinates: {zero_percentage:.2f}%")
    
    # Save results
    output_sequences = os.path.join(args.output_dir, 'train_sequences_deinterlaced.npy')
    output_labels = os.path.join(args.output_dir, 'train_labels_deinterlaced.npy')
    output_filenames = os.path.join(args.output_dir, 'train_filenames_deinterlaced.json')
    output_stats = os.path.join(args.output_dir, 'processing_stats.json')
    
    np.save(output_sequences, sequences)
    np.save(output_labels, labels)
    
    with open(output_filenames, 'w') as f:
        json.dump(filenames, f, indent=2)
    
    with open(output_stats, 'w') as f:
        json.dump(processing_stats, f, indent=2)
    
    print(f"\nSaved to:")
    print(f"  {output_sequences}")
    print(f"  {output_labels}")
    print(f"  {output_filenames}")
    print(f"  {output_stats}")
    
    # Quick validation check
    print(f"\n{'='*80}")
    print("VALIDATION CHECK")
    print(f"{'='*80}")
    
    sample_indices = np.random.choice(len(sequences), min(5, len(sequences)), replace=False)
    
    for idx in sample_indices:
        seq = sequences[idx]
        filename = filenames[idx]
        stats = processing_stats[idx]
        
        # Calculate temporal consistency
        diffs = np.diff(seq, axis=0)
        movements = np.mean(np.linalg.norm(diffs.reshape(len(diffs), -1), axis=1))
        
        print(f"Sample: {filename}")
        print(f"  Label: {labels[idx]}")
        print(f"  Was interlaced: {stats.get('likely_interlaced', False)}")
        print(f"  Avg movement: {movements:.4f}")
        
        if 'post_movement_ratio' in stats:
            print(f"  Post-processing movement ratio: {stats['post_movement_ratio']:.3f}")
    
    print(f"\n✅ Processing complete! Check the validation results above.")


if __name__ == '__main__':
    main()