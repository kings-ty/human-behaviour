#!/usr/bin/env python3
"""
YOLOv8-Pose Feature Extraction for HRI30 Dataset
Extracts skeleton keypoints from videos using YOLOv8-Pose model on Jetson Xavier AGX

This script performs:
1. Video-level pose extraction (17 COCO keypoints per frame)
2. Center-relative normalization (invariant to camera position)
3. Coordinate scaling to [-1, 1] range
4. Occlusion handling (padding with zeros or previous frame)
5. Fixed-length sequence generation (pad/truncate to 60 frames)
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

# Import YOLOv8-Pose from ultralytics
from ultralytics import YOLO


# COCO 17 keypoint format used by YOLOv8-Pose
KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

# Reference points for normalization (indices)
NECK_IDX = 5  # left_shoulder (we'll average with right_shoulder for neck approximation)
HIP_CENTER_IDX = [11, 12]  # left_hip, right_hip


def get_reference_point(keypoints: np.ndarray, confidence: np.ndarray,
                        min_confidence: float = 0.3) -> Optional[np.ndarray]:
    """
    Calculate reference point (neck/hip center) for normalization.

    Why normalization is needed:
    - Makes the model invariant to camera position and distance
    - Focuses on relative pose rather than absolute position
    - Improves generalization across different viewpoints
    - Reduces variance in training data

    Args:
        keypoints: (17, 2) array of [x, y] coordinates
        confidence: (17,) array of confidence scores
        min_confidence: Minimum confidence threshold for valid keypoints

    Returns:
        (2,) array of reference point [x, y] or None if not found
    """
    # Try to use hip center (more stable for industrial actions)
    left_hip = keypoints[11]
    right_hip = keypoints[12]
    left_hip_conf = confidence[11]
    right_hip_conf = confidence[12]

    if left_hip_conf > min_confidence and right_hip_conf > min_confidence:
        # Both hips visible - use center
        return (left_hip + right_hip) / 2.0
    elif left_hip_conf > min_confidence:
        return left_hip
    elif right_hip_conf > min_confidence:
        return right_hip

    # Fallback to shoulder center (neck approximation)
    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]
    left_shoulder_conf = confidence[5]
    right_shoulder_conf = confidence[6]

    if left_shoulder_conf > min_confidence and right_shoulder_conf > min_confidence:
        return (left_shoulder + right_shoulder) / 2.0
    elif left_shoulder_conf > min_confidence:
        return left_shoulder
    elif right_shoulder_conf > min_confidence:
        return right_shoulder

    # No valid reference point found
    return None


def normalize_keypoints(keypoints: np.ndarray, confidence: np.ndarray,
                        reference_point: np.ndarray,
                        scale_factor: float = 100.0) -> np.ndarray:
    """
    Normalize keypoints to be center-relative and scaled to [-1, 1] range.

    Normalization steps:
    1. Subtract reference point (center-relative coordinates)
    2. Scale by typical human body dimensions (~100 pixels)
    3. Clip to [-1, 1] range to handle outliers

    Args:
        keypoints: (17, 2) array of [x, y] coordinates
        confidence: (17,) array of confidence scores
        reference_point: (2,) array of reference [x, y]
        scale_factor: Expected body size in pixels (for scaling)

    Returns:
        (17, 2) array of normalized keypoints
    """
    # Center-relative coordinates (subtract reference point)
    normalized = keypoints - reference_point

    # Scale to approximate [-1, 1] range based on typical body size
    # scale_factor should be roughly half the body height in pixels
    normalized = normalized / scale_factor

    # Clip to [-1, 1] to handle outliers and ensure bounded inputs
    normalized = np.clip(normalized, -1.0, 1.0)

    # Zero out low-confidence keypoints (occlusion handling)
    mask = confidence[:, np.newaxis] < 0.3  # Shape: (17, 1)
    normalized[mask[:, 0]] = 0.0

    return normalized


def deinterlace_frame(frame: np.ndarray, method: str = 'linear') -> np.ndarray:
    """
    Apply deinterlacing to remove interlacing artifacts.

    Interlacing artifacts appear as horizontal lines/combing in videos.
    This is critical for accurate pose detection.

    Args:
        frame: Input frame (H, W, 3) BGR format
        method: 'bob' (fast, recommended) or 'linear' (slower, higher quality)

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
        # Linear interpolation - higher quality
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


def extract_pose_from_frame(model: YOLO, frame: np.ndarray,
                            device: str = 'cuda', deinterlace_method: str = 'linear') -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Extract pose keypoints from a single frame using YOLOv8-Pose.

    Args:
        model: YOLOv8-Pose model
        frame: Input frame (H, W, 3) BGR format
        device: Device to run inference on
        deinterlace_method: Deinterlacing method ('bob', 'linear', or 'none')

    Returns:
        keypoints: (17, 2) array of [x, y] coordinates
        confidence: (17,) array of confidence scores
        detected: Boolean indicating if person was detected
    """
    # Apply deinterlacing to remove artifacts
    if deinterlace_method != 'none':
        frame = deinterlace_frame(frame, deinterlace_method)

    # Run YOLOv8-Pose inference
    results = model(frame, device=device, verbose=False)

    # Check if any person detected
    if len(results) == 0 or results[0].keypoints is None:
        # No detection - return zeros
        return np.zeros((17, 2), dtype=np.float32), np.zeros(17, dtype=np.float32), False

    # Get keypoints from first detected person (highest confidence)
    keypoints_data = results[0].keypoints

    if keypoints_data.xy is None or len(keypoints_data.xy) == 0:
        return np.zeros((17, 2), dtype=np.float32), np.zeros(17, dtype=np.float32), False

    # Extract x, y coordinates and confidence
    kpts = keypoints_data.xy[0].cpu().numpy()  # (17, 2) [x, y]
    conf = keypoints_data.conf[0].cpu().numpy() if keypoints_data.conf is not None else np.ones(17)  # (17,)

    return kpts.astype(np.float32), conf.astype(np.float32), True


def process_video(video_path: str, model: YOLO, device: str = 'cuda',
                 max_frames: int = 60, deinterlace_method: str = 'linear') -> Tuple[np.ndarray, bool]:
    """
    Process a video and extract normalized pose sequences.

    Args:
        video_path: Path to input video file
        model: YOLOv8-Pose model
        device: Device to run inference on
        max_frames: Maximum sequence length (pad/truncate)
        deinterlace_method: Deinterlacing method ('bob', 'linear', or 'none')

    Returns:
        sequence: (max_frames, 17, 2) array of normalized keypoints
        success: Boolean indicating if processing was successful
    """
    video_name = os.path.basename(video_path)

    # Redirect stderr to capture FFmpeg/OpenCV errors
    import sys
    from io import StringIO

    old_stderr = sys.stderr
    sys.stderr = StringIO()

    cap = cv2.VideoCapture(video_path)

    # Capture any stderr output
    stderr_output = sys.stderr.getvalue()
    sys.stderr = old_stderr

    # If there were errors in stderr, print them with video name
    if stderr_output:
        print(f"\n[{video_name}] Video codec warnings/errors:")
        print(stderr_output.strip())

    if not cap.isOpened():
        print(f"Error: Cannot open video {video_name}")
        return np.zeros((max_frames, 17, 2), dtype=np.float32), False

    frames_data = []
    last_valid_keypoints = None
    last_valid_confidence = None

    # Read all frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Extract pose from frame (with deinterlacing)
        keypoints, confidence, detected = extract_pose_from_frame(model, frame, device, deinterlace_method)

        # Handle occlusion/no detection
        if not detected:
            if last_valid_keypoints is not None:
                # Use previous frame's keypoints
                keypoints = last_valid_keypoints.copy()
                confidence = last_valid_confidence.copy()
            else:
                # No valid previous frame - use zeros
                keypoints = np.zeros((17, 2), dtype=np.float32)
                confidence = np.zeros(17, dtype=np.float32)
        else:
            # Update last valid keypoints
            last_valid_keypoints = keypoints.copy()
            last_valid_confidence = confidence.copy()

        frames_data.append((keypoints, confidence))

    cap.release()

    if len(frames_data) == 0:
        print(f"Warning: No frames extracted from {video_path}")
        return np.zeros((max_frames, 17, 2), dtype=np.float32), False

    # Normalize all keypoints
    normalized_sequence = []

    for keypoints, confidence in frames_data:
        # Get reference point for normalization
        ref_point = get_reference_point(keypoints, confidence)

        if ref_point is not None:
            # Normalize keypoints
            normalized = normalize_keypoints(keypoints, confidence, ref_point)
        else:
            # No valid reference - use zero-centered (last resort)
            normalized = np.zeros((17, 2), dtype=np.float32)

        normalized_sequence.append(normalized)

    # Convert to numpy array
    normalized_sequence = np.array(normalized_sequence, dtype=np.float32)  # (T, 17, 2)

    # Pad or truncate to fixed length
    sequence = pad_or_truncate_sequence(normalized_sequence, max_frames)

    return sequence, True


def pad_or_truncate_sequence(sequence: np.ndarray, max_frames: int) -> np.ndarray:
    """
    Pad with zeros or truncate sequence to fixed length.

    Args:
        sequence: (T, 17, 2) variable length sequence
        max_frames: Target sequence length

    Returns:
        (max_frames, 17, 2) fixed length sequence
    """
    T = len(sequence)

    if T == max_frames:
        return sequence
    elif T < max_frames:
        # Pad with zeros
        padding = np.zeros((max_frames - T, 17, 2), dtype=np.float32)
        return np.concatenate([sequence, padding], axis=0)
    else:
        # Truncate (uniformly sample frames)
        indices = np.linspace(0, T - 1, max_frames, dtype=int)
        return sequence[indices]


def load_class_labels(annotations_path: str) -> Dict[str, int]:
    """
    Load class labels from annotations file.

    Args:
        annotations_path: Path to classInd.txt file

    Returns:
        Dictionary mapping class names to indices
    """
    class_to_idx = {}

    with open(annotations_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                idx, class_name = line.split()
                class_to_idx[class_name] = int(idx)

    return class_to_idx


def get_video_label(video_name: str, class_to_idx: Dict[str, int]) -> int:
    """
    Extract class label from video filename.

    Expected format: CID{class_id}_SID{subject_id}_VID{video_id}.{ext}
    or: v_{class_id}_g{group}_c{clip}.{ext}
    or: {class_name}_*.{ext}

    Args:
        video_name: Video filename
        class_to_idx: Dictionary mapping class names to indices

    Returns:
        Class index (0-29 for HRI30) - note: returns 1-based index from CID
    """
    # Try format: CID{class_id}_SID{subject_id}_VID{video_id}.ext (HRI30 format)
    if video_name.startswith('CID'):
        parts = video_name.split('_')
        if len(parts) >= 1:
            try:
                # Extract class_id from CID23 -> 23
                class_id_str = parts[0].replace('CID', '')
                class_id = int(class_id_str)
                # HRI30 uses 1-indexed classes (1-30), need 0-indexed (0-29)
                return class_id - 1
            except ValueError:
                pass

    # Try format: v_{class_id}_g{group}_c{clip}.ext
    if video_name.startswith('v_'):
        parts = video_name.split('_')
        if len(parts) >= 2:
            try:
                class_id = int(parts[1])
                return class_id
            except ValueError:
                pass

    # Try format: {class_name}_*.ext
    for class_name, class_id in class_to_idx.items():
        if video_name.startswith(class_name):
            return class_id

    # Unable to determine class
    print(f"Warning: Cannot determine class for {video_name}")
    return -1


def main():
    parser = argparse.ArgumentParser(
        description='Extract pose features from HRI30 videos using YOLOv8-Pose',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--data_root', type=str, default='/home/ty/human-behaviour',
                       help='Root directory containing train_set and test_set')
    parser.add_argument('--output_dir', type=str, default='/home/ty/human-behaviour/pose_features',
                       help='Output directory for processed .npy files')
    parser.add_argument('--model_path', type=str, default='yolov8n-pose.pt',
                       help='Path to YOLOv8-Pose model weights')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to run inference on')
    parser.add_argument('--max_frames', type=int, default=60,
                       help='Maximum sequence length (frames)')
    parser.add_argument('--batch_mode', type=str, default='train',
                       choices=['train', 'test', 'both'],
                       help='Process train, test, or both sets')
    parser.add_argument('--deinterlace_method', type=str, default='linear',
                       choices=['bob', 'linear', 'none'],
                       help='Deinterlacing method: bob (fast, recommended), linear (slower, better quality), or none')

    args = parser.parse_args()

    print("="*80)
    print("YOLOv8-Pose Feature Extraction for HRI30")
    print("="*80)
    print(f"Data root: {args.data_root}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {args.device}")
    print(f"Max frames: {args.max_frames}")
    print(f"Deinterlacing: {args.deinterlace_method}")
    print("="*80)

    # Check CUDA availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        args.device = 'cpu'

    if args.device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'test'), exist_ok=True)

    # Load YOLOv8-Pose model
    print(f"\nLoading YOLOv8-Pose model: {args.model_path}")
    model = YOLO(args.model_path)

    # Load class labels
    annotations_path = os.path.join(args.data_root, 'annotations', 'classInd.txt')
    class_to_idx = load_class_labels(annotations_path)
    print(f"Loaded {len(class_to_idx)} classes")

    # Process train and/or test sets
    sets_to_process = []
    if args.batch_mode in ['train', 'both']:
        sets_to_process.append('train_set')
    if args.batch_mode in ['test', 'both']:
        sets_to_process.append('test_set')

    for dataset_name in sets_to_process:
        print(f"\n{'='*80}")
        print(f"Processing {dataset_name}")
        print(f"{'='*80}")

        dataset_dir = os.path.join(args.data_root, dataset_name)

        if not os.path.exists(dataset_dir):
            print(f"Warning: {dataset_dir} does not exist, skipping...")
            continue

        # Get all video files
        video_extensions = ['.avi', '.mp4', '.mov', '.mkv']
        video_files = []
        for ext in video_extensions:
            video_files.extend(list(Path(dataset_dir).glob(f'*{ext}')))

        print(f"Found {len(video_files)} videos")

        # Process each video
        sequences = []
        labels = []
        filenames = []

        for video_path in tqdm(video_files, desc=f"Extracting poses"):
            video_name = video_path.name

            # Print which video is being processed (helps identify problematic videos)
            tqdm.write(f"Processing: {video_name}")

            # Extract pose sequence (with deinterlacing)
            sequence, success = process_video(str(video_path), model, args.device, args.max_frames, args.deinterlace_method)

            if not success:
                tqdm.write(f"Failed to process {video_name}")
                continue

            # Get label
            label = get_video_label(video_name, class_to_idx)
            if label is None or label < 0:
                label = -1

            sequences.append(sequence)
            labels.append(label)
            filenames.append(video_name)

        # Convert to numpy arrays
        sequences = np.array(sequences, dtype=np.float32)  # (N, max_frames, 17, 2)
        labels = np.array(labels, dtype=np.int64)  # (N,)

        print(f"\nExtracted {len(sequences)} sequences")
        print(f"Sequence shape: {sequences.shape}")
        print(f"Labels shape: {labels.shape}")

        # Save to disk
        output_name = 'train' if dataset_name == 'train_set' else 'test'
        output_path = os.path.join(args.output_dir, f'{output_name}_sequences.npy')
        labels_path = os.path.join(args.output_dir, f'{output_name}_labels.npy')
        filenames_path = os.path.join(args.output_dir, f'{output_name}_filenames.json')

        np.save(output_path, sequences)
        np.save(labels_path, labels)

        with open(filenames_path, 'w') as f:
            json.dump(filenames, f, indent=2)

        print(f"\nSaved to:")
        print(f"  Sequences: {output_path}")
        print(f"  Labels: {labels_path}")
        print(f"  Filenames: {filenames_path}")

    print(f"\n{'='*80}")
    print("Feature extraction completed!")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
