#!/usr/bin/env python3
"""
OPTIMIZED BATCH POSE FEATURE EXTRACTION
Fixes for pose artifacts + efficient batch processing for 2099 videos

Key Improvements:
1. Deinterlacing to remove interlacing artifacts
2. Temporal smoothing to reduce artifacts
3. Better reference point stability
4. Outlier detection and filtering
5. Batch processing with resume capability
6. Memory optimization for large datasets
7. Progress tracking and validation
"""

import os
import cv2
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import json
import argparse
import time
from concurrent.futures import ThreadPoolExecutor
import threading
from scipy.signal import savgol_filter
import pickle

from ultralytics import YOLO


# Global configuration
KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

class OptimizedPoseExtractor:
    def __init__(self, model_path='yolov8n-pose.pt', device='cuda', deinterlace_method='bob'):
        self.device = device
        self.model = YOLO(model_path)
        self.deinterlace_method = deinterlace_method

        # Optimization settings
        self.min_confidence = 0.3
        self.outlier_threshold = 3.0  # Standard deviations for outlier detection
        self.smoothing_window = 5     # For temporal smoothing

    def deinterlace_frame(self, frame):
        """
        Apply deinterlacing to a frame
        """
        if self.deinterlace_method == 'bob':
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

        elif self.deinterlace_method == 'linear':
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

    def get_stable_reference_point(self, keypoints_sequence, confidence_sequence):
        """
        Get more stable reference point using temporal information
        """
        valid_frames = []
        
        for kpts, conf in zip(keypoints_sequence, confidence_sequence):
            # Try hip center first (most stable)
            left_hip_conf = conf[11]
            right_hip_conf = conf[12]

            if left_hip_conf > self.min_confidence and right_hip_conf > self.min_confidence:
                hip_center = (np.array(kpts[11]) + np.array(kpts[12])) / 2.0
                valid_frames.append(hip_center)
            elif left_hip_conf > self.min_confidence:
                valid_frames.append(np.array(kpts[11]))
            elif right_hip_conf > self.min_confidence:
                valid_frames.append(np.array(kpts[12]))
        
        if len(valid_frames) > 0:
            # Use median for stability
            reference_points = np.array(valid_frames)
            return np.median(reference_points, axis=0)
        
        # Fallback to shoulder center
        valid_frames = []
        for kpts, conf in zip(keypoints_sequence, confidence_sequence):
            left_shoulder_conf = conf[5]
            right_shoulder_conf = conf[6]

            if left_shoulder_conf > self.min_confidence and right_shoulder_conf > self.min_confidence:
                shoulder_center = (np.array(kpts[5]) + np.array(kpts[6])) / 2.0
                valid_frames.append(shoulder_center)
        
        if len(valid_frames) > 0:
            reference_points = np.array(valid_frames)
            return np.median(reference_points, axis=0)
        
        return None
    
    def detect_and_filter_outliers(self, keypoints_sequence):
        """
        Detect and filter outliers using movement analysis
        """
        if len(keypoints_sequence) < 3:
            return keypoints_sequence
        
        # Calculate frame-to-frame movements
        movements = []
        for i in range(1, len(keypoints_sequence)):
            prev_kpts = keypoints_sequence[i-1].reshape(-1, 2)
            curr_kpts = keypoints_sequence[i].reshape(-1, 2)
            
            # Calculate movement magnitude per keypoint
            movement = np.linalg.norm(curr_kpts - prev_kpts, axis=1)
            avg_movement = np.mean(movement)
            movements.append(avg_movement)
        
        if len(movements) < 3:
            return keypoints_sequence
        
        # Identify outliers using statistical method
        movements = np.array(movements)
        median_movement = np.median(movements)
        mad = np.median(np.abs(movements - median_movement))
        
        # Modified Z-score for outlier detection
        threshold = self.outlier_threshold
        outlier_indices = []
        
        for i, movement in enumerate(movements):
            if mad > 0:
                modified_z_score = 0.6745 * (movement - median_movement) / mad
                if abs(modified_z_score) > threshold:
                    outlier_indices.append(i + 1)  # +1 because movements is offset by 1
        
        # Replace outliers with interpolated values
        filtered_sequence = keypoints_sequence.copy()
        
        for idx in outlier_indices:
            if 0 < idx < len(filtered_sequence) - 1:
                # Linear interpolation between neighbors
                prev_kpts = filtered_sequence[idx-1]
                next_kpts = filtered_sequence[idx+1]
                filtered_sequence[idx] = (prev_kpts + next_kpts) / 2.0
        
        return filtered_sequence
    
    def apply_temporal_smoothing(self, keypoints_sequence):
        """
        Apply temporal smoothing to reduce jitter
        """
        if len(keypoints_sequence) < self.smoothing_window:
            return keypoints_sequence
        
        # Reshape for processing
        sequence_array = np.array(keypoints_sequence)  # (T, 17, 2)
        T, num_joints, coords = sequence_array.shape
        
        # Apply Savitzky-Golay filter for smoothing
        smoothed_sequence = np.zeros_like(sequence_array)
        
        for joint in range(num_joints):
            for coord in range(coords):
                signal = sequence_array[:, joint, coord]
                
                # Only smooth if we have enough points and signal is not all zeros
                if len(signal) >= self.smoothing_window and np.any(signal != 0):
                    try:
                        # Use smaller window for short sequences
                        window_size = min(self.smoothing_window, len(signal))
                        if window_size % 2 == 0:
                            window_size -= 1  # Must be odd
                        
                        if window_size >= 3:
                            smoothed = savgol_filter(signal, window_size, 2)
                            smoothed_sequence[:, joint, coord] = smoothed
                        else:
                            smoothed_sequence[:, joint, coord] = signal
                    except:
                        smoothed_sequence[:, joint, coord] = signal
                else:
                    smoothed_sequence[:, joint, coord] = signal
        
        return smoothed_sequence.tolist()
    
    def normalize_keypoints_improved(self, keypoints, reference_point, scale_factor=150.0):
        """
        Improved normalization with better scaling
        """
        if reference_point is None:
            return np.zeros((17, 2), dtype=np.float32)
        
        # Center-relative coordinates  
        normalized = keypoints - reference_point
        
        # Dynamic scaling based on actual pose size
        pose_size = np.max(np.abs(normalized))
        if pose_size > 0:
            # Adaptive scaling
            actual_scale = min(scale_factor, pose_size * 1.2)
            normalized = normalized / actual_scale
        else:
            normalized = normalized / scale_factor
        
        # Ensure [-1, 1] range
        normalized = np.clip(normalized, -1.0, 1.0)
        
        return normalized.astype(np.float32)
    
    def process_single_video(self, video_path, max_frames=60):
        """
        Process a single video with all optimizations
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            return None, f"Cannot open video {video_path}"
        
        # Extract all frames first
        raw_keypoints = []
        raw_confidences = []
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Apply deinterlacing
            frame = self.deinterlace_frame(frame)

            # YOLOv8 inference
            results = self.model(frame, device=self.device, verbose=False)
            
            if len(results) > 0 and results[0].keypoints is not None:
                keypoints_data = results[0].keypoints
                
                if keypoints_data.xy is not None and len(keypoints_data.xy) > 0:
                    kpts = keypoints_data.xy[0].cpu().numpy()
                    conf = keypoints_data.conf[0].cpu().numpy() if keypoints_data.conf is not None else np.ones(17)
                else:
                    kpts = np.zeros((17, 2), dtype=np.float32)
                    conf = np.zeros(17, dtype=np.float32)
            else:
                kpts = np.zeros((17, 2), dtype=np.float32)
                conf = np.zeros(17, dtype=np.float32)
            
            raw_keypoints.append(kpts)
            raw_confidences.append(conf)
            frame_count += 1
        
        cap.release()
        
        if len(raw_keypoints) == 0:
            return None, "No frames extracted"
        
        # Apply optimizations
        # 1. Outlier filtering
        filtered_keypoints = self.detect_and_filter_outliers(raw_keypoints)
        
        # 2. Temporal smoothing
        smoothed_keypoints = self.apply_temporal_smoothing(filtered_keypoints)
        
        # 3. Get stable reference point
        reference_point = self.get_stable_reference_point(smoothed_keypoints, raw_confidences)
        
        # 4. Normalize all keypoints
        normalized_sequence = []
        for kpts, conf in zip(smoothed_keypoints, raw_confidences):
            normalized = self.normalize_keypoints_improved(np.array(kpts), reference_point)
            
            # Zero out low-confidence keypoints
            mask = conf < self.min_confidence
            normalized[mask] = 0.0
            
            normalized_sequence.append(normalized)
        
        # 5. Pad or truncate to fixed length
        normalized_sequence = np.array(normalized_sequence)
        final_sequence = self.pad_or_truncate_sequence(normalized_sequence, max_frames)
        
        return final_sequence, None
    
    def pad_or_truncate_sequence(self, sequence, max_frames):
        """Pad or truncate to fixed length"""
        T = len(sequence)
        
        if T == max_frames:
            return sequence
        elif T < max_frames:
            # Pad with zeros
            padding = np.zeros((max_frames - T, 17, 2), dtype=np.float32)
            return np.concatenate([sequence, padding], axis=0)
        else:
            # Uniform sampling for truncation
            indices = np.linspace(0, T - 1, max_frames, dtype=int)
            return sequence[indices]

def get_video_label(video_name):
    """Extract class label from video filename"""
    if video_name.startswith('CID'):
        try:
            class_id_str = video_name.split('_')[0].replace('CID', '')
            class_id = int(class_id_str)
            return class_id - 1  # Convert to 0-indexed
        except ValueError:
            pass
    
    return -1

def process_video_batch(video_paths, extractor, batch_id, progress_callback=None):
    """Process a batch of videos"""
    results = []
    failed_videos = []

    for i, video_path in enumerate(video_paths):
        video_name = video_path.name
        label = get_video_label(video_name)

        if label < 0:
            continue

        sequence, error = extractor.process_single_video(video_path)

        if error is None:
            results.append({
                'sequence': sequence,
                'label': label,
                'filename': video_name
            })
        else:
            print(f"Failed {video_name}: {error}")
            failed_videos.append({
                'filename': video_name,
                'path': str(video_path),
                'error': error,
                'batch_id': batch_id
            })

        if progress_callback:
            progress_callback(batch_id, i + 1, len(video_paths))

    return results, failed_videos

def save_batch_results(batch_results, batch_id, output_dir):
    """Save batch results to disk"""
    batch_sequences = [r['sequence'] for r in batch_results]
    batch_labels = [r['label'] for r in batch_results]
    batch_filenames = [r['filename'] for r in batch_results]
    
    if len(batch_sequences) > 0:
        sequences_array = np.array(batch_sequences, dtype=np.float32)
        labels_array = np.array(batch_labels, dtype=np.int64)
        
        batch_file = output_dir / f'batch_{batch_id:03d}.npz'
        np.savez_compressed(batch_file, 
                           sequences=sequences_array,
                           labels=labels_array,
                           filenames=batch_filenames)
        
        return len(batch_sequences), batch_file
    
    return 0, None

def merge_batch_files(output_dir, final_output_dir):
    """Merge all batch files into final arrays"""
    
    batch_files = sorted(list(output_dir.glob('batch_*.npz')))
    
    if len(batch_files) == 0:
        print("❌ No batch files found")
        return
    
    print(f"Merging {len(batch_files)} batch files...")
    
    all_sequences = []
    all_labels = []
    all_filenames = []
    
    for batch_file in tqdm(batch_files, desc="Merging batches"):
        data = np.load(batch_file)
        all_sequences.append(data['sequences'])
        all_labels.extend(data['labels'])
        all_filenames.extend(data['filenames'])
    
    # Concatenate all sequences
    final_sequences = np.concatenate(all_sequences, axis=0)
    final_labels = np.array(all_labels, dtype=np.int64)
    
    print(f"Final dataset: {len(final_sequences)} sequences")
    
    # Save final results
    os.makedirs(final_output_dir, exist_ok=True)
    
    np.save(final_output_dir / 'train_sequences_optimized.npy', final_sequences)
    np.save(final_output_dir / 'train_labels_optimized.npy', final_labels)
    with open(final_output_dir / 'train_filenames_optimized.json', 'w') as f:
        json.dump(all_filenames, f, indent=2)
    
    # Validation check
    coords = final_sequences.reshape(-1, 2)
    zero_percentage = np.sum(np.all(coords == 0, axis=1)) / len(coords) * 100
    
    print(f"\nValidation:")
    print(f"  Coordinate range: [{np.min(coords):.4f}, {np.max(coords):.4f}]")
    print(f"  Zero coordinates: {zero_percentage:.2f}%")
    print(f"  Properly normalized: {np.max(np.abs(coords)) <= 1.0}")
    
    return final_sequences, final_labels, all_filenames

def main():
    parser = argparse.ArgumentParser(description='Optimized batch pose extraction')
    parser.add_argument('--data_dir', type=str, default='train_set',
                       help='Directory containing videos')
    parser.add_argument('--output_dir', type=str, default='pose_features_optimized',
                       help='Output directory')
    parser.add_argument('--batch_size', type=int, default=50,
                       help='Videos per batch')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--deinterlace_method', type=str, default='bob',
                       choices=['bob', 'linear', 'none'],
                       help='Deinterlacing method')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from existing batches')
    
    args = parser.parse_args()
    
    print("="*80)
    print("OPTIMIZED BATCH POSE EXTRACTION")
    print("="*80)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {args.device}")
    print(f"Deinterlacing: {args.deinterlace_method}")
    
    # Setup directories
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    batch_dir = output_dir / 'batches'
    
    os.makedirs(batch_dir, exist_ok=True)
    
    # Find video files
    video_files = []
    for ext in ['.avi', '.mp4', '.mov', '.mkv']:
        video_files.extend(list(data_dir.glob(f'*{ext}')))
    
    print(f"Found {len(video_files)} videos")
    
    # Check for existing batches if resuming
    start_batch = 0
    if args.resume:
        existing_batches = list(batch_dir.glob('batch_*.npz'))
        if len(existing_batches) > 0:
            start_batch = max([int(f.stem.split('_')[1]) for f in existing_batches]) + 1
            print(f"Resuming from batch {start_batch}")
    
    # Create batches
    batches = []
    for i in range(0, len(video_files), args.batch_size):
        batch = video_files[i:i + args.batch_size]
        batches.append(batch)
    
    print(f"Total batches: {len(batches)}")
    
    # Initialize pose extractor
    extractor = OptimizedPoseExtractor(device=args.device, deinterlace_method=args.deinterlace_method)
    
    # Process batches
    total_processed = 0
    all_failed_videos = []
    start_time = time.time()

    for batch_id in range(start_batch, len(batches)):
        batch_start_time = time.time()

        print(f"\nProcessing batch {batch_id+1}/{len(batches)}")

        batch_results, failed_videos = process_video_batch(batches[batch_id], extractor, batch_id)
        all_failed_videos.extend(failed_videos)

        count, batch_file = save_batch_results(batch_results, batch_id, batch_dir)
        total_processed += count

        batch_time = time.time() - batch_start_time

        print(f"  Processed: {count}/{len(batches[batch_id])} videos in {batch_time:.2f}s")
        if len(failed_videos) > 0:
            print(f"  Failed: {len(failed_videos)} videos")
        print(f"  Total so far: {total_processed}")

        if batch_file:
            print(f"  Saved: {batch_file}")
    
    # Merge all batches
    print(f"\nMerging all batches...")
    final_sequences, final_labels, final_filenames = merge_batch_files(batch_dir, output_dir)
    
    total_time = time.time() - start_time

    # Save failed videos list
    if len(all_failed_videos) > 0:
        failed_log_path = output_dir / 'failed_videos.json'
        with open(failed_log_path, 'w') as f:
            json.dump(all_failed_videos, f, indent=2)
        print(f"\n⚠️  Failed videos log saved: {failed_log_path}")

    print(f"\n{'='*80}")
    print("PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"Total videos processed: {total_processed}")
    print(f"Total videos failed: {len(all_failed_videos)}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Speed: {total_processed/total_time:.2f} videos/second")
    
    print(f"\nOutput files:")
    print(f"  train_sequences_optimized.npy")
    print(f"  train_labels_optimized.npy")
    print(f"  train_filenames_optimized.json")

if __name__ == '__main__':
    main()