#!/usr/bin/env python3
"""
Check for interlacing artifacts in pose data
"""

import numpy as np
import matplotlib.pyplot as plt
import json

def check_temporal_consistency(sequences, labels, filenames, sample_count=5):
    """Check for temporal inconsistencies that might indicate interlacing"""
    
    print(f"Analyzing {len(sequences)} sequences...")
    print(f"Sequence shape: {sequences.shape}")
    
    # 1. Check for sudden jumps in pose coordinates
    temporal_jumps = []
    
    for i in range(min(sample_count, len(sequences))):
        seq = sequences[i]  # (60, 17, 2)
        
        # Calculate frame-to-frame differences
        diffs = np.diff(seq, axis=0)  # (59, 17, 2)
        
        # Calculate magnitude of movement per frame
        movement_magnitudes = np.linalg.norm(diffs, axis=2)  # (59, 17)
        
        # Average movement per frame
        avg_movement = np.mean(movement_magnitudes, axis=1)  # (59,)
        
        temporal_jumps.append(avg_movement)
        
        print(f"\nSample {i+1}: {filenames[i]}")
        print(f"Label: {labels[i]}")
        print(f"Max movement: {np.max(avg_movement):.4f}")
        print(f"Mean movement: {np.mean(avg_movement):.4f}")
        print(f"Std movement: {np.std(avg_movement):.4f}")
        
        # Check for alternating pattern (interlacing signature)
        if len(avg_movement) > 10:
            even_frames = avg_movement[::2]
            odd_frames = avg_movement[1::2][:len(even_frames)]
            
            even_mean = np.mean(even_frames)
            odd_mean = np.mean(odd_frames)
            
            print(f"Even frames mean movement: {even_mean:.4f}")
            print(f"Odd frames mean movement: {odd_mean:.4f}")
            print(f"Even/Odd ratio: {even_mean/odd_mean:.4f}")
            
            # Interlacing signature: alternating high/low movement
            if abs(even_mean/odd_mean - 1.0) > 0.3:
                print("⚠️  POTENTIAL INTERLACING DETECTED!")
            else:
                print("✅ Temporal consistency looks good")
    
    return temporal_jumps

def check_coordinate_ranges(sequences):
    """Check if coordinates are properly normalized"""
    
    print("\n" + "="*50)
    print("COORDINATE RANGE ANALYSIS")
    print("="*50)
    
    # Flatten all coordinates
    all_coords = sequences.reshape(-1, 2)  # (N*60*17, 2)
    
    print(f"All coordinates shape: {all_coords.shape}")
    print(f"X range: [{np.min(all_coords[:, 0]):.4f}, {np.max(all_coords[:, 0]):.4f}]")
    print(f"Y range: [{np.min(all_coords[:, 1]):.4f}, {np.max(all_coords[:, 1]):.4f}]")
    
    # Check for too many zeros (missing detections)
    zero_coords = np.sum(np.all(all_coords == 0, axis=1))
    total_coords = len(all_coords)
    zero_percentage = (zero_coords / total_coords) * 100
    
    print(f"Zero coordinates: {zero_coords}/{total_coords} ({zero_percentage:.2f}%)")
    
    if zero_percentage > 30:
        print("⚠️  HIGH ZERO PERCENTAGE - Many missing detections!")
    else:
        print("✅ Reasonable detection rate")

def main():
    print("="*80)
    print("INTERLACING ARTIFACT CHECKER")
    print("="*80)
    
    # Load updated data
    print("Loading updated pose data...")
    sequences = np.load('pose_features/train_sequences_updated.npy')
    labels = np.load('pose_features/train_labels_updated.npy')
    with open('pose_features/train_filenames_updated.json', 'r') as f:
        filenames = json.load(f)
    
    print(f"Loaded {len(sequences)} sequences")
    
    # Check coordinate ranges
    check_coordinate_ranges(sequences)
    
    # Check temporal consistency
    print("\n" + "="*50)
    print("TEMPORAL CONSISTENCY ANALYSIS")
    print("="*50)
    
    temporal_jumps = check_temporal_consistency(sequences, labels, filenames)
    
    # Focus on newly added CID29/CID30 data
    print("\n" + "="*50)
    print("CHECKING NEWLY ADDED CID29/CID30 DATA")
    print("="*50)
    
    cid_indices = []
    for i, filename in enumerate(filenames):
        if filename.startswith('CID29') or filename.startswith('CID30'):
            cid_indices.append(i)
    
    print(f"Found {len(cid_indices)} CID29/CID30 files")
    
    if len(cid_indices) > 0:
        cid_sequences = sequences[cid_indices]
        cid_labels = labels[cid_indices]
        cid_filenames = [filenames[i] for i in cid_indices]
        
        check_temporal_consistency(cid_sequences, cid_labels, cid_filenames, len(cid_indices))

if __name__ == '__main__':
    main()