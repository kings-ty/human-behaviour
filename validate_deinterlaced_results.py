#!/usr/bin/env python3
"""
Validation script to compare original vs deinterlaced pose data
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_temporal_consistency(sequences, filenames, title="Dataset"):
    """Analyze temporal consistency of pose sequences"""
    
    print(f"\n{'='*50}")
    print(f"{title} ANALYSIS")
    print(f"{'='*50}")
    
    # Calculate movement statistics
    all_movements = []
    interlacing_suspects = []
    
    for i, seq in enumerate(sequences[:50]):  # Check first 50 sequences
        # Calculate frame-to-frame differences
        diffs = np.diff(seq, axis=0)  # (59, 17, 2)
        movement_magnitudes = np.linalg.norm(diffs.reshape(len(diffs), -1), axis=1)  # (59,)
        
        all_movements.extend(movement_magnitudes)
        
        # Check for alternating pattern
        if len(movement_magnitudes) > 10:
            even_movements = movement_magnitudes[::2]
            odd_movements = movement_magnitudes[1::2][:len(even_movements)]
            
            even_mean = np.mean(even_movements)
            odd_mean = np.mean(odd_movements)
            
            if odd_mean > 0:
                ratio = even_mean / odd_mean
                if abs(ratio - 1.0) > 0.3:  # Interlacing threshold
                    interlacing_suspects.append({
                        'filename': filenames[i],
                        'ratio': ratio,
                        'even_mean': even_mean,
                        'odd_mean': odd_mean,
                        'max_movement': np.max(movement_magnitudes)
                    })
    
    # Overall statistics
    all_movements = np.array(all_movements)
    
    print(f"Sequences analyzed: {len(sequences)}")
    print(f"Total frame transitions: {len(all_movements)}")
    print(f"Movement statistics:")
    print(f"  Mean: {np.mean(all_movements):.4f}")
    print(f"  Std: {np.std(all_movements):.4f}")
    print(f"  Max: {np.max(all_movements):.4f}")
    print(f"  95th percentile: {np.percentile(all_movements, 95):.4f}")
    
    print(f"\nInterlacing suspects: {len(interlacing_suspects)}")
    
    if len(interlacing_suspects) > 0:
        print("\nTop 5 suspects:")
        sorted_suspects = sorted(interlacing_suspects, key=lambda x: abs(x['ratio'] - 1.0), reverse=True)
        
        for suspect in sorted_suspects[:5]:
            print(f"  {suspect['filename']}")
            print(f"    Even/Odd ratio: {suspect['ratio']:.3f}")
            print(f"    Max movement: {suspect['max_movement']:.3f}")
    
    return {
        'movement_stats': {
            'mean': np.mean(all_movements),
            'std': np.std(all_movements),
            'max': np.max(all_movements),
            'p95': np.percentile(all_movements, 95)
        },
        'interlacing_suspects': len(interlacing_suspects),
        'suspect_details': interlacing_suspects[:10]  # Store top 10
    }

def compare_coordinate_ranges(orig_sequences, new_sequences):
    """Compare coordinate ranges between datasets"""
    
    print(f"\n{'='*50}")
    print("COORDINATE RANGE COMPARISON")
    print(f"{'='*50}")
    
    orig_coords = orig_sequences.reshape(-1, 2)
    new_coords = new_sequences.reshape(-1, 2)
    
    print("Original data:")
    print(f"  Shape: {orig_sequences.shape}")
    print(f"  X range: [{np.min(orig_coords[:, 0]):.4f}, {np.max(orig_coords[:, 0]):.4f}]")
    print(f"  Y range: [{np.min(orig_coords[:, 1]):.4f}, {np.max(orig_coords[:, 1]):.4f}]")
    print(f"  Zero coords: {np.sum(np.all(orig_coords == 0, axis=1)) / len(orig_coords) * 100:.2f}%")
    print(f"  Properly normalized: {np.max(np.abs(orig_coords)) <= 1.0 or np.max(orig_coords) > 10}")
    
    print("\nDeinterlaced data:")
    print(f"  Shape: {new_sequences.shape}")
    print(f"  X range: [{np.min(new_coords[:, 0]):.4f}, {np.max(new_coords[:, 0]):.4f}]")
    print(f"  Y range: [{np.min(new_coords[:, 1]):.4f}, {np.max(new_coords[:, 1]):.4f}]")
    print(f"  Zero coords: {np.sum(np.all(new_coords == 0, axis=1)) / len(new_coords) * 100:.2f}%")
    print(f"  Properly normalized: {np.max(np.abs(new_coords)) <= 1.0}")
    
    # Check improvement
    orig_unnormalized = np.max(orig_coords) > 10
    new_normalized = np.max(np.abs(new_coords)) <= 1.0
    
    if orig_unnormalized and new_normalized:
        print("\n✅ IMPROVEMENT: Coordinates now properly normalized!")
    elif not orig_unnormalized and new_normalized:
        print("\n✅ MAINTAINED: Coordinates remain properly normalized")
    else:
        print("\n⚠️  WARNING: Coordinate normalization may have issues")

def main():
    print("="*80)
    print("DEINTERLACED POSE DATA VALIDATION")
    print("="*80)
    
    # Check if files exist
    orig_seq_path = 'pose_features/train_sequences_updated.npy'
    orig_files_path = 'pose_features/train_filenames_updated.json'
    new_seq_path = 'pose_features_deinterlaced/train_sequences_deinterlaced.npy'
    new_files_path = 'pose_features_deinterlaced/train_filenames_deinterlaced.json'
    
    missing_files = []
    for path in [orig_seq_path, orig_files_path, new_seq_path, new_files_path]:
        if not Path(path).exists():
            missing_files.append(path)
    
    if missing_files:
        print("❌ Missing required files:")
        for path in missing_files:
            print(f"  - {path}")
        print("\nPlease run the preprocessing first!")
        return
    
    # Load data
    print("Loading data...")
    
    orig_sequences = np.load(orig_seq_path)
    with open(orig_files_path) as f:
        orig_filenames = json.load(f)
    
    new_sequences = np.load(new_seq_path)
    with open(new_files_path) as f:
        new_filenames = json.load(f)
    
    print(f"Original data: {len(orig_sequences)} sequences")
    print(f"Deinterlaced data: {len(new_sequences)} sequences")
    
    # Compare coordinate ranges
    compare_coordinate_ranges(orig_sequences, new_sequences)
    
    # Analyze temporal consistency
    orig_analysis = analyze_temporal_consistency(orig_sequences, orig_filenames, "ORIGINAL")
    new_analysis = analyze_temporal_consistency(new_sequences, new_filenames, "DEINTERLACED")
    
    # Summary comparison
    print(f"\n{'='*50}")
    print("IMPROVEMENT SUMMARY")
    print(f"{'='*50}")
    
    orig_suspects = orig_analysis['interlacing_suspects']
    new_suspects = new_analysis['interlacing_suspects']
    
    print(f"Interlacing suspects:")
    print(f"  Original: {orig_suspects}")
    print(f"  Deinterlaced: {new_suspects}")
    print(f"  Improvement: {orig_suspects - new_suspects} fewer suspects")
    
    orig_max_movement = orig_analysis['movement_stats']['max']
    new_max_movement = new_analysis['movement_stats']['max']
    
    print(f"\nMaximum movement:")
    print(f"  Original: {orig_max_movement:.3f}")
    print(f"  Deinterlaced: {new_max_movement:.3f}")
    
    if new_max_movement < orig_max_movement * 0.5:
        print("  ✅ Significantly reduced extreme movements")
    elif new_max_movement < orig_max_movement:
        print("  ✅ Reduced extreme movements")
    else:
        print("  ⚠️  Extreme movements not significantly reduced")
    
    # Final recommendation
    print(f"\n{'='*50}")
    print("RECOMMENDATION")
    print(f"{'='*50}")
    
    coord_improved = np.max(np.abs(new_sequences)) <= 1.0
    suspects_reduced = new_suspects < orig_suspects * 0.7
    movement_improved = new_max_movement < orig_max_movement * 0.8
    
    improvements = sum([coord_improved, suspects_reduced, movement_improved])
    
    if improvements >= 2:
        print("✅ RECOMMENDED: Use deinterlaced data for LSTM training")
        print("   - Significant improvements detected")
    elif improvements >= 1:
        print("🔄 CONSIDER: Deinterlaced data shows some improvements")
        print("   - Run LSTM training comparison to confirm benefits")
    else:
        print("❓ UNCLEAR: Limited improvements detected")
        print("   - May need to check deinterlacing parameters")
    
    print(f"\nImprovements detected: {improvements}/3")
    print(f"  - Proper normalization: {coord_improved}")
    print(f"  - Reduced interlacing suspects: {suspects_reduced}")
    print(f"  - Reduced extreme movements: {movement_improved}")

if __name__ == '__main__':
    main()