#!/usr/bin/env python3
"""
Compare original vs optimized pose data to validate improvements
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_temporal_artifacts(sequences, title="Dataset"):
    """Analyze temporal artifacts in pose sequences"""
    
    print(f"\n{'='*50}")
    print(f"{title} ANALYSIS")
    print(f"{'='*50}")
    
    # Sample sequences for analysis
    sample_size = min(100, len(sequences))
    sample_indices = np.random.choice(len(sequences), sample_size, replace=False)
    
    movement_stats = []
    artifact_count = 0
    
    for idx in sample_indices:
        seq = sequences[idx]  # (60, 17, 2)
        
        # Calculate frame-to-frame movements
        diffs = np.diff(seq, axis=0)  # (59, 17, 2)
        movements = np.mean(np.linalg.norm(diffs.reshape(len(diffs), -1), axis=1))
        movement_stats.append(movements)
        
        # Check for artifacts (extreme movements)
        if movements > 0.1:  # Threshold for extreme movement
            artifact_count += 1
    
    movement_stats = np.array(movement_stats)
    
    print(f"Sample sequences: {sample_size}")
    print(f"Movement statistics:")
    print(f"  Mean: {np.mean(movement_stats):.6f}")
    print(f"  Std: {np.std(movement_stats):.6f}")
    print(f"  Max: {np.max(movement_stats):.6f}")
    print(f"  95th percentile: {np.percentile(movement_stats, 95):.6f}")
    print(f"Sequences with artifacts: {artifact_count}/{sample_size} ({artifact_count/sample_size*100:.1f}%)")
    
    return {
        'mean_movement': np.mean(movement_stats),
        'max_movement': np.max(movement_stats),
        'artifact_percentage': artifact_count/sample_size*100,
        'movement_stats': movement_stats
    }

def check_coordinate_quality(sequences, title="Dataset"):
    """Check coordinate quality and normalization"""
    
    print(f"\n{'='*50}")
    print(f"{title} COORDINATE QUALITY")
    print(f"{'='*50}")
    
    coords = sequences.reshape(-1, 2)
    
    # Basic statistics
    print(f"Total coordinates: {len(coords)}")
    print(f"X range: [{np.min(coords[:, 0]):.6f}, {np.max(coords[:, 0]):.6f}]")
    print(f"Y range: [{np.min(coords[:, 1]):.6f}, {np.max(coords[:, 1]):.6f}]")
    
    # Zero coordinates (missing detections)
    zero_coords = np.sum(np.all(coords == 0, axis=1))
    zero_percentage = zero_coords / len(coords) * 100
    print(f"Zero coordinates: {zero_coords}/{len(coords)} ({zero_percentage:.2f}%)")
    
    # Normalization check
    properly_normalized = np.max(np.abs(coords)) <= 1.0
    print(f"Properly normalized: {properly_normalized}")
    
    # Outlier detection
    abs_coords = np.abs(coords)
    outliers = np.sum(abs_coords > 1.0)
    outlier_percentage = outliers / len(coords.flatten()) * 100
    print(f"Outliers (>1.0): {outliers}/{len(coords.flatten())} ({outlier_percentage:.2f}%)")
    
    return {
        'zero_percentage': zero_percentage,
        'properly_normalized': properly_normalized,
        'outlier_percentage': outlier_percentage,
        'max_abs_value': np.max(np.abs(coords))
    }

def detailed_comparison():
    """Compare original vs optimized datasets"""
    
    print("="*80)
    print("POSE DATA OPTIMIZATION COMPARISON")
    print("="*80)
    
    # File paths
    orig_seq_path = 'pose_features/train_sequences_updated.npy'
    orig_files_path = 'pose_features/train_filenames_updated.json'
    opt_seq_path = 'pose_features_optimized/train_sequences_optimized.npy'
    opt_files_path = 'pose_features_optimized/train_filenames_optimized.json'
    
    # Check file existence
    missing_files = []
    for path in [orig_seq_path, opt_seq_path]:
        if not Path(path).exists():
            missing_files.append(path)
    
    if missing_files:
        print("❌ Missing files:")
        for path in missing_files:
            print(f"  - {path}")
        
        if not Path(opt_seq_path).exists():
            print("\n💡 Run the optimized preprocessing first:")
            print("   ./run_optimized_batch.sh")
        return
    
    # Load datasets
    print("Loading datasets...")
    
    orig_sequences = np.load(orig_seq_path)
    opt_sequences = np.load(opt_seq_path)
    
    print(f"Original: {orig_sequences.shape}")
    print(f"Optimized: {opt_sequences.shape}")
    
    # Coordinate quality comparison
    orig_coord_stats = check_coordinate_quality(orig_sequences, "ORIGINAL")
    opt_coord_stats = check_coordinate_quality(opt_sequences, "OPTIMIZED")
    
    # Temporal artifact comparison
    orig_temporal_stats = analyze_temporal_artifacts(orig_sequences, "ORIGINAL")
    opt_temporal_stats = analyze_temporal_artifacts(opt_sequences, "OPTIMIZED")
    
    # Improvement analysis
    print(f"\n{'='*50}")
    print("IMPROVEMENT ANALYSIS")
    print(f"{'='*50}")
    
    improvements = []
    
    # 1. Normalization improvement
    if orig_coord_stats['max_abs_value'] > 10 and opt_coord_stats['properly_normalized']:
        print("✅ NORMALIZATION: Fixed coordinate scaling")
        improvements.append("normalization")
    elif opt_coord_stats['properly_normalized']:
        print("✅ NORMALIZATION: Maintained proper scaling")
    else:
        print("❌ NORMALIZATION: Still has issues")
    
    # 2. Artifact reduction
    artifact_reduction = orig_temporal_stats['artifact_percentage'] - opt_temporal_stats['artifact_percentage']
    if artifact_reduction > 10:
        print(f"✅ ARTIFACTS: Reduced by {artifact_reduction:.1f}%")
        improvements.append("artifacts")
    elif artifact_reduction > 5:
        print(f"🔄 ARTIFACTS: Slightly reduced by {artifact_reduction:.1f}%")
    else:
        print(f"❌ ARTIFACTS: No significant reduction ({artifact_reduction:.1f}%)")
    
    # 3. Movement smoothing
    movement_reduction = (orig_temporal_stats['max_movement'] - opt_temporal_stats['max_movement']) / orig_temporal_stats['max_movement'] * 100
    if movement_reduction > 20:
        print(f"✅ SMOOTHING: Reduced extreme movements by {movement_reduction:.1f}%")
        improvements.append("smoothing")
    elif movement_reduction > 5:
        print(f"🔄 SMOOTHING: Slightly improved by {movement_reduction:.1f}%")
    else:
        print(f"❌ SMOOTHING: No significant improvement")
    
    # 4. Data quality
    outlier_reduction = orig_coord_stats['outlier_percentage'] - opt_coord_stats['outlier_percentage']
    if outlier_reduction > 1:
        print(f"✅ QUALITY: Reduced outliers by {outlier_reduction:.2f}%")
        improvements.append("quality")
    
    # Overall recommendation
    print(f"\n{'='*50}")
    print("RECOMMENDATION")
    print(f"{'='*50}")
    
    print(f"Improvements detected: {len(improvements)}/4")
    print(f"  - Normalization: {'✅' if 'normalization' in improvements else '❌'}")
    print(f"  - Artifact reduction: {'✅' if 'artifacts' in improvements else '❌'}")
    print(f"  - Movement smoothing: {'✅' if 'smoothing' in improvements else '❌'}")
    print(f"  - Data quality: {'✅' if 'quality' in improvements else '❌'}")
    
    if len(improvements) >= 2:
        print("\n🎯 RECOMMENDED: Use optimized data for LSTM training")
        print("   Significant improvements in pose data quality detected")
    elif len(improvements) >= 1:
        print("\n🔄 CONSIDER: Optimized data shows some improvements")
        print("   May be worth testing both versions")
    else:
        print("\n❓ UNCLEAR: Limited improvements detected")
        print("   May need to adjust optimization parameters")
    
    # Detailed metrics for reference
    print(f"\nDetailed metrics:")
    print(f"  Original max movement: {orig_temporal_stats['max_movement']:.6f}")
    print(f"  Optimized max movement: {opt_temporal_stats['max_movement']:.6f}")
    print(f"  Original artifacts: {orig_temporal_stats['artifact_percentage']:.1f}%")
    print(f"  Optimized artifacts: {opt_temporal_stats['artifact_percentage']:.1f}%")
    
    return {
        'original': {
            'coord_stats': orig_coord_stats,
            'temporal_stats': orig_temporal_stats
        },
        'optimized': {
            'coord_stats': opt_coord_stats,
            'temporal_stats': opt_temporal_stats
        },
        'improvements': improvements
    }

if __name__ == '__main__':
    detailed_comparison()