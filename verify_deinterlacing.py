#!/usr/bin/env python3
"""
Verify that deinterlacing was applied by comparing frames with/without deinterlacing
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

def deinterlace_frame_bob(frame):
    """Bob deinterlacing - same as in preprocessing_optimized_batch.py"""
    height, width = frame.shape[:2]
    deinterlaced = np.zeros_like(frame)

    # Keep odd lines, interpolate even lines
    deinterlaced[1::2] = frame[1::2]  # Odd lines
    deinterlaced[0::2] = frame[1::2]  # Interpolate even from odd

    # Handle boundary
    if height > 1:
        deinterlaced[0] = frame[1]

    return deinterlaced

def detect_interlacing_artifacts(frame):
    """
    Detect interlacing artifacts by measuring difference between even/odd lines
    Interlaced video has high difference, deinterlaced has low difference
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Extract even and odd lines
    even_lines = gray[0::2, :]
    odd_lines = gray[1::2, :]

    # Make same size
    min_height = min(even_lines.shape[0], odd_lines.shape[0])
    even_lines = even_lines[:min_height, :]
    odd_lines = odd_lines[:min_height, :]

    # Calculate difference
    diff = np.abs(even_lines.astype(float) - odd_lines.astype(float))

    # High difference indicates interlacing artifacts
    mean_diff = np.mean(diff)
    max_diff = np.max(diff)

    return mean_diff, max_diff, diff

def test_video_deinterlacing(video_path, frame_number=30):
    """
    Test a video by comparing original vs deinterlaced frames
    """
    print(f"\nTesting: {video_path}")
    print("="*80)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return None

    # Seek to specific frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, original_frame = cap.read()
    cap.release()

    if not ret:
        print(f"❌ Cannot read frame {frame_number}")
        return None

    # Apply deinterlacing
    deinterlaced_frame = deinterlace_frame_bob(original_frame)

    # Analyze interlacing artifacts
    orig_mean, orig_max, orig_diff = detect_interlacing_artifacts(original_frame)
    deint_mean, deint_max, deint_diff = detect_interlacing_artifacts(deinterlaced_frame)

    print(f"\nInterlacing Artifact Analysis (frame {frame_number}):")
    print(f"  Original frame:")
    print(f"    Mean even/odd line difference: {orig_mean:.2f}")
    print(f"    Max even/odd line difference:  {orig_max:.2f}")
    print(f"  Deinterlaced frame:")
    print(f"    Mean even/odd line difference: {deint_mean:.2f}")
    print(f"    Max even/odd line difference:  {deint_max:.2f}")

    # Calculate reduction
    reduction = (orig_mean - deint_mean) / orig_mean * 100 if orig_mean > 0 else 0
    print(f"  Artifact reduction: {reduction:.1f}%")

    # Verdict
    if orig_mean > deint_mean * 1.5:
        print(f"  ✅ DEINTERLACING EFFECTIVE - Artifacts reduced significantly")
    elif orig_mean > 10:
        print(f"  ⚠️  Original video has interlacing artifacts")
    else:
        print(f"  ℹ️  Original video may not be interlaced")

    return {
        'video': str(video_path),
        'frame': frame_number,
        'original': {
            'mean_diff': float(orig_mean),
            'max_diff': float(orig_max)
        },
        'deinterlaced': {
            'mean_diff': float(deint_mean),
            'max_diff': float(deint_max)
        },
        'reduction_percent': float(reduction),
        'original_frame': original_frame,
        'deinterlaced_frame': deinterlaced_frame,
        'original_diff': orig_diff,
        'deinterlaced_diff': deint_diff
    }

def save_comparison_image(result, output_path):
    """Save side-by-side comparison image"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Original frame
    axes[0, 0].imshow(cv2.cvtColor(result['original_frame'], cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Frame')
    axes[0, 0].axis('off')

    # Deinterlaced frame
    axes[0, 1].imshow(cv2.cvtColor(result['deinterlaced_frame'], cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('Deinterlaced Frame')
    axes[0, 1].axis('off')

    # Difference
    frame_diff = cv2.absdiff(result['original_frame'], result['deinterlaced_frame'])
    axes[0, 2].imshow(cv2.cvtColor(frame_diff, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title('Absolute Difference')
    axes[0, 2].axis('off')

    # Original even/odd line difference
    axes[1, 0].imshow(result['original_diff'], cmap='hot', vmin=0, vmax=50)
    axes[1, 0].set_title(f"Original Line Diff (mean={result['original']['mean_diff']:.2f})")
    axes[1, 0].axis('off')

    # Deinterlaced even/odd line difference
    axes[1, 1].imshow(result['deinterlaced_diff'], cmap='hot', vmin=0, vmax=50)
    axes[1, 1].set_title(f"Deinterlaced Line Diff (mean={result['deinterlaced']['mean_diff']:.2f})")
    axes[1, 1].axis('off')

    # Stats
    stats_text = f"""
Video: {Path(result['video']).name}
Frame: {result['frame']}

Original:
  Mean diff: {result['original']['mean_diff']:.2f}
  Max diff: {result['original']['max_diff']:.2f}

Deinterlaced:
  Mean diff: {result['deinterlaced']['mean_diff']:.2f}
  Max diff: {result['deinterlaced']['max_diff']:.2f}

Reduction: {result['reduction_percent']:.1f}%
"""
    axes[1, 2].text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                    verticalalignment='center')
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  💾 Saved comparison: {output_path}")
    plt.close()

def main():
    print("="*80)
    print("DEINTERLACING VERIFICATION")
    print("="*80)

    # Find some videos to test
    data_dir = Path('train_set')
    video_files = list(data_dir.glob('*.avi'))[:5]  # Test first 5 videos

    if len(video_files) == 0:
        print("❌ No videos found in train_set/")
        return

    print(f"\nTesting {len(video_files)} videos...")

    results = []
    output_dir = Path('deinterlacing_verification')
    output_dir.mkdir(exist_ok=True)

    for i, video_path in enumerate(video_files):
        # Test multiple frames to catch motion
        for frame_num in [30, 60, 90]:
            result = test_video_deinterlacing(video_path, frame_num)

            if result:
                results.append(result)

                # Save comparison image
                output_path = output_dir / f"{video_path.stem}_frame{frame_num}_comparison.png"
                save_comparison_image(result, output_path)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    if len(results) > 0:
        avg_orig_diff = np.mean([r['original']['mean_diff'] for r in results])
        avg_deint_diff = np.mean([r['deinterlaced']['mean_diff'] for r in results])
        avg_reduction = np.mean([r['reduction_percent'] for r in results])

        print(f"\nAverage across {len(results)} test frames:")
        print(f"  Original mean line difference:     {avg_orig_diff:.2f}")
        print(f"  Deinterlaced mean line difference: {avg_deint_diff:.2f}")
        print(f"  Average artifact reduction:        {avg_reduction:.1f}%")

        if avg_reduction > 30:
            print(f"\n✅ DEINTERLACING IS WORKING EFFECTIVELY!")
            print(f"   Interlacing artifacts reduced by {avg_reduction:.1f}%")
        elif avg_orig_diff > 10:
            print(f"\n⚠️  Original videos have interlacing but reduction is moderate")
        else:
            print(f"\nℹ️  Videos may not be heavily interlaced")

        # Save detailed results
        summary_file = output_dir / 'verification_results.json'
        with open(summary_file, 'w') as f:
            # Remove image data before saving
            results_to_save = []
            for r in results:
                r_copy = {k: v for k, v in r.items()
                         if k not in ['original_frame', 'deinterlaced_frame',
                                     'original_diff', 'deinterlaced_diff']}
                results_to_save.append(r_copy)

            json.dump({
                'summary': {
                    'avg_original_diff': float(avg_orig_diff),
                    'avg_deinterlaced_diff': float(avg_deint_diff),
                    'avg_reduction_percent': float(avg_reduction)
                },
                'results': results_to_save
            }, f, indent=2)

        print(f"\n📊 Detailed results saved: {summary_file}")
        print(f"🖼️  Comparison images saved: {output_dir}/")
    else:
        print("\n❌ No results to analyze")

if __name__ == '__main__':
    main()
