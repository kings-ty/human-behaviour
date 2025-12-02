#!/usr/bin/env python3
"""
Detailed video analysis for DV format videos
"""

import cv2
import numpy as np
import os
from pathlib import Path

def analyze_dv_video_directly(video_path):
    """Analyze DV video directly using OpenCV"""
    
    print(f"\nAnalyzing: {os.path.basename(video_path)}")
    print("-" * 50)
    
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print("❌ Cannot open video")
        return None
    
    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Properties: {width}x{height} @ {fps:.2f} FPS, {frame_count} frames")
    
    # Sample frames for analysis
    sample_frames = min(30, frame_count)
    frame_diffs = []
    field_analysis = []
    
    prev_frame = None
    
    for i in range(sample_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Check for field patterns (interlacing signature)
        # In interlaced video, even and odd lines come from different time points
        even_lines = gray[::2, :]  # Even rows
        odd_lines = gray[1::2, :]   # Odd rows
        
        # Calculate variance in even vs odd lines
        even_var = np.var(even_lines)
        odd_var = np.var(odd_lines)
        
        # Calculate difference between adjacent lines (should be higher in interlaced)
        line_diffs = []
        for y in range(0, gray.shape[0]-1, 2):
            if y+1 < gray.shape[0]:
                diff = np.mean(np.abs(gray[y, :].astype(int) - gray[y+1, :].astype(int)))
                line_diffs.append(diff)
        
        avg_line_diff = np.mean(line_diffs) if line_diffs else 0
        
        field_analysis.append({
            'frame': i,
            'even_var': even_var,
            'odd_var': odd_var,
            'var_ratio': even_var / odd_var if odd_var > 0 else 1.0,
            'avg_line_diff': avg_line_diff
        })
        
        # Frame-to-frame differences
        if prev_frame is not None:
            diff = cv2.absdiff(prev_frame, gray)
            mean_diff = np.mean(diff)
            frame_diffs.append(mean_diff)
        
        prev_frame = gray
    
    cap.release()
    
    # Analyze results
    if len(frame_diffs) < 5:
        print("❌ Not enough frames for analysis")
        return None
    
    # Check for alternating frame difference pattern (interlacing)
    even_diffs = np.array(frame_diffs[::2])
    odd_diffs = np.array(frame_diffs[1::2][:len(even_diffs)])
    
    even_mean = np.mean(even_diffs)
    odd_mean = np.mean(odd_diffs)
    frame_diff_ratio = even_mean / odd_mean if odd_mean > 0 else 1.0
    
    # Check field variance patterns
    var_ratios = [f['var_ratio'] for f in field_analysis]
    line_diffs = [f['avg_line_diff'] for f in field_analysis]
    
    avg_var_ratio = np.mean(var_ratios)
    avg_line_diff = np.mean(line_diffs)
    
    print(f"Frame difference analysis:")
    print(f"  Even frames avg diff: {even_mean:.2f}")
    print(f"  Odd frames avg diff: {odd_mean:.2f}")
    print(f"  Ratio: {frame_diff_ratio:.3f}")
    
    print(f"Field analysis:")
    print(f"  Avg variance ratio (even/odd lines): {avg_var_ratio:.3f}")
    print(f"  Avg line-to-line difference: {avg_line_diff:.2f}")
    
    # Interlacing indicators:
    # 1. High line-to-line differences (fields from different times)
    # 2. Significant variance differences between even/odd lines
    # 3. Alternating frame difference pattern
    
    interlacing_score = 0
    indicators = []
    
    if avg_line_diff > 15:  # High line differences
        interlacing_score += 1
        indicators.append("High line-to-line differences")
    
    if abs(avg_var_ratio - 1.0) > 0.2:  # Uneven field variances
        interlacing_score += 1
        indicators.append("Uneven field variances")
    
    if abs(frame_diff_ratio - 1.0) > 0.3:  # Alternating pattern
        interlacing_score += 1
        indicators.append("Alternating frame differences")
    
    print(f"\nInterlacing analysis:")
    print(f"  Interlacing score: {interlacing_score}/3")
    
    if interlacing_score >= 2:
        print("  🚨 LIKELY INTERLACED")
    elif interlacing_score == 1:
        print("  ❓ POSSIBLY INTERLACED")  
    else:
        print("  ✅ LIKELY PROGRESSIVE")
    
    if indicators:
        print(f"  Indicators: {', '.join(indicators)}")
    
    return {
        'filename': os.path.basename(video_path),
        'properties': {
            'width': width,
            'height': height, 
            'fps': fps,
            'frame_count': frame_count
        },
        'analysis': {
            'frame_diff_ratio': frame_diff_ratio,
            'avg_var_ratio': avg_var_ratio,
            'avg_line_diff': avg_line_diff,
            'interlacing_score': interlacing_score,
            'indicators': indicators
        }
    }

def main():
    print("="*80)
    print("DETAILED DV VIDEO ANALYSIS")
    print("="*80)
    
    # Get problem videos from our previous list
    problem_videos = [
        'CID30_SID02_VID05.avi',
        'CID29_SID02_VID04.avi', 
        'CID29_SID10_VID02.avi',
        'CID30_SID02_VID01.avi',
        'CID30_SID02_VID03.avi'
    ]
    
    train_dir = Path('train_set')
    results = []
    
    interlaced_count = 0
    progressive_count = 0
    unclear_count = 0
    
    for video_name in problem_videos:
        video_path = train_dir / video_name
        
        if not video_path.exists():
            print(f"❌ {video_name} not found")
            continue
        
        result = analyze_dv_video_directly(video_path)
        
        if result:
            results.append(result)
            score = result['analysis']['interlacing_score']
            
            if score >= 2:
                interlaced_count += 1
            elif score == 1:
                unclear_count += 1
            else:
                progressive_count += 1
    
    # Summary
    print(f"\n{'='*80}")
    print("FINAL ANALYSIS")
    print(f"{'='*80}")
    
    total = len(results)
    print(f"Videos analyzed: {total}")
    print(f"Likely interlaced: {interlaced_count}")
    print(f"Likely progressive: {progressive_count}")
    print(f"Unclear: {unclear_count}")
    
    if interlaced_count > progressive_count:
        print(f"\n🚨 CONCLUSION: Videos appear to be INTERLACED")
        print("   → The artifacts are likely due to interlacing")
        print("   → Use the deinterlaced preprocessing script")
    elif progressive_count > interlaced_count:
        print(f"\n✅ CONCLUSION: Videos appear to be PROGRESSIVE") 
        print("   → Artifacts are NOT due to interlacing")
        print("   → Look for other causes:")
        print("     - Pose detection accuracy issues")
        print("     - Fast movement handling")
        print("     - Normalization problems")
        print("     - DV compression artifacts")
    else:
        print(f"\n❓ CONCLUSION: Results are mixed")
        print("   → Need manual inspection")
    
    # Additional notes for DV format
    print(f"\nℹ️  Notes about DV format:")
    print("   - DV (Digital Video) can be either progressive or interlaced")
    print("   - Your videos are 720x480 @ ~30fps (NTSC DV)")
    print("   - NTSC DV is typically interlaced (60i → 30fps)")
    print("   - But some DV cameras can record progressive")
    
    return results

if __name__ == '__main__':
    main()