#!/usr/bin/env python3
"""
Extract list of problem videos that showed interlacing artifacts
and check them with ffmpeg
"""

import os
import subprocess
import json
from pathlib import Path

def get_problem_videos_from_previous_analysis():
    """Extract the problem videos identified in the previous check"""
    
    # These are the videos that showed interlacing artifacts from our previous analysis
    problem_videos = [
        # High Even/Odd ratio problems (potential interlacing)
        'CID30_SID02_VID05.avi',  # ratio: 0.2139
        'CID30_SID02_VID04.avi',  # ratio: 0.2464  
        'CID29_SID02_VID04.avi',  # ratio: 0.2778
        'CID29_SID10_VID02.avi',  # ratio: 0.2708
        'CID30_SID02_VID01.avi',  # ratio: 5.7824
        'CID30_SID02_VID03.avi',  # ratio: 0.2221
        'CID30_SID08_VID07.avi',  # ratio: 0.2273
        'CID29_SID09_VID04.avi',  # ratio: 5.6570
        'CID29_SID05_VID05.avi',  # ratio: 0.2542
        'CID30_SID02_VID07.avi',  # ratio: 0.2635
        'CID29_SID05_VID03.avi',  # ratio: 3.9347
        'CID29_SID05_VID07.avi',  # ratio: 0.2455
        'CID30_SID05_VID01.avi',  # ratio: 4.8171
        'CID29_SID10_VID03.avi',  # ratio: 6.0038
        'CID29_SID03_VID02.avi',  # ratio: 0.1933
        'CID29_SID02_VID06.avi',  # ratio: 5.4815
        'CID30_SID03_VID07.avi',  # ratio: 0.1853
        'CID30_SID10_VID05.avi',  # ratio: 4.3518
        'CID30_SID02_VID02.avi',  # ratio: 0.2292
        'CID30_SID05_VID07.avi',  # ratio: 4.3267
        'CID30_SID10_VID02.avi',  # ratio: 5.2876
        'CID29_SID02_VID02.avi',  # ratio: 3.7229
        'CID29_SID02_VID05.avi',  # ratio: 6.5111
        'CID29_SID06_VID04.avi',  # ratio: 2.5452
        'CID29_SID05_VID06.avi',  # ratio: 0.2072
        'CID29_SID05_VID01.avi',  # ratio: 0.2770
        'CID29_SID02_VID01.avi',  # ratio: 0.1831
        
        # Also some from original dataset
        'CID13_SID06_VID02.avi',  # ratio: 1.8440
        'CID06_SID08_VID07.avi',  # ratio: 0.3250
    ]
    
    return problem_videos

def check_video_with_ffprobe(video_path):
    """Check video properties with ffprobe"""
    
    try:
        # Get basic video info
        cmd = [
            'ffprobe', 
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            str(video_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            import json
            data = json.loads(result.stdout)
            
            # Extract video stream info
            video_stream = None
            for stream in data.get('streams', []):
                if stream.get('codec_type') == 'video':
                    video_stream = stream
                    break
            
            if video_stream:
                return {
                    'filename': os.path.basename(video_path),
                    'codec': video_stream.get('codec_name'),
                    'width': video_stream.get('width'),
                    'height': video_stream.get('height'),
                    'fps': video_stream.get('r_frame_rate'),
                    'duration': video_stream.get('duration'),
                    'field_order': video_stream.get('field_order', 'unknown'),
                    'pix_fmt': video_stream.get('pix_fmt'),
                    'interlaced': video_stream.get('field_order') not in ['progressive', 'unknown']
                }
        else:
            return {'filename': os.path.basename(video_path), 'error': result.stderr}
            
    except subprocess.TimeoutExpired:
        return {'filename': os.path.basename(video_path), 'error': 'ffprobe timeout'}
    except Exception as e:
        return {'filename': os.path.basename(video_path), 'error': str(e)}

def check_interlacing_with_idet(video_path):
    """Check interlacing using ffmpeg idet filter"""
    
    try:
        # Use idet filter to detect interlacing
        cmd = [
            'ffmpeg',
            '-i', str(video_path),
            '-vf', 'idet',
            '-frames:v', '100',  # Check first 100 frames
            '-f', 'null',
            '-'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        # Parse idet output from stderr
        idet_info = {}
        for line in result.stderr.split('\n'):
            if 'Multi frame detection:' in line:
                # Extract detection results
                parts = line.split(':')[1].strip()
                # Example: "TFF:0 BFF:0 Progressive:98 Undetermined:2"
                for part in parts.split():
                    if ':' in part:
                        key, value = part.split(':')
                        idet_info[key.lower()] = int(value)
        
        return idet_info
        
    except subprocess.TimeoutExpired:
        return {'error': 'idet timeout'}
    except Exception as e:
        return {'error': str(e)}

def main():
    print("="*80)
    print("PROBLEM VIDEO ANALYSIS")
    print("="*80)
    
    # Check if ffmpeg/ffprobe are available
    try:
        subprocess.run(['ffprobe', '-version'], capture_output=True, check=True)
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        print("✅ ffmpeg/ffprobe found")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ ffmpeg/ffprobe not found. Install with:")
        print("   sudo apt install ffmpeg")
        return
    
    # Get list of problem videos
    problem_videos = get_problem_videos_from_previous_analysis()
    print(f"\nFound {len(problem_videos)} problem videos to check")
    
    # Check train_set directory
    train_dir = Path('train_set')
    if not train_dir.exists():
        print("❌ train_set directory not found")
        return
    
    print(f"Checking directory: {train_dir}")
    
    # Find existing problem videos
    existing_videos = []
    missing_videos = []
    
    for video_name in problem_videos:
        video_path = train_dir / video_name
        if video_path.exists():
            existing_videos.append(video_path)
        else:
            missing_videos.append(video_name)
    
    print(f"Found {len(existing_videos)} videos, {len(missing_videos)} missing")
    
    if missing_videos:
        print("\nMissing videos:")
        for video in missing_videos[:5]:  # Show first 5
            print(f"  - {video}")
        if len(missing_videos) > 5:
            print(f"  ... and {len(missing_videos) - 5} more")
    
    if len(existing_videos) == 0:
        print("❌ No problem videos found to analyze")
        return
    
    # Analyze videos
    print(f"\n{'='*80}")
    print("VIDEO ANALYSIS RESULTS")
    print(f"{'='*80}")
    
    results = []
    progressive_count = 0
    interlaced_count = 0
    unknown_count = 0
    
    for i, video_path in enumerate(existing_videos[:10], 1):  # Check first 10
        print(f"\n[{i}/{min(10, len(existing_videos))}] {video_path.name}")
        print("-" * 60)
        
        # Basic info with ffprobe
        info = check_video_with_ffprobe(video_path)
        
        if 'error' in info:
            print(f"❌ Error: {info['error']}")
            unknown_count += 1
            continue
        
        print(f"Codec: {info.get('codec', 'unknown')}")
        print(f"Resolution: {info.get('width', '?')}x{info.get('height', '?')}")
        print(f"FPS: {info.get('fps', 'unknown')}")
        print(f"Field order: {info.get('field_order', 'unknown')}")
        print(f"Pixel format: {info.get('pix_fmt', 'unknown')}")
        
        # Check with idet filter
        print("Running interlacing detection...")
        idet_results = check_interlacing_with_idet(video_path)
        
        if 'error' not in idet_results:
            total_frames = sum(idet_results.values())
            if total_frames > 0:
                progressive_pct = idet_results.get('progressive', 0) / total_frames * 100
                tff_pct = idet_results.get('tff', 0) / total_frames * 100
                bff_pct = idet_results.get('bff', 0) / total_frames * 100
                
                print(f"idet results: Progressive: {progressive_pct:.1f}%, TFF: {tff_pct:.1f}%, BFF: {bff_pct:.1f}%")
                
                if progressive_pct > 90:
                    print("✅ Video is PROGRESSIVE (not interlaced)")
                    progressive_count += 1
                elif tff_pct > 50 or bff_pct > 50:
                    print("⚠️  Video is INTERLACED")
                    interlaced_count += 1
                else:
                    print("❓ Unclear - mixed content")
                    unknown_count += 1
            else:
                print("❓ No idet data")
                unknown_count += 1
        else:
            print(f"❌ idet error: {idet_results['error']}")
            unknown_count += 1
        
        # Store results
        result = info.copy()
        result.update(idet_results)
        results.append(result)
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    total_checked = progressive_count + interlaced_count + unknown_count
    print(f"Videos analyzed: {total_checked}")
    print(f"Progressive (not interlaced): {progressive_count}")
    print(f"Interlaced: {interlaced_count}")
    print(f"Unknown/Error: {unknown_count}")
    
    if progressive_count > interlaced_count:
        print("\n🎯 CONCLUSION: Most videos are PROGRESSIVE")
        print("   → The artifacts are likely NOT due to interlacing")
        print("   → Look for other causes:")
        print("     - Pose detection failures")
        print("     - Fast movements")
        print("     - Occlusion/lighting issues")
        print("     - Preprocessing bugs")
    elif interlaced_count > progressive_count:
        print("\n⚠️  CONCLUSION: Videos are INTERLACED")
        print("   → Need deinterlacing preprocessing")
    else:
        print("\n❓ CONCLUSION: Mixed or unclear results")
        print("   → Need more analysis")
    
    # Save results
    with open('problem_video_analysis.json', 'w') as f:
        json.dump({
            'problem_videos': problem_videos,
            'analysis_results': results,
            'summary': {
                'total_checked': total_checked,
                'progressive': progressive_count,
                'interlaced': interlaced_count,
                'unknown': unknown_count
            }
        }, f, indent=2)
    
    print(f"\n📝 Detailed results saved to: problem_video_analysis.json")

if __name__ == '__main__':
    main()