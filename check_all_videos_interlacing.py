#!/usr/bin/env python3
"""
Check all 2099 videos in train_set for interlacing patterns
Fast batch analysis with progress tracking
"""

import os
import cv2
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Global stats for thread safety
stats_lock = threading.Lock()
global_stats = {
    'progressive': 0,
    'interlaced': 0,
    'unclear': 0,
    'errors': 0
}

def quick_interlacing_check(video_path):
    """
    Fast interlacing check using minimal frames
    Returns: (filename, is_interlaced, score, details)
    """
    
    try:
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            return (video_path.name, None, -1, {'error': 'Cannot open video'})
        
        # Get basic properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Sample only first 10 frames for speed
        sample_size = min(10, frame_count)
        frame_diffs = []
        line_diffs = []
        
        prev_frame = None
        
        for i in range(sample_size):
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Quick line difference check (interlacing signature)
            if i < 3:  # Only check first 3 frames for speed
                # Sample every 10th line to speed up
                sampled_lines = gray[::10, ::4]  # Downsample for speed
                if sampled_lines.shape[0] > 2:
                    line_diff = np.mean([
                        np.mean(np.abs(sampled_lines[j].astype(int) - sampled_lines[j+1].astype(int)))
                        for j in range(0, len(sampled_lines)-1, 2)
                        if j+1 < len(sampled_lines)
                    ])
                    line_diffs.append(line_diff)
            
            # Frame-to-frame differences
            if prev_frame is not None:
                # Downsample for speed
                small_prev = cv2.resize(prev_frame, (180, 120))
                small_curr = cv2.resize(gray, (180, 120))
                diff = cv2.absdiff(small_prev, small_curr)
                mean_diff = np.mean(diff)
                frame_diffs.append(mean_diff)
            
            prev_frame = gray
        
        cap.release()
        
        if len(frame_diffs) < 3:
            return (video_path.name, None, -1, {'error': 'Not enough frames'})
        
        # Quick analysis
        details = {
            'width': width,
            'height': height,
            'fps': fps,
            'frame_count': frame_count,
            'sample_size': len(frame_diffs)
        }
        
        # Check for alternating frame pattern
        if len(frame_diffs) >= 6:
            even_diffs = np.array(frame_diffs[::2])
            odd_diffs = np.array(frame_diffs[1::2][:len(even_diffs)])
            
            even_mean = np.mean(even_diffs)
            odd_mean = np.mean(odd_diffs)
            
            frame_ratio = even_mean / odd_mean if odd_mean > 0 else 1.0
            details['frame_ratio'] = frame_ratio
            
            # Interlacing score (simplified)
            score = 0
            
            # Strong alternating pattern
            if abs(frame_ratio - 1.0) > 0.4:
                score += 2
                details['strong_alternating'] = True
            elif abs(frame_ratio - 1.0) > 0.2:
                score += 1
                details['weak_alternating'] = True
            
            # High line differences (if we have data)
            if line_diffs:
                avg_line_diff = np.mean(line_diffs)
                details['avg_line_diff'] = avg_line_diff
                if avg_line_diff > 20:
                    score += 1
                    details['high_line_diff'] = True
            
            # Classification
            if score >= 2:
                classification = 'interlaced'
            elif score == 1:
                classification = 'unclear'
            else:
                classification = 'progressive'
            
            details['interlacing_score'] = score
            
            return (video_path.name, classification, score, details)
        
        else:
            # Too few frames for reliable analysis
            return (video_path.name, 'unclear', 0, details)
            
    except Exception as e:
        return (video_path.name, None, -1, {'error': str(e)})

def update_global_stats(classification):
    """Thread-safe stats update"""
    with stats_lock:
        if classification == 'progressive':
            global_stats['progressive'] += 1
        elif classification == 'interlaced':
            global_stats['interlaced'] += 1
        elif classification == 'unclear':
            global_stats['unclear'] += 1
        else:
            global_stats['errors'] += 1

def process_batch(video_paths, batch_id, total_batches):
    """Process a batch of videos"""
    results = []
    
    for video_path in video_paths:
        filename, classification, score, details = quick_interlacing_check(video_path)
        
        results.append({
            'filename': filename,
            'classification': classification,
            'score': score,
            'details': details
        })
        
        # Update global stats
        update_global_stats(classification)
    
    return results

def main():
    print("="*80)
    print("BATCH INTERLACING ANALYSIS - ALL 2099 VIDEOS")
    print("="*80)
    
    # Find all video files
    train_dir = Path('train_set')
    if not train_dir.exists():
        print("❌ train_set directory not found")
        return
    
    # Get all video files
    video_extensions = ['.avi', '.mp4', '.mov', '.mkv']
    video_files = []
    for ext in video_extensions:
        video_files.extend(list(train_dir.glob(f'*{ext}')))
    
    print(f"Found {len(video_files)} video files")
    
    if len(video_files) == 0:
        print("❌ No videos found")
        return
    
    # Setup parallel processing
    num_threads = min(8, os.cpu_count())  # Use up to 8 threads
    batch_size = max(1, len(video_files) // (num_threads * 4))  # Small batches for better progress
    
    print(f"Using {num_threads} threads with batch size {batch_size}")
    print("Starting analysis...")
    
    start_time = time.time()
    all_results = []
    
    # Create batches
    batches = []
    for i in range(0, len(video_files), batch_size):
        batch = video_files[i:i + batch_size]
        batches.append(batch)
    
    # Process batches in parallel
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit all batches
        future_to_batch = {
            executor.submit(process_batch, batch, i, len(batches)): i 
            for i, batch in enumerate(batches)
        }
        
        # Progress tracking
        with tqdm(total=len(video_files), desc="Processing videos") as pbar:
            completed_videos = 0
            
            for future in as_completed(future_to_batch):
                batch_id = future_to_batch[future]
                
                try:
                    batch_results = future.result()
                    all_results.extend(batch_results)
                    
                    # Update progress
                    pbar.update(len(batch_results))
                    completed_videos += len(batch_results)
                    
                    # Show current stats
                    with stats_lock:
                        pbar.set_postfix({
                            'Prog': global_stats['progressive'],
                            'Int': global_stats['interlaced'], 
                            'Unc': global_stats['unclear'],
                            'Err': global_stats['errors']
                        })
                    
                except Exception as e:
                    print(f"\n❌ Batch {batch_id} failed: {e}")
    
    processing_time = time.time() - start_time
    
    # Final analysis
    print(f"\n{'='*80}")
    print("FINAL RESULTS")
    print(f"{'='*80}")
    
    total_processed = len(all_results)
    print(f"Videos processed: {total_processed}/{len(video_files)}")
    print(f"Processing time: {processing_time:.2f} seconds")
    print(f"Speed: {total_processed/processing_time:.1f} videos/second")
    
    print(f"\nClassification results:")
    print(f"  Progressive: {global_stats['progressive']} ({global_stats['progressive']/total_processed*100:.1f}%)")
    print(f"  Interlaced: {global_stats['interlaced']} ({global_stats['interlaced']/total_processed*100:.1f}%)")
    print(f"  Unclear: {global_stats['unclear']} ({global_stats['unclear']/total_processed*100:.1f}%)")
    print(f"  Errors: {global_stats['errors']} ({global_stats['errors']/total_processed*100:.1f}%)")
    
    # Detailed analysis of interlaced videos
    interlaced_videos = [r for r in all_results if r['classification'] == 'interlaced']
    if len(interlaced_videos) > 0:
        print(f"\n🚨 INTERLACED VIDEOS FOUND ({len(interlaced_videos)}):")
        
        # Show top 10 most suspicious
        sorted_interlaced = sorted(interlaced_videos, key=lambda x: x['score'], reverse=True)
        for video in sorted_interlaced[:10]:
            details = video['details']
            print(f"  {video['filename']} (score: {video['score']})")
            if 'frame_ratio' in details:
                print(f"    Frame ratio: {details['frame_ratio']:.3f}")
    
    # Check for patterns in properties
    progressive_videos = [r for r in all_results if r['classification'] == 'progressive']
    if len(progressive_videos) > 0:
        print(f"\n✅ PROGRESSIVE VIDEOS CONFIRMED ({len(progressive_videos)}):")
        
        # Check common properties
        sample_progressive = progressive_videos[:5]
        print("  Sample properties:")
        for video in sample_progressive:
            details = video['details']
            print(f"    {video['filename']}: {details.get('width', '?')}x{details.get('height', '?')} @ {details.get('fps', '?')} FPS")
    
    # Overall conclusion
    print(f"\n{'='*80}")
    print("CONCLUSION")
    print(f"{'='*80}")
    
    progressive_ratio = global_stats['progressive'] / total_processed
    interlaced_ratio = global_stats['interlaced'] / total_processed
    
    if progressive_ratio > 0.8:
        print("🎯 DATASET IS PREDOMINANTLY PROGRESSIVE")
        print("   → Interlacing is NOT the main cause of artifacts")
        print("   → Focus on pose detection and preprocessing improvements")
    elif interlaced_ratio > 0.5:
        print("⚠️  DATASET HAS SIGNIFICANT INTERLACING")
        print("   → Use deinterlacing preprocessing")
    else:
        print("❓ MIXED DATASET")
        print("   → May need case-by-case handling")
    
    # Save results
    output_file = 'all_videos_interlacing_analysis.json'
    
    summary = {
        'total_videos': len(video_files),
        'processed': total_processed,
        'processing_time': processing_time,
        'classification_counts': global_stats.copy(),
        'classification_percentages': {
            'progressive': progressive_ratio * 100,
            'interlaced': interlaced_ratio * 100,
            'unclear': global_stats['unclear'] / total_processed * 100,
            'errors': global_stats['errors'] / total_processed * 100
        },
        'detailed_results': all_results
    }
    
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📝 Detailed results saved to: {output_file}")

if __name__ == '__main__':
    main()