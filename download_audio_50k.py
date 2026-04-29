#!/usr/bin/env python3
"""
Parallel Audio Download Script for Suno-660k
Downloads MP3 files from suno.jsonl in parallel chunks
"""

import json
import argparse
import requests
from pathlib import Path
from tqdm import tqdm
import time

def load_metadata(jsonl_path, job_id, total_jobs):
    """Load and split metadata for this job"""
    print(f"Loading metadata from {jsonl_path}...")
    
    all_tracks = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    all_tracks.append(data)
                except:
                    continue
    
    total_tracks = len(all_tracks)
    print(f"Total tracks in dataset: {total_tracks}")
    
    # Calculate chunk for this job
    chunk_size = (total_tracks + total_jobs - 1) // total_jobs
    start_idx = job_id * chunk_size
    end_idx = min(start_idx + chunk_size, total_tracks)
    
    job_tracks = all_tracks[start_idx:end_idx]
    print(f"Job {job_id}/{total_jobs}: Processing tracks {start_idx} to {end_idx-1} ({len(job_tracks)} tracks)")
    
    return job_tracks

def download_audio(track, output_dir, max_retries=3, timeout=30):
    """Download audio file for a single track with retry logic"""
    track_id = track['id']
    audio_url = track.get('audio_url')
    
    if not audio_url:
        return False, "No audio URL"
    
    output_path = output_dir / f"{track_id}.mp3"
    
    # Skip if already exists
    if output_path.exists():
        file_size = output_path.stat().st_size
        if file_size > 10000:  # At least 10KB
            return True, "Already exists"
        else:
            # File exists but too small - redownload
            output_path.unlink()
    
    # Headers for Suno CDN
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Referer': 'https://suno.ai/',
        'Accept': 'audio/webm,audio/ogg,audio/wav,audio/*;q=0.9',
    }
    
    # Download with retries
    for attempt in range(max_retries):
        try:
            response = requests.get(audio_url, headers=headers, timeout=timeout, stream=True)
            response.raise_for_status()
            
            # Write to file
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Verify file size
            file_size = output_path.stat().st_size
            if file_size < 10000:
                output_path.unlink()
                return False, f"File too small ({file_size} bytes)"
            
            return True, "Success"
            
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            return False, "Timeout"
            
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            return False, f"Request error: {str(e)}"
            
        except Exception as e:
            return False, f"Unexpected error: {str(e)}"
    
    return False, "Max retries exceeded"

def main():
    parser = argparse.ArgumentParser(description='Download Suno audio files in parallel')
    parser.add_argument('--job-id', type=int, required=True, help='Job array index (0-based)')
    parser.add_argument('--total-jobs', type=int, required=True, help='Total number of parallel jobs')
    parser.add_argument('--jsonl-path', type=str, required=True,
                       help='Path to suno.jsonl metadata file')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for audio files')
    parser.add_argument('--workers', type=int, default=1,
                       help='Number of parallel workers (ignored, kept for compatibility)')
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    jsonl_path = Path(args.jsonl_path)
    if not jsonl_path.exists():
        print(f"ERROR: Metadata file not found: {jsonl_path}")
        return
    
    # Load tracks for this job
    tracks = load_metadata(jsonl_path, args.job_id, args.total_jobs)
    
    if len(tracks) == 0:
        print("No tracks to process for this job")
        return
    
    # Download tracks
    print(f"\nStarting download of {len(tracks)} tracks...")
    print(f"Output directory: {output_dir}")
    print("-" * 60)
    
    success_count = 0
    skip_count = 0
    fail_count = 0
    failed_tracks = []
    
    for track in tqdm(tracks, desc=f"Job {args.job_id}"):
        success, message = download_audio(track, output_dir)
        
        if success:
            if "Already exists" in message:
                skip_count += 1
            else:
                success_count += 1
        else:
            fail_count += 1
            failed_tracks.append((track['id'], message))
    
    # Summary
    print("\n" + "=" * 60)
    print(f"DOWNLOAD SUMMARY - Job {args.job_id}")
    print("=" * 60)
    print(f"Total tracks:     {len(tracks)}")
    print(f"Downloaded:       {success_count}")
    print(f"Already existed:  {skip_count}")
    print(f"Failed:           {fail_count}")
    print(f"Success rate:     {((success_count + skip_count) / len(tracks) * 100):.1f}%")
    
    # Save failed tracks log
    if failed_tracks:
        log_file = output_dir / f"failed_downloads_job{args.job_id}.txt"
        with open(log_file, 'w') as f:
            for track_id, reason in failed_tracks:
                f.write(f"{track_id}\t{reason}\n")
        print(f"\nFailed tracks logged to: {log_file}")
    
    print("=" * 60)
    print(f"\nJob {args.job_id} complete!")

if __name__ == '__main__':
    main()