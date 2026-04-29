#!/usr/bin/env python3
"""
Generate Captions for Pilot Dataset
Creates all 4 caption variants: full, structural, affective, baseline
"""

import pandas as pd
import json
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from generate_captions_v2 import generate_caption, generate_redacted_caption
from clean_tags import clean_tag_string

def load_metadata_tags(jsonl_path):
    """Load tags from JSONL metadata file"""
    print(f"Loading metadata from {jsonl_path}...")
    
    tags_dict = {}
    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    track_id = data['id']
                    
                    # Extract tags from metadata
                    metadata = data.get('metadata', {})
                    tags = metadata.get('tags', '')
                    
                    tags_dict[track_id] = tags
                except json.JSONDecodeError:
                    continue
                except KeyError:
                    continue
    
    print(f"Loaded tags for {len(tags_dict)} tracks")
    return tags_dict

def generate_all_captions(features_csv, metadata_jsonl, output_dir):
    """Generate all 4 caption variants for pilot dataset"""
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load features
    print(f"\nLoading features from {features_csv}...")
    df = pd.read_csv(features_csv)
    print(f"Loaded {len(df)} tracks")
    
    # Load tags
    tags_dict = load_metadata_tags(metadata_jsonl)
    
    # Add tags to dataframe
    df['tags'] = df['track_id'].map(tags_dict)
    
    # Check how many tracks have tags
    tracks_with_tags = df['tags'].notna().sum()
    print(f"\nTracks with tags: {tracks_with_tags}/{len(df)} ({tracks_with_tags/len(df)*100:.1f}%)")
    
    # Generate captions for each type
    caption_types = ['full', 'structural', 'affective']
    
    for caption_type in caption_types:
        print(f"\n{'='*70}")
        print(f"Generating {caption_type.upper()} captions...")
        print('='*70)
        
        captions = []
        
        for idx, row in df.iterrows():
            track_features = row.to_dict()
            metadata_tags = row.get('tags', None)
            
            # Generate caption
            caption = generate_caption(track_features, metadata_tags, caption_type)
            captions.append(caption)
            
            # Show first 3 examples
            if idx < 3:
                print(f"\nTrack {idx+1}: {row['track_id']}")
                print(f"  Key: {row['key']}, Tempo: {row['tempo']:.1f} BPM")
                if metadata_tags:
                    clean_tags = clean_tag_string(metadata_tags)
                    print(f"  Tags: {metadata_tags[:80]}...")
                    print(f"  Cleaned: {clean_tags}")
                print(f"  Caption: {caption}")
        
        # Save to CSV
        output_df = df[['track_id']].copy()
        output_df['caption'] = captions
        
        output_file = output_dir / f"captions_{caption_type}.csv"
        output_df.to_csv(output_file, index=False)
        print(f"\n✓ Saved {len(captions)} captions to {output_file}")
    
    # Generate baseline (redacted) captions
    print(f"\n{'='*70}")
    print("Generating BASELINE (redacted) captions...")
    print('='*70)
    
    baseline_captions = []
    
    for idx, row in df.iterrows():
        metadata_tags = row.get('tags', None)
        
        # For baseline, we use the tags as the "original prompt"
        # and redact structural info
        if metadata_tags:
            caption = generate_redacted_caption(metadata_tags, metadata_tags)
        else:
            caption = "A musical composition."
        
        baseline_captions.append(caption)
        
        # Show first 3 examples
        if idx < 3:
            print(f"\nTrack {idx+1}: {row['track_id']}")
            if metadata_tags:
                print(f"  Original tags: {metadata_tags[:80]}...")
            print(f"  Baseline: {caption}")
    
    # Save baseline
    output_df = df[['track_id']].copy()
    output_df['caption'] = baseline_captions
    
    output_file = output_dir / "captions_baseline.csv"
    output_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved {len(baseline_captions)} captions to {output_file}")
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print('='*70)
    print(f"Total tracks processed: {len(df)}")
    print(f"Tracks with genre tags: {tracks_with_tags}")
    print(f"\nGenerated caption files:")
    print(f"  - captions_full.csv (structural + affective + genre)")
    print(f"  - captions_structural.csv (key + tempo + genre)")
    print(f"  - captions_affective.csv (emotions + genre)")
    print(f"  - captions_baseline.csv (genre only, no theory)")
    print(f"\nAll files saved to: {output_dir}")
    print('='*70)

if __name__ == '__main__':
    # Paths
    FEATURES_CSV = '/scratch/user/esomtoochiobi/thesis_work/outputs/features_50k/features_50k.csv'
    METADATA_JSONL = '/scratch/user/esomtoochiobi/thesis_work/data/sample_50k.jsonl'
    OUTPUT_DIR = '/scratch/user/esomtoochiobi/thesis_work/outputs/captions_50k'
    
    # Generate captions
    generate_all_captions(FEATURES_CSV, METADATA_JSONL, OUTPUT_DIR)
    
    print("\n✓ Caption generation complete!")