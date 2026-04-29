#!/usr/bin/env python3
"""
CLAP Text Encoding for Music Captions
Encodes theory-enriched captions into 512-dimensional text embeddings
"""

import pandas as pd
import numpy as np
from pathlib import Path
import torch
from tqdm import tqdm
import argparse

def load_clap_model(device='cuda'):
    """
    Load CLAP text encoder
    Using LAION-CLAP: https://github.com/LAION-AI/CLAP
    """
    try:
        import laion_clap
        print("Loading LAION-CLAP model...")
        
        # Initialize model
        model = laion_clap.CLAP_Module(enable_fusion=False)
        model.load_ckpt()  # Downloads pretrained weights automatically
        model.eval()
        
        # Move to device
        if device == 'cuda' and torch.cuda.is_available():
            model = model.to(device)
            print(f"✓ Model loaded on GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("✓ Model loaded on CPU (this will be slow!)")
            device = 'cpu'
        
        return model, device
    
    except ImportError:
        print("ERROR: laion_clap not installed!")
        print("Install with: pip install laion-clap")
        raise

def encode_captions(caption_csv, output_dir, model, device, batch_size=32):
    """
    Encode all captions in a CSV file
    
    Args:
        caption_csv: Path to CSV with columns [track_id, caption]
        output_dir: Directory to save .npy embeddings
        model: CLAP model
        device: 'cuda' or 'cpu'
        batch_size: Number of captions to encode at once
    
    Returns:
        dict mapping track_id -> embedding path
    """
    # Load captions
    print(f"\nLoading captions from {caption_csv}...")
    df = pd.read_csv(caption_csv)
    print(f"Found {len(df)} captions")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Track embeddings
    embedding_paths = {}
    
    # Encode in batches
    print(f"\nEncoding captions (batch_size={batch_size})...")
    
    with torch.no_grad():
        for i in tqdm(range(0, len(df), batch_size), desc="Encoding"):
            batch = df.iloc[i:i+batch_size]
            
            # Get batch of captions
            captions = batch['caption'].tolist()
            track_ids = batch['track_id'].tolist()
            
            # Encode with CLAP
            try:
                # CLAP expects list of strings
                text_embeddings = model.get_text_embedding(captions, use_tensor=False)
                
                # text_embeddings shape: (batch_size, 512)
                # Save each embedding
                for j, (track_id, embedding) in enumerate(zip(track_ids, text_embeddings)):
                    output_path = output_dir / f"{track_id}.npy"
                    np.save(output_path, embedding)
                    embedding_paths[track_id] = str(output_path)
            
            except Exception as e:
                print(f"\nError encoding batch {i}-{i+batch_size}: {e}")
                # Try one at a time as fallback
                for track_id, caption in zip(track_ids, captions):
                    try:
                        embedding = model.get_text_embedding([caption], use_tensor=False)[0]
                        output_path = output_dir / f"{track_id}.npy"
                        np.save(output_path, embedding)
                        embedding_paths[track_id] = str(output_path)
                    except Exception as e2:
                        print(f"  Failed on track {track_id}: {e2}")
                        continue
    
    print(f"\n✓ Encoded {len(embedding_paths)} captions")
    print(f"✓ Saved to {output_dir}")
    
    return embedding_paths

def verify_embeddings(embedding_dir):
    """Verify embeddings are correct shape and format"""
    embedding_dir = Path(embedding_dir)
    embedding_files = list(embedding_dir.glob("*.npy"))
    
    print(f"\nVerifying {len(embedding_files)} embeddings...")
    
    shapes = []
    issues = []
    
    for emb_file in embedding_files[:100]:  # Check first 100
        try:
            emb = np.load(emb_file)
            shapes.append(emb.shape)
            
            # Check for NaN/Inf
            if np.isnan(emb).any() or np.isinf(emb).any():
                issues.append(f"{emb_file.name}: Contains NaN/Inf")
        except Exception as e:
            issues.append(f"{emb_file.name}: {e}")
    
    # Report
    if shapes:
        unique_shapes = set(shapes)
        print(f"✓ Embedding shapes found: {unique_shapes}")
        if len(unique_shapes) == 1 and (512,) in unique_shapes:
            print("✓ All embeddings have correct shape (512,)")
        else:
            print(f"⚠ WARNING: Expected (512,) but found {unique_shapes}")
    
    if issues:
        print(f"\n⚠ Found {len(issues)} issues:")
        for issue in issues[:10]:
            print(f"  {issue}")
    else:
        print("✓ No issues found in sampled embeddings")

def main():
    parser = argparse.ArgumentParser(description='Encode captions with CLAP')
    parser.add_argument('--captions-dir', type=str, required=True,
                       help='Directory containing caption CSV files')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Directory to save text embeddings')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for encoding')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'], help='Device to use')
    parser.add_argument('--caption-types', type=str, nargs='+',
                       default=['full', 'structural', 'affective', 'baseline'],
                       help='Which caption types to encode')
    args = parser.parse_args()
    
    captions_dir = Path(args.captions_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load CLAP model
    model, device = load_clap_model(args.device)
    
    # Encode each caption type
    all_embeddings = {}
    
    for caption_type in args.caption_types:
        print("\n" + "="*70)
        print(f"Encoding {caption_type.upper()} captions")
        print("="*70)
        
        caption_csv = captions_dir / f"captions_{caption_type}.csv"
        if not caption_csv.exists():
            print(f"⚠ Skipping {caption_type}: {caption_csv} not found")
            continue
        
        type_output_dir = output_dir / caption_type
        
        # Encode
        embedding_paths = encode_captions(
            caption_csv, 
            type_output_dir, 
            model, 
            device,
            batch_size=args.batch_size
        )
        
        all_embeddings[caption_type] = embedding_paths
        
        # Verify
        verify_embeddings(type_output_dir)
    
    # Create index file
    print("\n" + "="*70)
    print("Creating embedding index")
    print("="*70)
    
    if all_embeddings:
        # Get all track IDs (from first caption type)
        first_type = list(all_embeddings.keys())[0]
        track_ids = list(all_embeddings[first_type].keys())
        
        # Build index DataFrame
        index_data = {'track_id': track_ids}
        for caption_type, paths in all_embeddings.items():
            index_data[f'{caption_type}_embedding_path'] = [
                paths.get(tid, '') for tid in track_ids
            ]
        
        index_df = pd.DataFrame(index_data)
        index_file = output_dir / 'embedding_index.csv'
        index_df.to_csv(index_file, index=False)
        
        print(f"✓ Saved embedding index to {index_file}")
        print(f"✓ Total tracks: {len(track_ids)}")
        print(f"✓ Caption types: {list(all_embeddings.keys())}")
    
    print("\n" + "="*70)
    print("CLAP ENCODING COMPLETE!")
    print("="*70)
    print(f"\nOutput directory: {output_dir}")
    print(f"Subdirectories: {[d.name for d in output_dir.iterdir() if d.is_dir()]}")

if __name__ == '__main__':
    main()