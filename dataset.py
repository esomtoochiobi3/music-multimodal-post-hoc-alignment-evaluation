#!/usr/bin/env python3
"""
Dataset for Audio-Text Contrastive Learning
Loads paired MYNA audio embeddings and CLAP text embeddings
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm 
import concurrent.futures # <--- Added for blazing fast multi-threaded I/O

class AudioTextDataset(Dataset):
    """
    Dataset for audio-text contrastive learning
    
    Loads:
    - Audio embeddings: MYNA embeddings (768-dim or 1536-dim)
    - Text embeddings: CLAP text embeddings (512-dim)
    - Preloads all data into memory using multi-threading to bypass HPRC I/O limits.
    """
    
    def __init__(
        self,
        audio_embedding_dir: str,
        text_embedding_dir: str,
        text_embedding_type: str = 'full',
        split: str = 'train',
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        seed: int = 42,
        precomputed_ids: list = None  # pass in to skip redundant glob scans
    ):
        self.audio_embedding_dir = Path(audio_embedding_dir)
        self.text_embedding_dir = Path(text_embedding_dir) / text_embedding_type
        self.text_embedding_type = text_embedding_type
        self.split = split

        # Use precomputed IDs if provided — avoids scanning 3x for train/val/test
        if precomputed_ids is not None:
            common_ids = precomputed_ids
            print(f"Using {len(common_ids)} precomputed common IDs (skipping glob scan)")
        else:
            print(f"Scanning audio embeddings...")
            audio_ids = set(f.stem for f in self.audio_embedding_dir.glob("*.npy"))
            if len(audio_ids) == 0:
                raise ValueError(f"No audio embeddings found in {self.audio_embedding_dir}")
            print(f"Scanning text embeddings...")
            text_ids = set(f.stem for f in self.text_embedding_dir.glob("*.npy"))
            common_ids = sorted(audio_ids & text_ids)
            print(f"Audio: {len(audio_ids)} | Text: {len(text_ids)} | Common: {len(common_ids)}")

        if len(common_ids) == 0:
            raise ValueError(f"No matching audio-text pairs found!")

        valid_pairs = [
            {
                'track_id': tid,
                'audio_path': self.audio_embedding_dir / f"{tid}.npy",
                'text_path': self.text_embedding_dir / f"{tid}.npy"
            }
            for tid in common_ids
        ]
        
        # Split into train/val/test
        np.random.seed(seed)
        indices = np.random.permutation(len(valid_pairs))
        
        n_train = int(len(valid_pairs) * train_ratio)
        n_val = int(len(valid_pairs) * val_ratio)
        
        if split == 'train':
            split_indices = indices[:n_train]
        elif split == 'val':
            split_indices = indices[n_train:n_train+n_val]
        elif split == 'test':
            split_indices = indices[n_train+n_val:]
        else:
            raise ValueError(f"Invalid split: {split}")
            
        self.pairs = [valid_pairs[i] for i in split_indices]
        print(f"Split '{split}': {len(self.pairs)} pairs to load")

        # --- MULTI-THREADED PRELOADING ---
        print(f"Preloading {split} embeddings into RAM using 32 concurrent workers...")
        
        # Pre-allocate lists so we can insert out-of-order but maintain exact alignment
        audio_data_list = [None] * len(self.pairs)
        text_data_list = [None] * len(self.pairs)
        self.track_ids = [None] * len(self.pairs)
        
        # Helper function for the threads to run
        def load_pair(index, pair_dict):
            a = np.load(pair_dict['audio_path']).astype(np.float32)
            t = np.load(pair_dict['text_path']).astype(np.float32)
            return index, a, t, pair_dict['track_id']
            
        # Launch 32 simultaneous requests to the file system
        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
            futures = [executor.submit(load_pair, i, pair) for i, pair in enumerate(self.pairs)]
            
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(self.pairs), desc=f"Loading {split} set"):
                i, a, t, tid = future.result()
                audio_data_list[i] = torch.from_numpy(a)
                text_data_list[i] = torch.from_numpy(t)
                self.track_ids[i] = tid
            
        # Stack lists into single tensors for optimal memory access
        self.audio_data = torch.stack(audio_data_list)
        self.text_data = torch.stack(text_data_list)
        
        self.audio_dim = self.audio_data.shape[1]
        self.text_dim = self.text_data.shape[1]
        
        print(f"✓ {split.capitalize()} set loaded! Audio dim: {self.audio_dim}, Text dim: {self.text_dim}")
    
    def __len__(self) -> int:
        return len(self.track_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'audio': self.audio_data[idx],
            'text': self.text_data[idx],
            'track_id': self.track_ids[idx]
        }

def create_dataloaders(
    audio_embedding_dir: str,
    text_embedding_dir: str,
    text_embedding_type: str = 'full',
    batch_size: int = 256,
    num_workers: int = 4,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    # Scan once and share IDs across all 3 splits — avoids 6 redundant glob scans
    print(f"Scanning embeddings once for all splits...")
    audio_ids = set(f.stem for f in Path(audio_embedding_dir).glob("*.npy"))
    text_ids = set(f.stem for f in (Path(text_embedding_dir) / text_embedding_type).glob("*.npy"))
    common_ids = sorted(audio_ids & text_ids)
    print(f"Audio: {len(audio_ids)} | Text: {len(text_ids)} | Common: {len(common_ids)}")

    train_dataset = AudioTextDataset(
        audio_embedding_dir=audio_embedding_dir,
        text_embedding_dir=text_embedding_dir,
        text_embedding_type=text_embedding_type,
        split='train',
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        precomputed_ids=common_ids
    )
    
    val_dataset = AudioTextDataset(
        audio_embedding_dir=audio_embedding_dir,
        text_embedding_dir=text_embedding_dir,
        text_embedding_type=text_embedding_type,
        split='val',
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        precomputed_ids=common_ids
    )
    
    test_dataset = AudioTextDataset(
        audio_embedding_dir=audio_embedding_dir,
        text_embedding_dir=text_embedding_dir,
        text_embedding_type=text_embedding_type,
        split='test',
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        precomputed_ids=common_ids
    )
    
    # num_workers=0 since data is preloaded into RAM
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=True 
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    
    return train_loader, val_loader, test_loader, train_dataset.audio_dim, train_dataset.text_dim

# Test
if __name__ == '__main__':
    # Test dataset creation
    audio_dir = '/scratch/user/esomtoochiobi/thesis_work/outputs/features_50k/myna_embeddings'
    text_dir = '/scratch/user/esomtoochiobi/thesis_work/outputs/text_embeddings_50k'
    
    print("="*70)
    print("TESTING DATASET CREATION")
    print("="*70)
    
    try:
        train_loader, val_loader, test_loader, audio_dim, text_dim = create_dataloaders(
            audio_embedding_dir=audio_dir,
            text_embedding_dir=text_dir,
            text_embedding_type='full',
            batch_size=32,
            num_workers=0,  # 0 for testing
            seed=42
        )
        
        print(f"\n✓ Dataloaders created successfully!")
        print(f"  Audio dim: {audio_dim}, Text dim: {text_dim}")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        print(f"  Test batches: {len(test_loader)}")
        
        # Test loading one batch
        batch = next(iter(train_loader))
        print(f"\n✓ Sample batch:")
        print(f"  Audio shape: {batch['audio'].shape}")
        print(f"  Text shape: {batch['text'].shape}")
        print(f"  Track IDs: {batch['track_id'][:3]}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()