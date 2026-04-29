#!/usr/bin/env python3
"""
Train Adapters for Audio-Text Contrastive Learning
Trains lightweight adapters to align MYNA audio and CLAP text embeddings
"""

import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import argparse
from pathlib import Path
import json
from tqdm import tqdm
import numpy as np

from dataset import create_dataloaders
from models import ContrastiveModel, info_nce_loss

def train_epoch(model, train_loader, optimizer, scaler, device, epoch):
    """Train for one epoch with mixed precision and gradient clipping"""
    model.train()
    
    total_loss = 0
    total_batches = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch in pbar:
        audio = batch['audio'].to(device)
        text = batch['text'].to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision forward pass
        with autocast():
            similarity = model.get_similarity_matrix(audio, text)
            loss = info_nce_loss(similarity)
        
        # Scaled backward pass
        scaler.scale(loss).backward()
        
        # Gradient clipping — prevents occasional loss spikes
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        # Track metrics
        total_loss += loss.item()
        total_batches += 1
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / total_batches
    return avg_loss

@torch.no_grad()
def validate(model, val_loader, device):
    """Validate on validation set"""
    model.eval()
    
    total_loss = 0
    total_batches = 0
    
    for batch in val_loader:
        audio = batch['audio'].to(device)
        text = batch['text'].to(device)
        
        # Forward pass
        similarity = model.get_similarity_matrix(audio, text)
        loss = info_nce_loss(similarity)
        
        total_loss += loss.item()
        total_batches += 1
    
    avg_loss = total_loss / total_batches
    return avg_loss

@torch.no_grad()
def compute_map(similarity_matrix):
    """
    Compute Mean Average Precision (mAP)
    
    Args:
        similarity_matrix: (N, N) similarity scores
        
    Returns:
        mAP score for both audio->text and text->audio
    """
    N = similarity_matrix.shape[0]
    
    # Audio -> Text mAP
    aps_a2t = []
    for i in range(N):
        # Get ranking for this query (descending similarity)
        ranked = torch.argsort(similarity_matrix[i], descending=True)
        
        # Find position of correct match (i-th text for i-th audio)
        position = (ranked == i).nonzero(as_tuple=True)[0].item() + 1  # 1-indexed
        
        # AP = 1 / position (for single relevant item)
        ap = 1.0 / position
        aps_a2t.append(ap)
    
    map_a2t = np.mean(aps_a2t)
    
    # Text -> Audio mAP (transpose similarity matrix)
    aps_t2a = []
    for i in range(N):
        ranked = torch.argsort(similarity_matrix[:, i], descending=True)
        position = (ranked == i).nonzero(as_tuple=True)[0].item() + 1
        ap = 1.0 / position
        aps_t2a.append(ap)
    
    map_t2a = np.mean(aps_t2a)
    
    return map_a2t, map_t2a

@torch.no_grad()
def compute_retrieval_metrics(model, dataloader, device, top_k=[1, 5, 10, 50, 100]):
    """
    Compute retrieval metrics: Recall@K and mAP
    Accelerated via GPU vectorization.
    """
    model.eval()
    
    all_audio_proj = []
    all_text_proj = []
    
    # Collect all embeddings directly on the GPU
    for batch in tqdm(dataloader, desc="Computing embeddings"):
        audio = batch['audio'].to(device)
        text = batch['text'].to(device)
        
        audio_proj, text_proj = model(audio, text)
        
        all_audio_proj.append(audio_proj)
        all_text_proj.append(text_proj)
    
    # Concatenate on GPU 
    all_audio_proj = torch.cat(all_audio_proj, dim=0)  # (N, D)
    all_text_proj = torch.cat(all_text_proj, dim=0)    # (N, D)
    
    N = all_audio_proj.shape[0]
    
    # Compute similarity matrix entirely on GPU
    similarity = all_audio_proj @ all_text_proj.T
    
    valid_top_k = [k for k in top_k if k <= N]
    if len(valid_top_k) < len(top_k):
        print(f"  Note: Test set has {N} samples, limiting K values to {valid_top_k}")
    
    max_k = max(valid_top_k) if valid_top_k else N
    
    # --- Vectorized Recall@K ---
    metrics = {}
    
    # The correct match index for item i is just i
    target = torch.arange(N, device=device).unsqueeze(1)
    
    # Audio -> Text
    _, indices_a2t = torch.topk(similarity, k=min(max_k, N), dim=1)
    for k in valid_top_k:
        # Check how many target indices appear in the top-k predictions
        correct_a2t = (indices_a2t[:, :k] == target).sum().item()
        metrics[f'recall@{k}'] = correct_a2t / N
        
    # Text -> Audio (transpose similarity matrix)
    _, indices_t2a = torch.topk(similarity.T, k=min(max_k, N), dim=1)
    for k in valid_top_k:
        correct_t2a = (indices_t2a[:, :k] == target).sum().item()
        metrics[f'recall@{k}_t2a'] = correct_t2a / N
        
        # Mean
        metrics[f'recall@{k}_mean'] = (metrics[f'recall@{k}'] + metrics[f'recall@{k}_t2a']) / 2

    # --- Vectorized mAP (No for-loops) ---
    # Extract the diagonal (the similarity score of the correct matches)
    diag_a2t = similarity.diag().unsqueeze(1)  # (N, 1)
    # Rank is the number of items with a higher similarity score than the correct match, plus 1
    ranks_a2t = (similarity > diag_a2t).sum(dim=1) + 1
    map_a2t = (1.0 / ranks_a2t.float()).mean().item()
    
    diag_t2a = similarity.diag().unsqueeze(0)  # (1, N)
    ranks_t2a = (similarity > diag_t2a).sum(dim=0) + 1
    map_t2a = (1.0 / ranks_t2a.float()).mean().item()
    
    metrics['mAP'] = map_a2t
    metrics['mAP_t2a'] = map_t2a
    metrics['mAP_mean'] = (map_a2t + map_t2a) / 2
    
    return metrics

def save_checkpoint(model, optimizer, epoch, val_loss, metrics, save_path):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'metrics': metrics,
    }
    torch.save(checkpoint, save_path)
    print(f"  ✓ Saved checkpoint to {save_path}")

def train_adapter(
    audio_embedding_dir: str,
    text_embedding_dir: str,
    caption_type: str,
    output_dir: str,
    audio_dim: int = 768,
    text_dim: int = 512,
    hidden_dim: int = 512,
    output_dim: int = 256,
    batch_size: int = 256,
    num_epochs: int = 50,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    patience: int = 10,
    device: str = 'cuda',
    seed: int = 42,
    resume: bool = False
):
    """
    Train adapter for a specific caption type
    
    Args:
        audio_embedding_dir: Path to audio embeddings
        text_embedding_dir: Path to text embeddings (contains subdirs)
        caption_type: 'full', 'structural', 'affective', or 'baseline'
        output_dir: Where to save checkpoints and logs
        audio_dim: Audio embedding dimension
        text_dim: Text embedding dimension
        hidden_dim: Hidden layer size
        output_dim: Shared space dimension
        batch_size: Batch size for training
        num_epochs: Maximum number of epochs
        learning_rate: Learning rate
        weight_decay: Weight decay for regularization
        patience: Early stopping patience
        device: 'cuda' or 'cpu'
        seed: Random seed
    """
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    writer = SummaryWriter(output_dir / 'tensorboard')
    
    print("="*70)
    print(f"TRAINING ADAPTER: {caption_type.upper()}")
    print("="*70)
    print(f"Audio dim: {audio_dim}, Text dim: {text_dim}")
    print(f"Hidden dim: {hidden_dim}, Output dim: {output_dim}")
    print(f"Batch size: {batch_size}, LR: {learning_rate}")
    print(f"Device: {device}")
    print()
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader, test_loader, _, _ = create_dataloaders(
        audio_embedding_dir=audio_embedding_dir,
        text_embedding_dir=text_embedding_dir,
        text_embedding_type=caption_type,
        batch_size=batch_size,
        num_workers=0,
        seed=seed
    )
    
    # Create model
    print("\nCreating model...")
    model = ContrastiveModel(
        audio_input_dim=audio_dim,
        text_input_dim=text_dim,
        audio_hidden_dim=audio_dim,
        text_hidden_dim=text_dim,
        output_dim=output_dim
    )
    model = model.to(device)
    
    # Compile model for faster execution (PyTorch 2.0+)
    # First epoch will be slow while compiling, then significantly faster after
    if device == 'cuda':
        print("  Compiling model with torch.compile...")
        model = torch.compile(model)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # GradScaler for mixed precision
    scaler = GradScaler()
    
    # Cosine LR schedule — decays smoothly over all epochs, more stable
    # for contrastive learning than ReduceLROnPlateau which can cause
    # premature convergence to sharp minima
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=1e-6
    )
    
    # Training loop
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    start_epoch = 1
    
    checkpoint_path = output_dir / 'best_model.pt'
    if resume and checkpoint_path.exists():
        print("\n" + "="*70)
        print(f"RESUMING FROM CHECKPOINT: {checkpoint_path.name}")
        print("="*70)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['val_loss']
        print(f"  Resumed at epoch {start_epoch}, previous best val loss: {best_val_loss:.4f}")
    
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)
    
    for epoch in range(start_epoch, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("-" * 70)
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scaler, device, epoch)
        print(f"  Train loss: {train_loss:.4f}")
        
        # Validate
        val_loss = validate(model, val_loader, device)
        print(f"  Val loss: {val_loss:.4f}")
        
        # Log to tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('Temperature', (1 / model.logit_scale.exp()).item(), epoch)
        
        # Learning rate scheduling
        scheduler.step()
        
        # Save checkpoint if best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            
            # Compute retrieval metrics on validation set
            print("  Computing retrieval metrics...")
            metrics = compute_retrieval_metrics(model, val_loader, device)
            
            for metric, value in metrics.items():
                print(f"    {metric}: {value:.6f}")
                writer.add_scalar(f'Metrics/{metric}', value, epoch)
            
            # Save best checkpoint
            save_checkpoint(
                model, optimizer, epoch, val_loss, metrics,
                output_dir / 'best_model.pt'
            )
        else:
            epochs_without_improvement += 1
        
        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"\n  Early stopping triggered after {epoch} epochs")
            break
    
    # Final evaluation on test set
    print("\n" + "="*70)
    print("FINAL EVALUATION ON TEST SET")
    print("="*70)
    
    # Load best model
    checkpoint = torch.load(output_dir / 'best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate
    test_metrics = compute_retrieval_metrics(model, test_loader, device)
    
    print("\nTest Set Results:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.6f}")
    
    # Save final results
    results = {
        'caption_type': caption_type,
        'best_epoch': checkpoint['epoch'],
        'best_val_loss': checkpoint['val_loss'],
        'val_metrics': checkpoint['metrics'],
        'test_metrics': test_metrics,
        'config': {
            'audio_dim': audio_dim,
            'text_dim': text_dim,
            'hidden_dim': hidden_dim,
            'output_dim': output_dim,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
        }
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_dir / 'results.json'}")
    
    writer.close()
    
    return model, results

def main():
    parser = argparse.ArgumentParser(description='Train audio-text adapters')
    
    # Data paths
    parser.add_argument('--audio-embedding-dir', type=str, required=True,
                       help='Directory with audio embeddings')
    parser.add_argument('--text-embedding-dir', type=str, required=True,
                       help='Directory with text embeddings')
    parser.add_argument('--caption-types', type=str, nargs='+',
                       default=['full', 'structural', 'affective', 'baseline'],
                       help='Caption types to train')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for checkpoints')
    
    # Model architecture
    parser.add_argument('--audio-dim', type=int, default=768,
                       help='Audio embedding dimension')
    parser.add_argument('--text-dim', type=int, default=512,
                       help='Text embedding dimension')
    parser.add_argument('--hidden-dim', type=int, default=512,
                       help='Hidden layer dimension')
    parser.add_argument('--output-dim', type=int, default=256,
                       help='Shared space dimension')
    
    # Training
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Batch size')
    parser.add_argument('--num-epochs', type=int, default=50,
                       help='Maximum number of epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    
    # Other
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'])
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from the last saved best checkpoint') 
    
    args = parser.parse_args()
    
    # Train adapter for each caption type
    all_results = {}
    
    for caption_type in args.caption_types:
        print("\n" + "="*70)
        print(f"TRAINING: {caption_type.upper()}")
        print("="*70)
        
        output_dir = Path(args.output_dir) / caption_type
        
        model, results = train_adapter(
            audio_embedding_dir=args.audio_embedding_dir,
            text_embedding_dir=args.text_embedding_dir,
            caption_type=caption_type,
            output_dir=str(output_dir),
            audio_dim=args.audio_dim,
            text_dim=args.text_dim,
            hidden_dim=args.hidden_dim,
            output_dim=args.output_dim,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            patience=args.patience,
            device=args.device,
            seed=args.seed,
            resume=args.resume
        )
        
        all_results[caption_type] = results
    
    # Save comparison of all models
    comparison_file = Path(args.output_dir) / 'all_results.json'
    with open(comparison_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "="*70)
    print("ALL TRAINING COMPLETE")
    print("="*70)
    print(f"Results saved to: {comparison_file}")
    
    # Print comparison table
    print("\nTest Set Comparison (Recall@K):")
    print("-" * 90)
    print(f"{'Caption Type':<15} {'R@1':>8} {'R@5':>8} {'R@10':>8} {'R@50':>8} {'R@100':>8} {'mAP':>10}")
    print("-" * 90)
    
    for caption_type, results in all_results.items():
        test_metrics = results['test_metrics']
        r1 = test_metrics.get('recall@1_mean', 0)
        r5 = test_metrics.get('recall@5_mean', 0)
        r10 = test_metrics.get('recall@10_mean', 0)
        r50 = test_metrics.get('recall@50_mean', 0)
        r100 = test_metrics.get('recall@100_mean', 0)
        map_score = test_metrics.get('mAP_mean', 0)
        
        print(f"{caption_type:<15} "
              f"{r1:>7.4f}  "
              f"{r5:>7.4f}  "
              f"{r10:>7.4f}  "
              f"{r50:>7.4f}  "
              f"{r100:>7.4f}  "
              f"{map_score:>9.6f}")
    
    print("-" * 90)

if __name__ == '__main__':
    main()