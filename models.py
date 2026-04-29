#!/usr/bin/env python3
"""
Adapter Models for Audio-Text Alignment
Lightweight MLPs that project frozen embeddings to shared space
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AudioAdapter(nn.Module):
    """
    Audio adapter: Projects MYNA embeddings to shared space
    """
    def __init__(self, input_dim: int, hidden_dim: int = 768, output_dim: int = 256):
        super().__init__()
        
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), # Added stability
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.adapter(x)
        return F.normalize(x, p=2, dim=-1)

class TextAdapter(nn.Module):
    """
    Text adapter: Projects CLAP text embeddings to shared space
    """
    def __init__(self, input_dim: int = 512, hidden_dim: int = 512, output_dim: int = 256):
        super().__init__()
        
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), # Added stability
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.adapter(x)
        return F.normalize(x, p=2, dim=-1)

class ContrastiveModel(nn.Module):
    """
    Complete model with audio and text adapters
    """
    def __init__(
        self,
        audio_input_dim: int,
        text_input_dim: int = 512,
        audio_hidden_dim: int = 768,
        text_hidden_dim: int = 512,
        output_dim: int = 256,
        temperature: float = 0.07
    ):
        super().__init__()
        
        self.audio_adapter = AudioAdapter(audio_input_dim, audio_hidden_dim, output_dim)
        self.text_adapter = TextAdapter(text_input_dim, text_hidden_dim, output_dim)
        
        # Safely parameterized log-temperature (CLIP style)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))
        self.output_dim = output_dim
    
    def forward(self, audio_emb: torch.Tensor, text_emb: torch.Tensor):
        audio_proj = self.audio_adapter(audio_emb)
        text_proj = self.text_adapter(text_emb)
        return audio_proj, text_proj
    
    def get_similarity_matrix(self, audio_emb: torch.Tensor, text_emb: torch.Tensor):
        audio_proj, text_proj = self.forward(audio_emb, text_emb)
        
        # Clamp logit scale to prevent exploding gradients (max ~100)
        logit_scale = torch.clamp(self.logit_scale.exp(), max=100)
        
        # Multiply by scale instead of dividing by temperature
        similarity = (audio_proj @ text_proj.T) * logit_scale
        return similarity

def info_nce_loss(similarity_matrix: torch.Tensor) -> torch.Tensor:
    batch_size = similarity_matrix.shape[0]
    labels = torch.arange(batch_size, device=similarity_matrix.device)
    
    loss_a2t = F.cross_entropy(similarity_matrix, labels)
    loss_t2a = F.cross_entropy(similarity_matrix.T, labels)
    loss = (loss_a2t + loss_t2a) / 2
    return loss

# Test
if __name__ == '__main__':
    print("="*70)
    print("TESTING ADAPTER MODELS")
    print("="*70)
    
    # Test with MYNA-768 dimensions
    batch_size = 32
    audio_dim = 768
    text_dim = 512
    
    # Create model
    model = ContrastiveModel(
        audio_input_dim=audio_dim,
        text_input_dim=text_dim,
        audio_hidden_dim=512,
        text_hidden_dim=512,
        output_dim=256
    )
    
    print(f"\n✓ Model created")
    print(f"  Audio adapter params: {sum(p.numel() for p in model.audio_adapter.parameters()):,}")
    print(f"  Text adapter params: {sum(p.numel() for p in model.text_adapter.parameters()):,}")
    print(f"  Total params: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    audio = torch.randn(batch_size, audio_dim)
    text = torch.randn(batch_size, text_dim)
    
    audio_proj, text_proj = model(audio, text)
    
    print(f"\n✓ Forward pass successful")
    print(f"  Audio projection: {audio_proj.shape}")
    print(f"  Text projection: {text_proj.shape}")
    print(f"  Audio L2 norm: {torch.norm(audio_proj, dim=-1).mean():.4f} (should be ~1.0)")
    print(f"  Text L2 norm: {torch.norm(text_proj, dim=-1).mean():.4f} (should be ~1.0)")
    
    # Test loss computation
    similarity = model.get_similarity_matrix(audio, text)
    loss = info_nce_loss(similarity)
    
    print(f"\n✓ Loss computation successful")
    print(f"  Similarity matrix: {similarity.shape}")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Logit scale: {model.logit_scale.exp().item():.4f}")
    
    # Test with MYNA-85m dimensions
    print(f"\n{'='*70}")
    print("Testing with MYNA-85m (1536-dim)")
    print('='*70)
    
    model_85m = ContrastiveModel(
        audio_input_dim=1536,
        text_input_dim=512,
        audio_hidden_dim=768,
        text_hidden_dim=512,
        output_dim=256
    )
    
    audio_85m = torch.randn(batch_size, 1536)
    audio_proj_85m, text_proj_85m = model_85m(audio_85m, text)
    
    print(f"✓ MYNA-85m model works!")
    print(f"  Audio adapter params: {sum(p.numel() for p in model_85m.audio_adapter.parameters()):,}")
    print(f"  Projection shape: {audio_proj_85m.shape}")
    
    print("\n" + "="*70)
    print("All tests passed!")
    print("="*70)