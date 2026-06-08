"""Class-guided visual token adapter used for LLaVA-1.5 CODA-LM experiments."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassGuidedVisualAdapter(nn.Module):
    """Refine frozen VLM visual tokens using learned class embeddings.

    The adapter encodes LLaVA visual tokens into the class-embedding space,
    applies cross-attention over class prototypes, and decodes the refined
    latent tokens back to the language-model embedding dimension.
    """

    def __init__(
        self,
        visual_dim: int = 4096,
        latent_dim: int = 1024,
        class_emb_dim: int = 1024,
        num_classes: int = 29,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.visual_dim = visual_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        self.encoder = nn.Sequential(
            nn.Linear(visual_dim, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2048, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        self.class_proj = nn.Identity() if class_emb_dim == latent_dim else nn.Linear(class_emb_dim, latent_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(latent_dim)
        self.ffn = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 4, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2048, visual_dim),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, visual_tokens: torch.Tensor, class_embeddings: torch.Tensor):
        if visual_tokens.shape[-1] != self.visual_dim:
            raise ValueError(f"Expected visual_dim={self.visual_dim}, got {visual_tokens.shape[-1]}")

        class_embeddings = class_embeddings.to(device=visual_tokens.device, dtype=visual_tokens.dtype)
        latent = self.encoder(visual_tokens)
        class_tokens = self.class_proj(class_embeddings).unsqueeze(0).expand(visual_tokens.shape[0], -1, -1)

        attn_out, attn_weights = self.cross_attn(query=latent, key=class_tokens, value=class_tokens)
        latent = self.attn_norm(latent + attn_out)
        latent = latent + self.ffn(latent)
        refined = self.decoder(latent)
        return refined, latent, attn_weights


class AdapterLoss(nn.Module):
    """Reconstruction plus autoregressive VQA loss."""

    def __init__(self, reconstruction_weight: float = 1.0, autoregressive_weight: float = 1.0) -> None:
        super().__init__()
        self.reconstruction_weight = reconstruction_weight
        self.autoregressive_weight = autoregressive_weight

    def forward(self, refined_tokens: torch.Tensor, original_tokens: torch.Tensor, lm_loss: torch.Tensor):
        reconstruction = F.mse_loss(refined_tokens.float(), original_tokens.float())
        total = self.reconstruction_weight * reconstruction + self.autoregressive_weight * lm_loss
        return total, {
            "total": float(total.detach().cpu()),
            "reconstruction": float(reconstruction.detach().cpu()),
            "autoregressive": float(lm_loss.detach().cpu()),
        }


def load_class_embeddings(checkpoint_path: str | Path, device: str | torch.device = "cpu") -> torch.Tensor:
    """Load class prototypes from the public checkpoint format."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    for key in ("bank.w", "class_embeddings", "prototypes"):
        if key in checkpoint:
            return checkpoint[key].to(device)
    raise KeyError(f"No class embedding tensor found in {checkpoint_path}. Keys: {sorted(checkpoint.keys())}")
