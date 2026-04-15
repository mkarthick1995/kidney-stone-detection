import torch
import torch.nn as nn

from .encoders import build_feature_encoder


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int = 1792, out_dim: int = 512):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.act = nn.ReLU(inplace=True)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.act(self.fc(x)))


class CrossAttentionFusion(nn.Module):
    """Bidirectional cross-attention between two modality embeddings."""

    def __init__(self, dim: int = 512, heads: int = 8):
        super().__init__()
        self.ct_to_us = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.us_to_ct = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.proj = nn.Linear(dim * 2, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, ct: torch.Tensor, us: torch.Tensor) -> torch.Tensor:
        # Add sequence dim: (B, 512) -> (B, 1, 512)
        ct_seq, us_seq = ct.unsqueeze(1), us.unsqueeze(1)
        ct_attn, _ = self.ct_to_us(ct_seq, us_seq, us_seq)
        us_attn, _ = self.us_to_ct(us_seq, ct_seq, ct_seq)
        fused = torch.cat([ct_attn.squeeze(1), us_attn.squeeze(1)], dim=-1)
        return self.norm(self.proj(fused) + 0.5 * (ct + us))  # residual


class Classifier(nn.Module):
    def __init__(self, dim: int = 512, num_classes: int = 2, dropout: float = 0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FusionModel(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.ct_encoder = build_feature_encoder(pretrained=pretrained)
        self.us_encoder = build_feature_encoder(pretrained=pretrained)
        self.ct_proj = ProjectionHead()
        self.us_proj = ProjectionHead()
        self.fusion = CrossAttentionFusion()
        self.classifier = Classifier()

    def freeze_encoders(self) -> None:
        for p in self.ct_encoder.parameters():
            p.requires_grad = False
        for p in self.us_encoder.parameters():
            p.requires_grad = False

    def unfreeze_last_blocks(self, n: int = 2) -> None:
        """Unfreeze the last n blocks of each EfficientNet encoder."""
        for enc in (self.ct_encoder, self.us_encoder):
            for block in enc.blocks[-n:]:
                for p in block.parameters():
                    p.requires_grad = True

    def forward(self, ct: torch.Tensor, us: torch.Tensor) -> torch.Tensor:
        ct_feat = self.ct_proj(self.ct_encoder(ct))
        us_feat = self.us_proj(self.us_encoder(us))
        fused = self.fusion(ct_feat, us_feat)
        return self.classifier(fused)
