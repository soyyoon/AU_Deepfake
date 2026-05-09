"""
Stage 1: Artifact-based deepfake detector.

Backbone: Xception (FaceForensics++/CelebDF standard).
Output: single logit (binary). sigmoid -> P(fake).
"""
import torch
import torch.nn as nn
import timm


class ArtifactNet(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            "xception", pretrained=pretrained, num_classes=0, global_pool="avg"
        )
        feat_dim = self.backbone.num_features  # 2048 for xception
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, 299, 299) normalized to ImageNet stats
        feat = self.backbone(x)
        return self.head(feat).squeeze(-1)  # (B,) logit
