"""
ST-MGM: Multi-Task Fusion Model
==================================
Combines all three feature branches into a unified prediction:

Feature Fusion (Concatenation):
    F_fusion = {F_temporal (~18), F_morphological (40), F_eda (64)} → ~122 features

Shared MLP:
    Linear(122, 64) → ReLU → Dropout(0.3)
    → Linear(64, 32) → ReLU → Dropout(0.2)

Multi-Task Heads:
    1. Classification Head → Linear(32, 1) → Sigmoid → Hypo/Normal
    2. Regression Head    → Linear(32, 1) → BG Value (mg/dL)

Phase 2 of the ST-MGM pipeline.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from libs.resnet1d import ResNet1D
from libs.eda_encoder import EDAEncoder


# ============================================================
# Multi-Task Fusion Model
# ============================================================

class STMGMModel(nn.Module):
    """
    Spatiotemporal Multimodal Glucose Monitor — Full Model.

    Combines:
        - F_temporal:       Pre-extracted DWT statistical features (~18 dim)
        - F_morphological:  1D-ResNet embeddings from ECG+PPG beats (40 dim)
        - F_eda:            Conv1D+BiLSTM embeddings from EDA signal (64 dim)

    Total fusion input: ~122 features
    Outputs: (classification_prob, regression_value)
    """

    def __init__(self,
                 temporal_dim=18,
                 morph_output_dim=40,
                 eda_output_dim=64,
                 shared_hidden_1=64,
                 shared_dropout_1=0.3,
                 shared_hidden_2=32,
                 shared_dropout_2=0.2):
        super().__init__()

        self.temporal_dim = temporal_dim
        self.morph_output_dim = morph_output_dim
        self.eda_output_dim = eda_output_dim

        # Sub-models for deep feature extraction
        self.resnet = ResNet1D(output_features=morph_output_dim)
        self.eda_encoder = EDAEncoder(output_features=eda_output_dim)

        # Branch-level normalization to balance feature scales before fusion
        self.norm_temporal = nn.LayerNorm(temporal_dim)
        self.norm_morph = nn.LayerNorm(morph_output_dim)
        self.norm_eda = nn.LayerNorm(eda_output_dim)

        # Fusion input dimension
        fusion_dim = temporal_dim + morph_output_dim + eda_output_dim

        # Shared MLP with LayerNorm and GELU to prevent Dying ReLU
        self.shared_mlp = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, shared_hidden_1),
            nn.LayerNorm(shared_hidden_1),
            nn.GELU(),
            nn.Dropout(shared_dropout_1),
            nn.Linear(shared_hidden_1, shared_hidden_2),
            nn.LayerNorm(shared_hidden_2),
            nn.GELU(),
            nn.Dropout(shared_dropout_2),
        )

        # Residual skip projection from 5-min DWT temporal features
        self.temporal_skip = nn.Linear(temporal_dim, shared_hidden_2)

        # Classification Head: Hypo (1) / Normal (0) — outputs raw logits
        self.cls_head = nn.Linear(shared_hidden_2, 1)

        # Regression Head: BG value (normalized during training)
        self.reg_head = nn.Linear(shared_hidden_2, 1)

        # Threshold-aware regression-to-classification logit coupling
        self.register_buffer('reg_threshold', torch.tensor(-1.0, dtype=torch.float32))
        self.reg_logit_scale = nn.Parameter(torch.tensor(1.5, dtype=torch.float32))

    def set_reg_threshold(self, hypo_threshold=70.0, glucose_mean=0.0, glucose_std=1.0):
        """Set normalized hypoglycemia threshold for regression-classification coupling."""
        std_safe = max(float(glucose_std), 1e-6)
        z_thr = (float(hypo_threshold) - float(glucose_mean)) / std_safe
        self.reg_threshold.fill_(z_thr)

    def forward(self, f_temporal, ecg_beats, ppg_pulses, eda_signal):
        """
        Full forward pass through all branches and fusion.

        Args:
            f_temporal:  (batch, ~18) — standardized DWT features
            ecg_beats:   (batch, ecg_len) — representative ECG beats
            ppg_pulses:  (batch, ppg_len) — representative PPG pulses
            eda_signal:  (batch, 1200) — Z-score normalized EDA

        Returns:
            cls_logits: (batch, 1) — classification logits (Hypo/Normal)
            reg_value:  (batch, 1) — predicted glucose value
        """
        # Extract and normalize branch features
        f_temp_norm = self.norm_temporal(f_temporal)                    # (batch, ~18)
        f_morph = self.norm_morph(self.resnet(ecg_beats, ppg_pulses))   # (batch, 40)
        f_eda = self.norm_eda(self.eda_encoder(eda_signal))             # (batch, 64)

        # Feature fusion (concatenation)
        f_fusion = torch.cat([f_temp_norm, f_morph, f_eda], dim=1)      # (batch, ~122)

        # Shared MLP with temporal residual connection
        shared = self.shared_mlp(f_fusion) + self.temporal_skip(f_temp_norm)  # (batch, 32)

        # Dual output heads with threshold-aware coupling
        reg_value = self.reg_head(shared)  # (batch, 1)
        cls_logits = (
            self.cls_head(shared)
            + F.softplus(self.reg_logit_scale) * (self.reg_threshold - reg_value)
        )  # (batch, 1)

        return cls_logits, reg_value


# ============================================================
# Feature-Only Fusion Model (uses pre-extracted features)
# ============================================================

class STMGMFusionOnly(nn.Module):
    """
    Fusion-only variant of ST-MGM.
    Takes all pre-extracted features as input.
    Used when features are pre-computed and stored.

    Inputs: F_temporal, F_morphological, F_eda (all pre-extracted)
    """

    def __init__(self,
                 temporal_dim=18,
                 morph_dim=40,
                 eda_dim=64,
                 shared_hidden_1=64,
                 shared_dropout_1=0.3,
                 shared_hidden_2=32,
                 shared_dropout_2=0.2):
        super().__init__()

        fusion_dim = temporal_dim + morph_dim + eda_dim

        self.norm_temporal = nn.LayerNorm(temporal_dim)
        self.norm_morph = nn.LayerNorm(morph_dim)
        self.norm_eda = nn.LayerNorm(eda_dim)

        self.shared_mlp = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, shared_hidden_1),
            nn.LayerNorm(shared_hidden_1),
            nn.GELU(),
            nn.Dropout(shared_dropout_1),
            nn.Linear(shared_hidden_1, shared_hidden_2),
            nn.LayerNorm(shared_hidden_2),
            nn.GELU(),
            nn.Dropout(shared_dropout_2),
        )

        self.temporal_skip = nn.Linear(temporal_dim, shared_hidden_2)
        self.cls_head = nn.Linear(shared_hidden_2, 1)
        self.reg_head = nn.Linear(shared_hidden_2, 1)

        self.register_buffer('reg_threshold', torch.tensor(-1.0, dtype=torch.float32))
        self.reg_logit_scale = nn.Parameter(torch.tensor(1.5, dtype=torch.float32))

    def set_reg_threshold(self, hypo_threshold=70.0, glucose_mean=0.0, glucose_std=1.0):
        std_safe = max(float(glucose_std), 1e-6)
        z_thr = (float(hypo_threshold) - float(glucose_mean)) / std_safe
        self.reg_threshold.fill_(z_thr)

    def forward(self, f_temporal, f_morph, f_eda):
        """
        Args:
            f_temporal: (batch, ~18)
            f_morph:    (batch, 40)
            f_eda:      (batch, 64)

        Returns:
            cls_logits: (batch, 1)
            reg_value:  (batch, 1)
        """
        f_temp_norm = self.norm_temporal(f_temporal)
        f_morph_norm = self.norm_morph(f_morph)
        f_eda_norm = self.norm_eda(f_eda)
        f_fusion = torch.cat([f_temp_norm, f_morph_norm, f_eda_norm], dim=1)
        shared = self.shared_mlp(f_fusion) + self.temporal_skip(f_temp_norm)
        reg_value = self.reg_head(shared)
        cls_logits = (
            self.cls_head(shared)
            + F.softplus(self.reg_logit_scale) * (self.reg_threshold - reg_value)
        )
        return cls_logits, reg_value


# ============================================================
# Multi-Task Loss
# ============================================================

class MultiTaskLoss(nn.Module):
    """
    Combined loss for multi-task learning:
        L = alpha * BCEWithLogits(cls_logits, cls_label) + beta * (0.5 * MSE + 0.5 * SmoothL1)
    """

    def __init__(self, alpha=0.5, beta=0.5, pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        if pos_weight is not None:
            if not isinstance(pos_weight, torch.Tensor):
                pos_weight = torch.tensor([float(pos_weight)], dtype=torch.float32)
            self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()
        self.huber = nn.SmoothL1Loss(beta=1.0)

    def forward(self, cls_pred, cls_label, reg_pred, reg_label):
        """
        Args:
            cls_pred:  (batch, 1) — predicted logits
            cls_label: (batch,) or (batch, 1) — binary labels
            reg_pred:  (batch, 1) — predicted glucose values (normalized)
            reg_label: (batch,) or (batch, 1) — target glucose values (normalized)

        Returns:
            total_loss: weighted sum of BCEWithLogits + regression loss
            cls_loss: BCEWithLogits loss
            reg_loss: combined MSE + SmoothL1 loss
        """
        if cls_label.dim() == 1:
            cls_label = cls_label.unsqueeze(1)
        if reg_label.dim() == 1:
            reg_label = reg_label.unsqueeze(1)

        cls_loss = self.bce(cls_pred, cls_label.float())
        reg_target = reg_label.float()
        reg_loss = 0.5 * self.mse(reg_pred, reg_target) + 0.5 * self.huber(reg_pred, reg_target)

        total_loss = self.alpha * cls_loss + self.beta * reg_loss
        return total_loss, cls_loss, reg_loss


# ============================================================
# Quick Test
# ============================================================

if __name__ == '__main__':
    print("--- ST-MGM Fusion Model Test ---\n")

    batch_size = 4
    temporal_dim = 18

    # Test full model
    model = STMGMModel(temporal_dim=temporal_dim)
    f_temporal = torch.randn(batch_size, temporal_dim)
    ecg_beats = torch.randn(batch_size, 750)
    ppg_pulses = torch.randn(batch_size, 1920)
    eda_signal = torch.randn(batch_size, 1200)

    cls_logits, reg_value = model(f_temporal, ecg_beats, ppg_pulses, eda_signal)
    cls_prob = torch.sigmoid(cls_logits)
    print(f"Full ST-MGM:")
    print(f"  Inputs: temporal {f_temporal.shape}, ECG {ecg_beats.shape}, "
          f"PPG {ppg_pulses.shape}, EDA {eda_signal.shape}")
    print(f"  Outputs: cls_logits {cls_logits.shape}, reg_value {reg_value.shape}")
    print(f"  cls_prob range: [{cls_prob.min():.4f}, {cls_prob.max():.4f}]")

    # Test fusion-only model
    fusion_model = STMGMFusionOnly(temporal_dim=temporal_dim)
    f_morph = torch.randn(batch_size, 40)
    f_eda = torch.randn(batch_size, 64)

    cls_logits2, reg_value2 = fusion_model(f_temporal, f_morph, f_eda)
    print(f"\nFusion-only model:")
    print(f"  Inputs: temporal {f_temporal.shape}, morph {f_morph.shape}, eda {f_eda.shape}")
    print(f"  Outputs: cls_logits {cls_logits2.shape}, reg_value {reg_value2.shape}")

    # Test multi-task loss with pos_weight
    criterion = MultiTaskLoss(alpha=0.5, beta=0.5, pos_weight=5.0)
    cls_labels = torch.tensor([1, 0, 1, 0], dtype=torch.float32)
    reg_labels = torch.tensor([-1.2, 0.3, -1.5, 0.9], dtype=torch.float32)

    total_loss, cls_loss, reg_loss = criterion(cls_logits, cls_labels, reg_value, reg_labels)
    print(f"\nMulti-Task Loss:")
    print(f"  Total: {total_loss.item():.4f} = "
          f"0.5 * BCEWithLogits({cls_loss.item():.4f}) + 0.5 * MSE({reg_loss.item():.4f})")

    # Parameter count
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal trainable parameters (full model): {params:,}")

    params_fusion = sum(p.numel() for p in fusion_model.parameters() if p.requires_grad)
    print(f"Total trainable parameters (fusion-only): {params_fusion:,}")

    print("\nAll ST-MGM model tests passed!")
