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

import torch
import torch.nn as nn
import torch.nn.functional as F

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

        # Fusion input dimension
        fusion_dim = temporal_dim + morph_output_dim + eda_output_dim

        # Shared MLP
        self.shared_mlp = nn.Sequential(
            nn.Linear(fusion_dim, shared_hidden_1),
            nn.ReLU(inplace=True),
            nn.Dropout(shared_dropout_1),
            nn.Linear(shared_hidden_1, shared_hidden_2),
            nn.ReLU(inplace=True),
            nn.Dropout(shared_dropout_2),
        )

        # Classification Head: Hypo (1) / Normal (0)
        self.cls_head = nn.Sequential(
            nn.Linear(shared_hidden_2, 1),
            nn.Sigmoid(),
        )

        # Regression Head: BG value in mg/dL
        self.reg_head = nn.Linear(shared_hidden_2, 1)

    def forward(self, f_temporal, ecg_beats, ppg_pulses, eda_signal):
        """
        Full forward pass through all branches and fusion.

        Args:
            f_temporal:  (batch, ~18) — pre-extracted DWT features
            ecg_beats:   (batch, ecg_len) — individual ECG beats
            ppg_pulses:  (batch, ppg_len) — individual PPG pulses
            eda_signal:  (batch, 1200) — Z-score normalized EDA

        Returns:
            cls_prob: (batch, 1) — classification probability (Hypo/Normal)
            reg_value: (batch, 1) — predicted glucose value (mg/dL)
        """
        # Extract deep features
        f_morph = self.resnet(ecg_beats, ppg_pulses)  # (batch, 40)
        f_eda = self.eda_encoder(eda_signal)           # (batch, 64)

        # Feature fusion (concatenation)
        f_fusion = torch.cat([f_temporal, f_morph, f_eda], dim=1)  # (batch, ~122)

        # Shared MLP
        shared = self.shared_mlp(f_fusion)  # (batch, 32)

        # Dual output heads
        cls_prob = self.cls_head(shared)    # (batch, 1)
        reg_value = self.reg_head(shared)   # (batch, 1)

        return cls_prob, reg_value


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

        self.shared_mlp = nn.Sequential(
            nn.Linear(fusion_dim, shared_hidden_1),
            nn.ReLU(inplace=True),
            nn.Dropout(shared_dropout_1),
            nn.Linear(shared_hidden_1, shared_hidden_2),
            nn.ReLU(inplace=True),
            nn.Dropout(shared_dropout_2),
        )

        self.cls_head = nn.Sequential(
            nn.Linear(shared_hidden_2, 1),
            nn.Sigmoid(),
        )

        self.reg_head = nn.Linear(shared_hidden_2, 1)

    def forward(self, f_temporal, f_morph, f_eda):
        """
        Args:
            f_temporal: (batch, ~18)
            f_morph:    (batch, 40)
            f_eda:      (batch, 64)

        Returns:
            cls_prob:  (batch, 1)
            reg_value: (batch, 1)
        """
        f_fusion = torch.cat([f_temporal, f_morph, f_eda], dim=1)
        shared = self.shared_mlp(f_fusion)
        cls_prob = self.cls_head(shared)
        reg_value = self.reg_head(shared)
        return cls_prob, reg_value


# ============================================================
# Multi-Task Loss
# ============================================================

class MultiTaskLoss(nn.Module):
    """
    Combined loss for multi-task learning:
        L = alpha * BCE(cls_pred, cls_label) + beta * MSE(reg_pred, reg_label)
    """

    def __init__(self, alpha=0.5, beta=0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.bce = nn.BCELoss()
        self.mse = nn.MSELoss()

    def forward(self, cls_pred, cls_label, reg_pred, reg_label):
        """
        Args:
            cls_pred:  (batch, 1) — predicted probabilities
            cls_label: (batch,) or (batch, 1) — binary labels
            reg_pred:  (batch, 1) — predicted glucose values
            reg_label: (batch,) or (batch, 1) — true glucose values

        Returns:
            total_loss: weighted sum of BCE + MSE
            cls_loss: BCE loss
            reg_loss: MSE loss
        """
        if cls_label.dim() == 1:
            cls_label = cls_label.unsqueeze(1)
        if reg_label.dim() == 1:
            reg_label = reg_label.unsqueeze(1)

        cls_loss = self.bce(cls_pred, cls_label.float())
        reg_loss = self.mse(reg_pred, reg_label.float())

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

    cls_prob, reg_value = model(f_temporal, ecg_beats, ppg_pulses, eda_signal)
    print(f"Full ST-MGM:")
    print(f"  Inputs: temporal {f_temporal.shape}, ECG {ecg_beats.shape}, "
          f"PPG {ppg_pulses.shape}, EDA {eda_signal.shape}")
    print(f"  Outputs: cls_prob {cls_prob.shape}, reg_value {reg_value.shape}")
    print(f"  cls_prob range: [{cls_prob.min():.4f}, {cls_prob.max():.4f}]")

    # Test fusion-only model
    fusion_model = STMGMFusionOnly(temporal_dim=temporal_dim)
    f_morph = torch.randn(batch_size, 40)
    f_eda = torch.randn(batch_size, 64)

    cls_prob2, reg_value2 = fusion_model(f_temporal, f_morph, f_eda)
    print(f"\nFusion-only model:")
    print(f"  Inputs: temporal {f_temporal.shape}, morph {f_morph.shape}, eda {f_eda.shape}")
    print(f"  Outputs: cls_prob {cls_prob2.shape}, reg_value {reg_value2.shape}")

    # Test multi-task loss
    criterion = MultiTaskLoss(alpha=0.5, beta=0.5)
    cls_labels = torch.tensor([1, 0, 1, 0], dtype=torch.float32)
    reg_labels = torch.tensor([65, 100, 55, 120], dtype=torch.float32)

    total_loss, cls_loss, reg_loss = criterion(cls_prob, cls_labels, reg_value, reg_labels)
    print(f"\nMulti-Task Loss:")
    print(f"  Total: {total_loss.item():.4f} = "
          f"0.5 * BCE({cls_loss.item():.4f}) + 0.5 * MSE({reg_loss.item():.4f})")

    # Parameter count
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal trainable parameters (full model): {params:,}")

    params_fusion = sum(p.numel() for p in fusion_model.parameters() if p.requires_grad)
    print(f"Total trainable parameters (fusion-only): {params_fusion:,}")

    print("\nAll ST-MGM model tests passed!")
