"""
ST-MGM: 1D-ResNet for Spatial Morphological Feature Extraction
================================================================
Dual-branch 1D-ResNet (5-layer) for ECG & PPG individual beats/pulses.
Extracts morphological embeddings and compresses via DNN → F_morphological (40 features).

Architecture per branch:
    Conv1d → [ConvBlock + IdentityBlock] × 4 → f(x) + x (residual)

DNN Feature Compression:
    Concatenate ECG + PPG embeddings
    Linear(in, 512) → Dropout(0.3) → Linear(512, 128) → Dropout(0.2) → Linear(128, 40)

Phase 2 of the ST-MGM pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# ResNet Building Blocks
# ============================================================

class ConvBlock1D(nn.Module):
    """Convolutional block with optional downsampling (stride)."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Shortcut for dimension matching
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + residual)
        return out


class IdentityBlock1D(nn.Module):
    """Identity block (no downsampling, same dimensions)."""

    def __init__(self, channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size,
                               padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size,
                               padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + residual)
        return out


# ============================================================
# Single Branch 1D-ResNet (5 layers)
# ============================================================

class ResNet1DBranch(nn.Module):
    """
    1D-ResNet branch for a single modality (ECG or PPG).

    Architecture:
        Stem Conv1d(1 → 32) → [ConvBlock + IdentityBlock] × 4 → GAP → embedding

    The 5 layers are: stem + 4 × (ConvBlock + IdentityBlock) pairs.
    """

    def __init__(self, channels=[32, 64, 128, 128]):
        super().__init__()

        # Stem convolution
        self.stem = nn.Sequential(
            nn.Conv1d(1, channels[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        # ResNet stages: 4 × (ConvBlock + IdentityBlock)
        layers = []
        in_ch = channels[0]
        for i, out_ch in enumerate(channels):
            stride = 2 if i > 0 else 1
            layers.append(ConvBlock1D(in_ch, out_ch, stride=stride))
            layers.append(IdentityBlock1D(out_ch))
            in_ch = out_ch
        self.stages = nn.Sequential(*layers)

        # Global average pooling
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.out_features = channels[-1]

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len) — individual segmented beats/pulses

        Returns:
            (batch, out_features) — embedding vector
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, seq_len)

        # Per-beat Z-score normalization for amplitude-invariant morphology extraction
        x_mean = x.mean(dim=-1, keepdim=True)
        x_std = x.std(dim=-1, keepdim=True).clamp(min=1e-6)
        x = (x - x_mean) / x_std

        x = self.stem(x)
        x = self.stages(x)
        x = self.gap(x).squeeze(-1)  # (batch, out_features)
        return x


# ============================================================
# Dual-Branch 1D-ResNet + DNN Compression
# ============================================================

class ResNet1D(nn.Module):
    """
    Dual-branch 1D-ResNet with DNN Feature Compression.

    Branch ECG: ResNet1DBranch → embedding_ecg
    Branch PPG: ResNet1DBranch → embedding_ppg
    Concatenate → DNN Compression → F_morphological (40 features)

    DNN:
        Linear(2*128, 512) → Dropout(0.3) → Linear(512, 128) → Dropout(0.2) → Linear(128, 40)
    """

    def __init__(self, output_features=40,
                 dnn_hidden_1=512, dnn_dropout_1=0.3,
                 dnn_hidden_2=128, dnn_dropout_2=0.2):
        super().__init__()

        self.ecg_branch = ResNet1DBranch(channels=[32, 64, 128, 128])
        self.ppg_branch = ResNet1DBranch(channels=[32, 64, 128, 128])

        concat_dim = self.ecg_branch.out_features + self.ppg_branch.out_features

        # DNN Feature Compression with LayerNorm and GELU
        self.dnn = nn.Sequential(
            nn.LayerNorm(concat_dim),
            nn.Linear(concat_dim, dnn_hidden_1),
            nn.LayerNorm(dnn_hidden_1),
            nn.GELU(),
            nn.Dropout(dnn_dropout_1),
            nn.Linear(dnn_hidden_1, dnn_hidden_2),
            nn.LayerNorm(dnn_hidden_2),
            nn.GELU(),
            nn.Dropout(dnn_dropout_2),
            nn.Linear(dnn_hidden_2, output_features),
        )

        self.output_features = output_features

    def forward(self, ecg_beats, ppg_pulses):
        """
        Args:
            ecg_beats: (batch, ecg_beat_len) — individual ECG beats
            ppg_pulses: (batch, ppg_pulse_len) — individual PPG pulses

        Returns:
            (batch, 40) — F_morphological features
        """
        ecg_emb = self.ecg_branch(ecg_beats)   # (batch, 128)
        ppg_emb = self.ppg_branch(ppg_pulses)   # (batch, 128)

        combined = torch.cat([ecg_emb, ppg_emb], dim=1)  # (batch, 256)
        features = self.dnn(combined)  # (batch, 40)
        return features

    def extract_ecg_embedding(self, ecg_beats):
        """Extract only ECG branch embedding."""
        return self.ecg_branch(ecg_beats)

    def extract_ppg_embedding(self, ppg_pulses):
        """Extract only PPG branch embedding."""
        return self.ppg_branch(ppg_pulses)


# ============================================================
# Quick Test
# ============================================================

if __name__ == '__main__':
    print("--- 1D-ResNet Module Test ---\n")

    batch_size = 4

    # Test single branch
    branch = ResNet1DBranch()
    ecg_input = torch.randn(batch_size, 750)   # 3-sec ECG beat @ 250Hz
    ppg_input = torch.randn(batch_size, 1920)  # 30-sec PPG pulse @ 64Hz

    ecg_emb = branch(ecg_input)
    print(f"ECG branch: input {ecg_input.shape} → embedding {ecg_emb.shape}")

    ppg_branch = ResNet1DBranch()
    ppg_emb = ppg_branch(ppg_input)
    print(f"PPG branch: input {ppg_input.shape} → embedding {ppg_emb.shape}")

    # Test dual-branch with DNN compression
    model = ResNet1D(output_features=40)
    morph_features = model(ecg_input, ppg_input)
    print(f"\nResNet1D full: ECG {ecg_input.shape} + PPG {ppg_input.shape} → F_morph {morph_features.shape}")

    # Parameter count
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {params:,}")

    print("\nAll 1D-ResNet tests passed!")
