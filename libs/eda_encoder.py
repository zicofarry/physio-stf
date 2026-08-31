"""
ST-MGM: EDA Encoder Module
=============================
Conv1D (2-layer) + Bi-LSTM for EDA (Electrodermal Activity) features.

Architecture:
    Input: Z-score normalized EDA (1200 samples @ 4Hz, 1 channel)
    Conv1D(1→8, k=12, s=6) → Conv1D(8→16, k=8, s=5) → Bi-LSTM(16, hidden=32)
    Output: F_eda (64 features from bidirectional hidden state)

Phase 2 of the ST-MGM pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EDAEncoder(nn.Module):
    """
    Conv1D + Bi-LSTM encoder for EDA signal.

    Input:  (batch, 1200) — Z-score normalized EDA @ 4Hz, 5 min
    Output: (batch, 64)   — F_eda features (bidirectional hidden state)

    Conv1D Encoder:
        Layer 1: Conv1d(in=1, out=8, kernel=12, stride=6)
        Layer 2: Conv1d(in=8, out=16, kernel=8, stride=5)

    Bi-LSTM:
        input_size=16, hidden_size=32, bidirectional=True
        → outputs 64 features (32 forward + 32 backward)
    """

    def __init__(self,
                 conv1_out=8, conv1_kernel=12, conv1_stride=6,
                 conv2_out=16, conv2_kernel=8, conv2_stride=5,
                 lstm_hidden=32, output_features=64):
        super().__init__()

        self.output_features = output_features

        # Conv1D Encoder
        self.conv1 = nn.Conv1d(1, conv1_out, kernel_size=conv1_kernel, stride=conv1_stride)
        self.bn1 = nn.BatchNorm1d(conv1_out)

        self.conv2 = nn.Conv1d(conv1_out, conv2_out, kernel_size=conv2_kernel, stride=conv2_stride)
        self.bn2 = nn.BatchNorm1d(conv2_out)

        # Bi-LSTM
        self.lstm = nn.LSTM(
            input_size=conv2_out,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # Output dimension = 2 * lstm_hidden = 64
        assert output_features == 2 * lstm_hidden, \
            f"output_features ({output_features}) must be 2 * lstm_hidden ({lstm_hidden})"

    def forward(self, x):
        """
        Args:
            x: (batch, 1200) — single-channel EDA signal

        Returns:
            (batch, 64) — F_eda features
        """
        # Reshape to (batch, channels=1, seq_len=1200)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # Conv1D encoder
        x = F.relu(self.bn1(self.conv1(x)))  # (batch, 8, ~198)
        x = F.relu(self.bn2(self.conv2(x)))  # (batch, 16, ~38)

        # Reshape for LSTM: (batch, seq_len, features)
        x = x.transpose(1, 2)  # (batch, ~38, 16)

        # Bi-LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out: (batch, ~38, 64)

        # Use last timestep output (contains full sequence information)
        x = lstm_out[:, -1, :]  # (batch, 64)

        return x

    def get_conv_output_length(self, input_length=1200):
        """Calculate the output sequence length after Conv1D layers."""
        # After conv1: (L - K + 2P) / S + 1, padding=0
        l1 = (input_length - self.conv1.kernel_size[0]) // self.conv1.stride[0] + 1
        # After conv2
        l2 = (l1 - self.conv2.kernel_size[0]) // self.conv2.stride[0] + 1
        return l2


# ============================================================
# Quick Test
# ============================================================

if __name__ == '__main__':
    print("--- EDA Encoder Module Test ---\n")

    batch_size = 4

    model = EDAEncoder()

    # Test with standard EDA input
    eda_input = torch.randn(batch_size, 1200)
    eda_features = model(eda_input)
    print(f"EDA input: {eda_input.shape} → F_eda: {eda_features.shape}")

    # Show intermediate sizes
    conv_out_len = model.get_conv_output_length(1200)
    print(f"Conv output sequence length: {conv_out_len}")

    # Parameter count
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {params:,}")

    # Verify output dimension
    assert eda_features.shape == (batch_size, 64), \
        f"Expected (4, 64), got {eda_features.shape}"
    print("\nAll EDA Encoder tests passed!")
