"""
ST-MGM: PyTorch Dataset & DataLoader
=======================================
Dataset classes for the ST-MGM multi-task pipeline.

Loads pre-extracted features and provides:
    - F_temporal (DWT statistical features)
    - ECG beats (individual segmented beats)
    - PPG pulses (individual segmented pulses)
    - EDA signal (Z-score normalized 1200 samples)
    - Labels: hypo_label (0/1), glucose (mg/dL)

Phase 3 of the ST-MGM pipeline.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


# ============================================================
# Data Augmentation (for training)
# ============================================================

def augment_jitter(signal, sigma_ratio=0.01):
    """Add Gaussian noise proportional to signal std."""
    sigma = sigma_ratio * np.std(signal)
    return signal + np.random.normal(0, sigma, signal.shape).astype(np.float32)


def augment_scaling(signal, low=0.9, high=1.1):
    """Scale signal by a random factor."""
    factor = np.random.uniform(low, high)
    return (signal * factor).astype(np.float32)


def augment_time_shift(signal, max_shift_ratio=0.05):
    """Shift signal along time axis by a random amount."""
    max_shift = int(len(signal) * max_shift_ratio)
    if max_shift < 1:
        return signal
    shift = np.random.randint(-max_shift, max_shift + 1)
    return np.roll(signal, shift).astype(np.float32)


# ============================================================
# ST-MGM Multi-Modal Dataset
# ============================================================

class STMGMDataset(Dataset):
    """
    Multi-modal dataset for ST-MGM.

    Each sample provides all inputs needed for the fusion model:
        - f_temporal: (temporal_dim,) DWT features
        - ecg_beat:   (ecg_beat_len,) individual segmented beat
        - ppg_pulse:  (ppg_pulse_len,) individual segmented pulse
        - eda_signal: (1200,) Z-score normalized EDA
        - hypo_label: float (1.0 if glucose < 70, else 0.0)
        - glucose:    float (mg/dL)
        - segment_idx: int (segment identifier)
    """

    def __init__(self, segments, temporal_features=None, feature_indices=None,
                 augment=False, hypo_threshold=70, verbose=True):
        """
        Args:
            segments: list of dicts, each containing:
                - 'ecg_beat': numpy array of individual beat
                - 'ppg_pulse': numpy array of individual pulse
                - 'eda': numpy array of EDA signal (1200 samples)
                - 'glucose': float glucose value
                - 'segment_idx': int segment identifier
            temporal_features: numpy array (N, 160) of DWT features, or None
            feature_indices: numpy array of selected temporal feature indices
            augment: whether to apply data augmentation
            hypo_threshold: glucose threshold for hypoglycemia (default 70 mg/dL)
            verbose: print dataset info
        """
        self.segments = segments
        self.temporal_features = temporal_features
        self.feature_indices = feature_indices
        self.augment = augment
        self.hypo_threshold = hypo_threshold

        # Compute labels
        self.labels = np.array([
            1.0 if seg['glucose'] < hypo_threshold else 0.0
            for seg in segments
        ], dtype=np.float32)

        self.glucose_values = np.array([
            seg['glucose'] for seg in segments
        ], dtype=np.float32)

        # Class statistics
        hypo_count = int(self.labels.sum())
        normal_count = len(self.labels) - hypo_count
        self.normal_hypo_ratio = normal_count / max(hypo_count, 1)

        if verbose:
            print(f"  STMGMDataset: {len(self)} samples "
                  f"(hypo={hypo_count}, normal={normal_count}, "
                  f"ratio={self.normal_hypo_ratio:.1f})")

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        seg = self.segments[idx]

        # ECG beat
        ecg_beat = seg['ecg_beat'].copy().astype(np.float32)
        if self.augment:
            if np.random.random() < 0.5:
                ecg_beat = augment_jitter(ecg_beat)
            if np.random.random() < 0.3:
                ecg_beat = augment_scaling(ecg_beat)

        # PPG pulse
        ppg_pulse = seg['ppg_pulse'].copy().astype(np.float32)
        if self.augment:
            if np.random.random() < 0.5:
                ppg_pulse = augment_jitter(ppg_pulse)
            if np.random.random() < 0.3:
                ppg_pulse = augment_scaling(ppg_pulse)

        # EDA signal
        eda = seg['eda'].copy().astype(np.float32)

        # Temporal features
        if self.temporal_features is not None:
            if self.feature_indices is not None:
                f_temporal = self.temporal_features[idx, self.feature_indices].astype(np.float32)
            else:
                f_temporal = self.temporal_features[idx].astype(np.float32)
        else:
            f_temporal = np.zeros(18, dtype=np.float32)

        # Labels
        hypo_label = self.labels[idx]
        glucose = self.glucose_values[idx]
        segment_idx = seg.get('segment_idx', idx)

        return {
            'f_temporal': torch.tensor(f_temporal, dtype=torch.float32),
            'ecg_beat': torch.tensor(ecg_beat, dtype=torch.float32),
            'ppg_pulse': torch.tensor(ppg_pulse, dtype=torch.float32),
            'eda': torch.tensor(eda, dtype=torch.float32),
            'hypo_label': torch.tensor(hypo_label, dtype=torch.float32),
            'glucose': torch.tensor(glucose, dtype=torch.float32),
            'segment_idx': segment_idx,
        }


# ============================================================
# Simple Feature Dataset (for pre-extracted features only)
# ============================================================

class FeatureDataset(Dataset):
    """
    Simple dataset for pre-extracted features (fusion-only model).

    Each sample provides:
        - f_temporal: (temporal_dim,)
        - f_morph:    (40,)
        - f_eda:      (64,)
        - hypo_label: float
        - glucose:    float
    """

    def __init__(self, f_temporal, f_morph, f_eda, glucose_values,
                 hypo_threshold=70, verbose=True):
        """
        Args:
            f_temporal: (N, temporal_dim) numpy array
            f_morph: (N, 40) numpy array
            f_eda: (N, 64) numpy array
            glucose_values: (N,) numpy array
            hypo_threshold: glucose threshold
        """
        self.f_temporal = f_temporal.astype(np.float32)
        self.f_morph = f_morph.astype(np.float32)
        self.f_eda = f_eda.astype(np.float32)
        self.glucose = glucose_values.astype(np.float32)
        self.labels = (glucose_values < hypo_threshold).astype(np.float32)

        if verbose:
            hypo = int(self.labels.sum())
            normal = len(self.labels) - hypo
            print(f"  FeatureDataset: {len(self)} samples "
                  f"(hypo={hypo}, normal={normal})")

    def __len__(self):
        return len(self.glucose)

    def __getitem__(self, idx):
        return {
            'f_temporal': torch.tensor(self.f_temporal[idx]),
            'f_morph': torch.tensor(self.f_morph[idx]),
            'f_eda': torch.tensor(self.f_eda[idx]),
            'hypo_label': torch.tensor(self.labels[idx]),
            'glucose': torch.tensor(self.glucose[idx]),
        }


# ============================================================
# Collate function (for variable-length beats/pulses)
# ============================================================

def stmgm_collate_fn(batch):
    """
    Custom collate function that pads ECG beats and PPG pulses
    to the same length within a batch.
    """
    # Find max lengths in this batch
    max_ecg_len = max(item['ecg_beat'].shape[0] for item in batch)
    max_ppg_len = max(item['ppg_pulse'].shape[0] for item in batch)

    f_temporal = torch.stack([item['f_temporal'] for item in batch])
    hypo_labels = torch.stack([item['hypo_label'] for item in batch])
    glucose = torch.stack([item['glucose'] for item in batch])
    eda = torch.stack([item['eda'] for item in batch])
    segment_idx = [item['segment_idx'] for item in batch]

    # Pad ECG beats
    ecg_padded = torch.zeros(len(batch), max_ecg_len)
    for i, item in enumerate(batch):
        ecg_len = item['ecg_beat'].shape[0]
        ecg_padded[i, :ecg_len] = item['ecg_beat']

    # Pad PPG pulses
    ppg_padded = torch.zeros(len(batch), max_ppg_len)
    for i, item in enumerate(batch):
        ppg_len = item['ppg_pulse'].shape[0]
        ppg_padded[i, :ppg_len] = item['ppg_pulse']

    return {
        'f_temporal': f_temporal,
        'ecg_beat': ecg_padded,
        'ppg_pulse': ppg_padded,
        'eda': eda,
        'hypo_label': hypo_labels,
        'glucose': glucose,
        'segment_idx': segment_idx,
    }


# ============================================================
# Quick Test
# ============================================================

if __name__ == '__main__':
    from torch.utils.data import DataLoader

    print("--- DataLoader Module Test ---\n")

    # Create fake segments
    np.random.seed(42)
    N = 20
    segments = []
    for i in range(N):
        segments.append({
            'ecg_beat': np.random.randn(750).astype(np.float32),
            'ppg_pulse': np.random.randn(1920).astype(np.float32),
            'eda': np.random.randn(1200).astype(np.float32),
            'glucose': np.random.uniform(40, 200),
            'segment_idx': i,
        })

    temporal_features = np.random.randn(N, 160).astype(np.float32)
    feature_indices = np.arange(18)

    # Test STMGMDataset
    dataset = STMGMDataset(segments, temporal_features, feature_indices, augment=True)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=stmgm_collate_fn)

    for batch in loader:
        print(f"Batch shapes:")
        print(f"  f_temporal: {batch['f_temporal'].shape}")
        print(f"  ecg_beat:   {batch['ecg_beat'].shape}")
        print(f"  ppg_pulse:  {batch['ppg_pulse'].shape}")
        print(f"  eda:        {batch['eda'].shape}")
        print(f"  hypo_label: {batch['hypo_label'].shape}")
        print(f"  glucose:    {batch['glucose'].shape}")
        break

    # Test FeatureDataset
    print("\n--- FeatureDataset ---")
    feat_dataset = FeatureDataset(
        np.random.randn(N, 18).astype(np.float32),
        np.random.randn(N, 40).astype(np.float32),
        np.random.randn(N, 64).astype(np.float32),
        np.random.uniform(40, 200, N).astype(np.float32),
    )
    feat_loader = DataLoader(feat_dataset, batch_size=4)
    for batch in feat_loader:
        print(f"Batch shapes:")
        for k, v in batch.items():
            print(f"  {k}: {v.shape}")
        break

    print("\nAll DataLoader tests passed!")
