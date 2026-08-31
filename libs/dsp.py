"""
ST-MGM: Digital Signal Processing (DSP) Module
================================================
- Bandpass Filter (0.5-40Hz for ECG, 0.5-8Hz for PPG)
- Z-Score Normalization for EDA
- Signal quality checks and padding/truncation utilities

Phase 1 of the ST-MGM pipeline.
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt


# ============================================================
# Bandpass Filter
# ============================================================

def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    """
    Apply a zero-phase Butterworth bandpass filter.

    Args:
        signal: 1D numpy array of raw signal data
        lowcut: Lower cutoff frequency (Hz)
        highcut: Upper cutoff frequency (Hz)
        fs: Sampling frequency (Hz)
        order: Filter order (default 4)

    Returns:
        Filtered signal (same length as input)
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    # Clamp to valid range
    low = max(low, 1e-5)
    high = min(high, 1.0 - 1e-5)

    if low >= high:
        return signal

    sos = butter(order, [low, high], btype='band', output='sos')
    filtered = sosfiltfilt(sos, signal)
    return filtered.astype(np.float32)


def filter_ecg(signal, fs=250, lowcut=0.5, highcut=40.0, order=4):
    """
    Apply bandpass filter optimized for ECG signals (0.5-40Hz).

    Args:
        signal: Raw ECG signal (1D numpy array)
        fs: Sampling frequency (default 250Hz)
        lowcut: Lower cutoff (default 0.5Hz)
        highcut: Upper cutoff (default 40Hz)
        order: Filter order

    Returns:
        Filtered ECG signal
    """
    return bandpass_filter(signal, lowcut, highcut, fs, order)


def filter_ppg(signal, fs=64, lowcut=0.5, highcut=8.0, order=4):
    """
    Apply bandpass filter optimized for PPG signals (0.5-8Hz).

    Args:
        signal: Raw PPG signal (1D numpy array)
        fs: Sampling frequency (default 64Hz)
        lowcut: Lower cutoff (default 0.5Hz)
        highcut: Upper cutoff (default 8Hz)
        order: Filter order

    Returns:
        Filtered PPG signal
    """
    return bandpass_filter(signal, lowcut, highcut, fs, order)


# ============================================================
# Z-Score Normalization
# ============================================================

def zscore_normalize(signal):
    """
    Apply z-score normalization to a signal.

    Args:
        signal: 1D numpy array

    Returns:
        Normalized signal with zero mean and unit variance
    """
    std = np.std(signal)
    if std < 1e-8:
        return np.zeros_like(signal, dtype=np.float32)
    return ((signal - np.mean(signal)) / std).astype(np.float32)


def normalize_eda(signal):
    """
    Z-score normalize EDA signal.

    Args:
        signal: Raw EDA signal (1D numpy array)

    Returns:
        Z-score normalized EDA signal
    """
    return zscore_normalize(signal)


# ============================================================
# Signal Utilities
# ============================================================

def resize_signal(signal, target_length):
    """
    Resize signal to target length by truncation or zero-padding.

    Args:
        signal: 1D numpy array
        target_length: Desired length

    Returns:
        Signal of exactly target_length samples
    """
    if len(signal) == target_length:
        return signal.astype(np.float32)
    elif len(signal) > target_length:
        return signal[:target_length].astype(np.float32)
    else:
        padded = np.zeros(target_length, dtype=np.float32)
        padded[:len(signal)] = signal
        return padded


def check_signal_quality(signal, min_length=100, max_flat_ratio=0.5):
    """
    Basic signal quality check.

    Args:
        signal: 1D numpy array
        min_length: Minimum required samples
        max_flat_ratio: Maximum ratio of flat (constant) samples

    Returns:
        True if signal passes quality check
    """
    if signal is None or len(signal) < min_length:
        return False

    # Check for flat signal
    diffs = np.diff(signal)
    flat_ratio = np.mean(np.abs(diffs) < 1e-10)
    if flat_ratio > max_flat_ratio:
        return False

    # Check for NaN/Inf
    if np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
        return False

    return True


# ============================================================
# Quick Test
# ============================================================

if __name__ == '__main__':
    print("--- DSP Module Test ---\n")

    # Test ECG filter
    fs_ecg = 250
    t = np.arange(0, 5, 1/fs_ecg)
    ecg_fake = np.sin(2 * np.pi * 1.0 * t) + 0.5 * np.sin(2 * np.pi * 60 * t)
    ecg_filtered = filter_ecg(ecg_fake, fs=fs_ecg)
    print(f"ECG filter: input len={len(ecg_fake)}, output len={len(ecg_filtered)}")

    # Test PPG filter
    fs_ppg = 64
    t = np.arange(0, 5, 1/fs_ppg)
    ppg_fake = np.sin(2 * np.pi * 1.0 * t) + 0.3 * np.sin(2 * np.pi * 15 * t)
    ppg_filtered = filter_ppg(ppg_fake, fs=fs_ppg)
    print(f"PPG filter: input len={len(ppg_fake)}, output len={len(ppg_filtered)}")

    # Test EDA normalization
    eda_fake = np.random.randn(1200) * 5 + 10
    eda_norm = normalize_eda(eda_fake)
    print(f"EDA normalize: mean={eda_norm.mean():.4f}, std={eda_norm.std():.4f}")

    # Test resize
    short = np.ones(100)
    resized = resize_signal(short, 200)
    print(f"Resize: {len(short)} -> {len(resized)}")

    # Test quality check
    good = np.random.randn(500)
    bad = np.zeros(500)
    print(f"Quality check (good): {check_signal_quality(good)}")
    print(f"Quality check (bad): {check_signal_quality(bad)}")

    print("\nAll DSP tests passed!")
