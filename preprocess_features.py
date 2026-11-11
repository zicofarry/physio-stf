"""
ST-MGM: Phase 3 — Feature Extraction
=======================================
Reads processed .pkl segments and extracts features for ST-MGM:

A. Temporal Statistical (F_temporal):
    ECG + PPG → DWT (db4, 7-level) → 10 stats × 8 sub-bands × 2 = 160 features
    → Feature Selection (UFS + L1-FS + RFE) → ~18 features

B. Spatial Morphological (F_morphological):
    Individual ECG beats + PPG pulses → to be processed by 1D-ResNet during training

C. EDA Features (F_eda):
    Z-score normalized EDA → to be processed by EDA Encoder during training

Also generates metadata with stratified k-fold CV splits.

Usage:
    python preprocess_features.py --subject c1s01_raw_single
"""

import os
import sys
import glob
import json
import pickle
import argparse
import numpy as np
import pandas as pd

try:
    import neurokit2 as nk
    HAS_NK = True
except ImportError:
    print("WARNING: neurokit2 not installed. ECG R-peak detection will use fallback.")
    HAS_NK = False

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from libs.dsp import resize_signal
from libs.dwt_features import extract_temporal_features, combined_feature_selection


# ============================================================
# ECG Beat Segmentation
# ============================================================

def segment_ecg_beats(ecg_signal, fs=250, beat_duration_sec=3):
    """
    Segment individual ECG beats centered on R-peaks.

    Args:
        ecg_signal: Filtered ECG signal (75,000 samples)
        fs: Sampling rate
        beat_duration_sec: Duration of each beat window (seconds)

    Returns:
        List of numpy arrays, each of shape (beat_samples,)
    """
    beat_samples = int(fs * beat_duration_sec)  # 750
    half_window = beat_samples // 2
    beats = []

    if HAS_NK:
        try:
            ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=fs)
            _, rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=fs)
            rpeak_indices = rpeaks['ECG_R_Peaks']
        except Exception:
            rpeak_indices = []
    else:
        # Simple threshold-based R-peak detection fallback
        rpeak_indices = _simple_rpeak_detect(ecg_signal, fs)

    for rpeak in rpeak_indices:
        start = rpeak - half_window
        end = rpeak + half_window
        if start >= 0 and end <= len(ecg_signal):
            beat = ecg_signal[start:end]
            if len(beat) == beat_samples:
                beats.append(beat.astype(np.float32))

    return beats


def _simple_rpeak_detect(ecg_signal, fs=250):
    """Simple R-peak detection fallback (threshold-based)."""
    from scipy.signal import find_peaks

    # Basic preprocessing
    signal = ecg_signal.copy()
    signal = signal - np.mean(signal)
    signal = np.abs(signal)

    # Find peaks with minimum distance (assume HR 40-200 bpm)
    min_distance = int(fs * 60 / 200)  # Max 200 bpm
    height = np.percentile(signal, 80)

    peaks, _ = find_peaks(signal, height=height, distance=min_distance)
    return peaks


# ============================================================
# PPG Pulse Segmentation
# ============================================================

def segment_ppg_pulses(ppg_signal, fs=64, pulse_duration_sec=3):
    """
    Segment individual PPG pulses.

    Args:
        ppg_signal: Filtered PPG signal (19,200 samples)
        fs: Sampling rate
        pulse_duration_sec: Duration of each pulse window (seconds)

    Returns:
        List of numpy arrays, each of shape (pulse_samples,)
    """
    pulse_samples = int(fs * pulse_duration_sec)  # 192
    half_window = pulse_samples // 2
    pulses = []

    if HAS_NK:
        try:
            ppg_cleaned = nk.ppg_clean(ppg_signal, sampling_rate=fs)
            ppg_info = nk.ppg_findpeaks(ppg_cleaned, sampling_rate=fs)
            peak_indices = ppg_info['PPG_Peaks']
        except Exception:
            peak_indices = _simple_ppg_peak_detect(ppg_signal, fs)
    else:
        peak_indices = _simple_ppg_peak_detect(ppg_signal, fs)

    for peak in peak_indices:
        start = peak - half_window
        end = peak + half_window
        if start >= 0 and end <= len(ppg_signal):
            pulse = ppg_signal[start:end]
            if len(pulse) == pulse_samples:
                pulses.append(pulse.astype(np.float32))

    return pulses


def _simple_ppg_peak_detect(ppg_signal, fs=64):
    """Simple PPG peak detection fallback."""
    from scipy.signal import find_peaks

    signal = ppg_signal.copy()
    signal = signal - np.mean(signal)

    min_distance = int(fs * 60 / 200)
    height = np.percentile(signal, 70)

    peaks, _ = find_peaks(signal, height=height, distance=min_distance)
    return peaks


def compute_clean_beat_template(beats, corr_threshold=0.5):
    """
    Compute a clean representative template beat/pulse from a 5-minute window
    using pointwise median after rejecting low-correlation artifact beats.
    """
    beats_arr = np.stack(beats, axis=0).astype(np.float32)
    if len(beats_arr) == 1:
        return beats_arr[0]

    median_template = np.median(beats_arr, axis=0)
    centered_beats = beats_arr - np.mean(beats_arr, axis=1, keepdims=True)
    centered_med = median_template - np.mean(median_template)
    med_norm = np.linalg.norm(centered_med)

    if med_norm > 1e-8:
        beat_norms = np.linalg.norm(centered_beats, axis=1)
        valid_norms = beat_norms > 1e-8
        corrs = np.zeros(len(beats_arr), dtype=np.float32)
        corrs[valid_norms] = (
            (centered_beats[valid_norms] @ centered_med)
            / (beat_norms[valid_norms] * med_norm)
        )
        clean_mask = corrs >= corr_threshold
        min_keep = max(1, min(3, len(beats_arr)))
        if np.sum(clean_mask) >= min_keep:
            clean_beats = beats_arr[clean_mask]
        else:
            top_k = max(1, len(beats_arr) // 2)
            top_idx = np.argsort(-corrs)[:top_k]
            clean_beats = beats_arr[top_idx]
        return np.median(clean_beats, axis=0).astype(np.float32)

    return median_template.astype(np.float32)


# ============================================================
# Main Pipeline
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ST-MGM: Feature Extraction')
    parser.add_argument('--subject', type=str, required=True, help='Subject ID')
    parser.add_argument('--processed_dir', default='./dataset/processed', type=str)
    parser.add_argument('--out_dir', default='./data', type=str)
    parser.add_argument('--n_folds', default=5, type=int, help='Number of CV folds')
    parser.add_argument('--hypo_threshold', default=70, type=float, help='Hypo threshold mg/dL')
    args = parser.parse_args()

    subject_id = args.subject
    processed_dir = os.path.join(args.processed_dir, subject_id)
    out_dir = os.path.join(args.out_dir, subject_id)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  ST-MGM: Feature Extraction for {subject_id}")
    print(f"{'='*60}")

    # Load all processed segments
    pkl_files = sorted(
        glob.glob(os.path.join(processed_dir, '*.pkl')),
        key=lambda x: int(os.path.basename(x).replace('.pkl', ''))
    )
    print(f"\nFound {len(pkl_files)} processed segments")

    if not pkl_files:
        print("ERROR: No processed segments found! Run preprocess_raw.py first.")
        sys.exit(1)

    # ===== Extract features =====
    all_segments = []  # For the dataset
    all_temporal_raw = []  # 160-dim DWT features
    glucose_list = []
    valid_count = 0
    skip_count = 0

    for pkl_path in tqdm(pkl_files, desc="Extracting features"):
        with open(pkl_path, 'rb') as f:
            segment = pickle.load(f)

        ecg_filtered = segment.get('ecg_filtered')
        ppg_filtered = segment.get('ppg_filtered')
        eda_normalized = segment.get('eda_normalized')
        glucose = segment['glucose']
        seg_idx = segment['Index']

        # Skip if missing any modality
        if ecg_filtered is None or ppg_filtered is None or eda_normalized is None:
            skip_count += 1
            continue

        # A. Extract temporal features (DWT)
        temporal_feat = extract_temporal_features(
            ecg_filtered, ppg_filtered,
            fs_ecg=250, fs_ppg=64
        )
        all_temporal_raw.append(temporal_feat)

        # B. Segment individual ECG beats
        ecg_beats = segment_ecg_beats(ecg_filtered, fs=250)
        if len(ecg_beats) == 0:
            # Fallback: use fixed-length windows
            beat_len = 750
            ecg_beats = []
            for i in range(0, len(ecg_filtered) - beat_len, beat_len):
                ecg_beats.append(ecg_filtered[i:i+beat_len].astype(np.float32))
            if len(ecg_beats) == 0:
                skip_count += 1
                continue

        # C. Segment individual PPG pulses
        ppg_pulses = segment_ppg_pulses(ppg_filtered, fs=64)
        if len(ppg_pulses) == 0:
            # Fallback: use fixed-length windows
            pulse_len = 192
            ppg_pulses = []
            for i in range(0, len(ppg_filtered) - pulse_len, pulse_len):
                ppg_pulses.append(ppg_filtered[i:i+pulse_len].astype(np.float32))
            if len(ppg_pulses) == 0:
                skip_count += 1
                continue

        # Compute clean median template beat/pulse across the 5-minute window
        representative_ecg = compute_clean_beat_template(ecg_beats)
        representative_ppg = compute_clean_beat_template(ppg_pulses)

        all_segments.append({
            'ecg_beat': representative_ecg,
            'ppg_pulse': representative_ppg,
            'eda': eda_normalized,
            'glucose': glucose,
            'segment_idx': seg_idx,
            'num_ecg_beats': len(ecg_beats),
            'num_ppg_pulses': len(ppg_pulses),
        })
        glucose_list.append(glucose)
        valid_count += 1

    print(f"\n--- Feature Extraction Summary ---")
    print(f"Valid segments (all 3 modalities): {valid_count}")
    print(f"Skipped segments: {skip_count}")

    if valid_count == 0:
        print("ERROR: No valid segments with all modalities!")
        sys.exit(1)

    temporal_matrix = np.array(all_temporal_raw, dtype=np.float32)
    glucose_array = np.array(glucose_list, dtype=np.float32)
    labels_cls = (glucose_array < args.hypo_threshold).astype(int)
    print(f"Raw temporal features: {temporal_matrix.shape}")

    # ===== Generate k-fold CV metadata & Per-Fold Feature Selection =====
    print(f"\n--- Generating {args.n_folds}-fold CV metadata & Per-Fold Feature Selection ---")
    from sklearn.model_selection import StratifiedKFold, KFold

    valid_indices = list(range(len(all_segments)))
    labels = labels_cls

    hypo_count = int(labels.sum())
    normal_count = len(labels) - hypo_count
    print(f"Segments: {len(labels)} (hypo={hypo_count}, normal={normal_count})")

    metadata = {'folds': {}}
    indices_array = np.array(valid_indices)
    fold_selected_indices = {}

    if len(labels) < 2:
        print("WARNING: Not enough samples for cross-validation. Creating a single fallback fold.")
        splits = [(np.array([0]), np.array([0]))]
    else:
        unique_classes = len(np.unique(labels))
        min_class = min(hypo_count, normal_count)
        if unique_classes > 1 and min_class >= 2:
            n_folds = max(2, min(args.n_folds, int(min_class)))
            try:
                splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
                splits = list(splitter.split(indices_array, labels))
                print(f"Using StratifiedKFold with {n_folds} folds.")
            except ValueError:
                n_folds = max(2, min(args.n_folds, len(labels)))
                splitter = KFold(n_splits=n_folds, shuffle=True, random_state=42)
                splits = list(splitter.split(indices_array, labels))
                print(f"Fallback to KFold with {n_folds} folds.")
        else:
            n_folds = max(2, min(args.n_folds, len(labels)))
            splitter = KFold(n_splits=n_folds, shuffle=True, random_state=42)
            splits = list(splitter.split(indices_array, labels))
            print(f"WARNING: Insufficient minority samples for stratification. Using KFold with {n_folds} folds.")

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        version = f"v{fold_idx + 1}"
        train_indices = [int(indices_array[i]) for i in train_idx]
        val_indices = [int(indices_array[i]) for i in val_idx]

        try:
            sel_idx, _ = combined_feature_selection(
                temporal_matrix[train_indices],
                labels_cls[train_indices],
                glucose_array[train_indices],
                n_final=18,
            )
        except Exception as e:
            print(f"  WARNING ({version}): Feature selection failed ({e}), using first 18 features")
            sel_idx = np.arange(min(18, temporal_matrix.shape[1]), dtype=int)

        fold_selected_indices[version] = [int(x) for x in sel_idx]

        metadata['folds'][version] = {
            'train': train_indices,
            'val': val_indices,
            'selected_indices': fold_selected_indices[version],
        }
        train_hypo = sum(1 for i in train_idx if labels[i] == 1)
        val_hypo = sum(1 for i in val_idx if labels[i] == 1)
        print(f"  {version}: train={len(train_indices)} (hypo={train_hypo}), "
              f"val={len(val_indices)} (hypo={val_hypo}), "
              f"selected_features={len(sel_idx)}")

    default_selected = np.array(
        fold_selected_indices.get('v1', list(range(min(18, temporal_matrix.shape[1])))),
        dtype=int,
    )
    temporal_selected = temporal_matrix[:, default_selected]

    # ===== Save features =====
    print("\n--- Saving features ---")

    segments_path = os.path.join(out_dir, 'segments.pkl')
    with open(segments_path, 'wb') as f:
        pickle.dump(all_segments, f)
    print(f"Saved segments.pkl ({len(all_segments)} segments)")

    temporal_path = os.path.join(out_dir, 'temporal_features.pkl')
    with open(temporal_path, 'wb') as f:
        pickle.dump({
            'raw': temporal_matrix,
            'selected': temporal_selected,
            'selected_indices': default_selected,
            'fold_selected_indices': fold_selected_indices,
        }, f)
    print(f"Saved temporal_features.pkl (raw: {temporal_matrix.shape}, per-fold selected: {len(default_selected)})")

    metadata['signal_specs'] = {
        'ecg_beat_len': len(all_segments[0]['ecg_beat']),
        'ppg_pulse_len': len(all_segments[0]['ppg_pulse']),
        'eda_len': len(all_segments[0]['eda']),
        'temporal_dim': len(default_selected),
    }
    metadata['hypo_threshold'] = args.hypo_threshold
    metadata['selected_feature_indices'] = default_selected.tolist()
    metadata['fold_selected_indices'] = fold_selected_indices

    metadata_path = os.path.join(out_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"\nSaved metadata to {metadata_path}")
    print("\nFeature extraction complete!")
