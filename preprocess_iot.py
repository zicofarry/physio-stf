"""
ST-MGM: IoT Data Adapter
===========================
Adapts data from the IoT prototype (IOT-prototype-deteksi-nyeri) to be
compatible with the ST-MGM pipeline.

IoT device records all signals at 250Hz via ESP32 firmware.
This script resamples to target rates:
    - ECG: 250Hz → 250Hz (keep as-is, already 75,000 samples for 5 min)
    - BVP (PPG): 250Hz → 64Hz (resample to 19,200 samples)
    - EDA: 250Hz → 4Hz (resample to 1,200 samples)

Then applies the same preprocessing as preprocess_raw.py:
    - ECG: Bandpass filter 0.5-40Hz
    - PPG: Bandpass filter 0.5-8Hz
    - EDA: Z-score normalization

Input format (IoT CSV):
    ECG.csv: Time,EcgWaveform (header + data rows, ~50Hz from Zephyr-style timestamps)
    BVP.csv: unix_timestamp (line 1), sampling_rate (line 2), data values (line 3+)
    EDA.csv: unix_timestamp (line 1), sampling_rate (line 2), data values (line 3+)

Usage:
    python preprocess_iot.py --iot_path D:/zicofarry/GitHub/IOT-prototype-deteksi-nyeri/data-capture/sample-1
    python preprocess_iot.py --iot_path ./iot_data --glucose_value 85
"""

import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
from scipy.signal import resample

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from libs.dsp import filter_ecg, filter_ppg, normalize_eda, resize_signal


# ============================================================
# Signal Specifications
# ============================================================

TARGET_SPECS = {
    'ecg': {'fs_target': 250, 'duration_sec': 300, 'num_samples': 75000},
    'ppg': {'fs_target': 64,  'duration_sec': 300, 'num_samples': 19200},
    'eda': {'fs_target': 4,   'duration_sec': 300, 'num_samples': 1200},
}


# ============================================================
# IoT Data Loaders
# ============================================================

def load_iot_ecg(csv_path):
    """
    Load ECG from IoT device.
    Format: Time,EcgWaveform (with timestamps)
    The IoT ECG is from AD8232 sensor, logged via ESP32.
    """
    df = pd.read_csv(csv_path)

    if 'EcgWaveform' in df.columns:
        signal = df['EcgWaveform'].values.astype(np.float32)
    else:
        # Fallback: try first numeric column
        signal = df.iloc[:, -1].values.astype(np.float32)

    # Estimate sampling rate from timestamps
    if 'Time' in df.columns:
        try:
            times = pd.to_datetime(df['Time'], format='%d/%m/%Y %H:%M:%S.%f')
            dt = times.diff().median().total_seconds()
            fs_estimated = 1.0 / dt if dt > 0 else 250.0
        except Exception:
            fs_estimated = 250.0
    else:
        fs_estimated = 250.0

    print(f"  ECG: {len(signal)} samples, estimated fs={fs_estimated:.1f}Hz")
    return signal, fs_estimated


def load_iot_e4_format(csv_path, signal_name='BVP'):
    """
    Load signal in E4-style format:
        Line 1: unix timestamp
        Line 2: sampling rate
        Line 3+: data values
    """
    df = pd.read_csv(csv_path, header=None)
    unix_start = df.loc[0][0]
    sampling_rate = df.loc[1][0]
    data = df.iloc[2:].to_numpy()[:, 0].astype(np.float32)

    print(f"  {signal_name}: {len(data)} samples @ {sampling_rate}Hz")
    return data, float(sampling_rate)


# ============================================================
# Resampling
# ============================================================

def resample_signal(signal, fs_source, fs_target, target_samples):
    """
    Resample a signal from source rate to target rate.

    Args:
        signal: 1D numpy array
        fs_source: Source sampling rate (Hz)
        fs_target: Target sampling rate (Hz)
        target_samples: Exact number of target samples

    Returns:
        Resampled signal of exactly target_samples length
    """
    if fs_source == fs_target and len(signal) == target_samples:
        return signal

    # Calculate how many samples to resample to
    duration = len(signal) / fs_source
    n_target = int(duration * fs_target)

    if n_target < 1:
        return np.zeros(target_samples, dtype=np.float32)

    resampled = resample(signal, n_target).astype(np.float32)
    # Resize to exact target length
    return resize_signal(resampled, target_samples)


# ============================================================
# Main Pipeline
# ============================================================

def process_iot_data(iot_path, out_path, glucose_value=None, segment_idx=0):
    """
    Process IoT data and save as ST-MGM compatible segment.

    Args:
        iot_path: Path to IoT data folder (containing ECG.csv, BVP.csv, EDA.csv)
        out_path: Output pickle file path
        glucose_value: Optional glucose value (mg/dL). If None, defaults to 100.
        segment_idx: Segment identifier
    """
    print(f"\n{'='*60}")
    print(f"  ST-MGM IoT Adapter")
    print(f"  Input: {iot_path}")
    print(f"{'='*60}")

    if glucose_value is None:
        glucose_value = 100.0
        print(f"\n  WARNING: No glucose value provided, using default: {glucose_value} mg/dL")

    # ===== Load signals =====
    print("\n--- Loading IoT signals ---")

    ecg_raw, fs_ecg = None, 250.0
    ppg_raw, fs_ppg = None, 250.0
    eda_raw, fs_eda = None, 250.0

    ecg_path = os.path.join(iot_path, 'ECG.csv')
    if os.path.exists(ecg_path):
        ecg_raw, fs_ecg = load_iot_ecg(ecg_path)

    bvp_path = os.path.join(iot_path, 'BVP.csv')
    if os.path.exists(bvp_path):
        ppg_raw, fs_ppg = load_iot_e4_format(bvp_path, 'BVP/PPG')

    eda_path = os.path.join(iot_path, 'EDA.csv')
    if os.path.exists(eda_path):
        eda_raw, fs_eda = load_iot_e4_format(eda_path, 'EDA')

    # ===== Resample to target rates =====
    print("\n--- Resampling ---")

    ecg_resampled = None
    if ecg_raw is not None:
        spec = TARGET_SPECS['ecg']
        ecg_resampled = resample_signal(ecg_raw, fs_ecg, spec['fs_target'], spec['num_samples'])
        print(f"  ECG: {len(ecg_raw)} samples @ {fs_ecg}Hz → {len(ecg_resampled)} samples @ {spec['fs_target']}Hz")

    ppg_resampled = None
    if ppg_raw is not None:
        spec = TARGET_SPECS['ppg']
        ppg_resampled = resample_signal(ppg_raw, fs_ppg, spec['fs_target'], spec['num_samples'])
        print(f"  PPG: {len(ppg_raw)} samples @ {fs_ppg}Hz → {len(ppg_resampled)} samples @ {spec['fs_target']}Hz")

    eda_resampled = None
    if eda_raw is not None:
        spec = TARGET_SPECS['eda']
        eda_resampled = resample_signal(eda_raw, fs_eda, spec['fs_target'], spec['num_samples'])
        print(f"  EDA: {len(eda_raw)} samples @ {fs_eda}Hz → {len(eda_resampled)} samples @ {spec['fs_target']}Hz")

    # ===== Apply preprocessing =====
    print("\n--- Applying filters ---")

    ecg_filtered = None
    if ecg_resampled is not None:
        ecg_filtered = filter_ecg(ecg_resampled, fs=250)
        print(f"  ECG: Bandpass filter 0.5-40Hz applied")

    ppg_filtered = None
    if ppg_resampled is not None:
        ppg_filtered = filter_ppg(ppg_resampled, fs=64)
        print(f"  PPG: Bandpass filter 0.5-8Hz applied")

    eda_normalized = None
    if eda_resampled is not None:
        eda_normalized = normalize_eda(eda_resampled)
        print(f"  EDA: Z-score normalization applied")

    # ===== Save segment =====
    segment = {
        'Index': segment_idx,
        'Timestamp': pd.Timestamp.now(),
        'glucose': float(glucose_value),
        'ecg_raw': ecg_resampled,
        'ppg_raw': ppg_resampled,
        'eda_raw': eda_resampled,
        'ecg_filtered': ecg_filtered,
        'ppg_filtered': ppg_filtered,
        'eda_normalized': eda_normalized,
        'source': 'iot',
        'iot_path': iot_path,
    }

    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else '.', exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump(segment, f)

    print(f"\n--- Summary ---")
    print(f"  ECG: {'✓' if ecg_filtered is not None else '✗'} ({len(ecg_filtered) if ecg_filtered is not None else 0} samples)")
    print(f"  PPG: {'✓' if ppg_filtered is not None else '✗'} ({len(ppg_filtered) if ppg_filtered is not None else 0} samples)")
    print(f"  EDA: {'✓' if eda_normalized is not None else '✗'} ({len(eda_normalized) if eda_normalized is not None else 0} samples)")
    print(f"  Glucose: {glucose_value} mg/dL")
    print(f"  Saved to: {out_path}")
    print("\nDone!")


# ============================================================
# CLI
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ST-MGM: IoT Data Adapter')
    parser.add_argument('--iot_path', type=str, required=True,
                        help='Path to IoT data folder (containing ECG.csv, BVP.csv, EDA.csv)')
    parser.add_argument('--out_path', type=str, default=None,
                        help='Output pickle path (default: ./dataset/processed/iot/0.pkl)')
    parser.add_argument('--glucose_value', type=float, default=None,
                        help='Glucose value (mg/dL) for this recording')
    parser.add_argument('--segment_idx', type=int, default=0,
                        help='Segment index identifier')
    args = parser.parse_args()

    if args.out_path is None:
        args.out_path = os.path.join('dataset', 'processed', 'iot', f'{args.segment_idx}.pkl')

    process_iot_data(args.iot_path, args.out_path, args.glucose_value, args.segment_idx)
