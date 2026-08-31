"""
ST-MGM: Phase 3 — Raw Data Preprocessing
==========================================
Reads raw sensor data from Zephyr (ECG) and E4 (BVP/PPG, EDA),
aligns them with CGM glucose readings, and saves per-CGM-segment .pkl files.

All signals are cut to uniform 5-minute (300-second) windows:
    - ECG: 5 min @ 250Hz = 75,000 samples → Bandpass 0.5-40Hz
    - PPG: 5 min @ 64Hz  = 19,200 samples → Bandpass 0.5-8Hz
    - EDA: 5 min @ 4Hz   = 1,200 samples  → Z-Score normalization

Usage:
    python preprocess_raw.py --subject c1s01_raw_single --data_path ./dataset/c1s01_raw_single
    python preprocess_raw.py --subject c1s01_raw_single --data_path ./dataset/c1s01_raw_single --limit_segments 10
"""

import os
import sys
import glob
import pickle
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

try:
    import tqdm as tqdm_module
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, **kwargs):
        return iterable

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from libs.dsp import filter_ecg, filter_ppg, normalize_eda, resize_signal, check_signal_quality


# ============================================================
# CGM Helper
# ============================================================

def process_glucose(glucose_path):
    """Read CGM data, filter EGV events, parse timestamps and glucose values."""
    glucose_df = pd.read_csv(glucose_path, delimiter=',')
    glucose_df = glucose_df[glucose_df['Event Type'] == 'EGV'].reset_index(drop=True)

    # Convert glucose values (handle 'Low' and 'High' strings)
    glucose_df['Glucose Value (mg/dL)'] = glucose_df['Glucose Value (mg/dL)'].astype(str)
    glucose_df['Glucose Value (mg/dL)'] = glucose_df['Glucose Value (mg/dL)'].str.replace('Low', '40')
    glucose_df['Glucose Value (mg/dL)'] = glucose_df['Glucose Value (mg/dL)'].str.replace('High', '400')

    glucose_df = glucose_df[['Timestamp (YYYY-MM-DDThh:mm:ss)', 'Glucose Value (mg/dL)', 'Index']]
    glucose_df.columns = ['Timestamp', 'glucose', 'Index']

    # Parse timestamps (try multiple formats)
    for fmt in ['%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M:%S.%f']:
        try:
            glucose_df['Timestamp'] = pd.to_datetime(glucose_df['Timestamp'], format=fmt)
            break
        except Exception:
            continue
    else:
        glucose_df['Timestamp'] = pd.to_datetime(glucose_df['Timestamp'])

    glucose_df['glucose'] = glucose_df['glucose'].astype(float)
    glucose_df = glucose_df.sort_values('Timestamp').reset_index(drop=True)
    return glucose_df


# ============================================================
# Signal Specs
# ============================================================

SIGNAL_SPECS = {
    'ecg': {'duration_sec': 300, 'sampling_rate': 250, 'num_samples': 75000},
    'ppg': {'duration_sec': 300, 'sampling_rate': 64,  'num_samples': 19200},
    'eda': {'duration_sec': 300, 'sampling_rate': 4,   'num_samples': 1200},
}


# ============================================================
# Main Pipeline
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ST-MGM: Raw Data Preprocessing')
    parser.add_argument('--subject', type=str, required=True, help='Subject ID (e.g., c1s01_raw_single)')
    parser.add_argument('--data_path', type=str, required=True, help='Path to raw data folder')
    parser.add_argument('--out_folder', default='./dataset/processed', type=str, help='Output folder')
    parser.add_argument('--limit_segments', type=int, default=0, help='Limit to first N segments (0=all)')
    args = parser.parse_args()

    subject_id = args.subject
    raw_dir = args.data_path
    out_dir = os.path.join(args.out_folder, subject_id)

    # Clean existing directory
    if os.path.exists(out_dir):
        print(f"Cleaning existing directory: {out_dir}")
        for f in glob.glob(os.path.join(out_dir, '*')):
            try:
                os.remove(f)
            except Exception:
                pass
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  ST-MGM: Preprocessing {subject_id}")
    print(f"{'='*60}")

    # ===== Step 1: Process CGM =====
    print("\n--- Processing CGM ---")
    glucose_path = os.path.join(raw_dir, 'cgm.csv')
    if not os.path.exists(glucose_path):
        print(f"ERROR: CGM file not found at {glucose_path}")
        sys.exit(1)
    glucose_df = process_glucose(glucose_path)
    print(f"CGM shape: {glucose_df.shape} (EGV readings)")
    print(f"Glucose range: {glucose_df['glucose'].min():.0f} - {glucose_df['glucose'].max():.0f} mg/dL")
    hypo_count = (glucose_df['glucose'] < 70).sum()
    print(f"Hypoglycemia segments (<70 mg/dL): {hypo_count}")

    # ===== Step 2: Find Zephyr and E4 folders =====
    zephyr_base = None
    for name in ['zephyr', 'zephyr-01', 'zephyr-02']:
        p = os.path.join(raw_dir, name)
        if os.path.exists(p):
            zephyr_base = p
            break

    if zephyr_base is None:
        print("ERROR: No Zephyr folder found!")
        sys.exit(1)

    e4_base = None
    for name in ['e4', 'e4-01', 'e4-02']:
        p = os.path.join(raw_dir, name)
        if os.path.exists(p):
            e4_base = p
            break

    if e4_base is None:
        print("ERROR: No E4 folder found!")
        sys.exit(1)

    # Find valid session folders
    valid_zephyr_folders = []
    for folder in sorted(glob.glob(os.path.join(zephyr_base, '*'))):
        if os.path.isdir(folder):
            ecg_files = glob.glob(os.path.join(folder, '*_ECG.csv'))
            summary_files = glob.glob(os.path.join(folder, '*_SummaryEnhanced.csv'))
            if ecg_files and summary_files:
                valid_zephyr_folders.append((folder, ecg_files[0], summary_files[0]))

    valid_e4_folders = sorted([
        d for d in glob.glob(os.path.join(e4_base, '*'))
        if os.path.isdir(d)
    ])

    print(f"\nFound {len(valid_zephyr_folders)} Zephyr sessions, {len(valid_e4_folders)} E4 sessions")

    # ===== Step 3: Scan time ranges & find overlapping CGM segments =====
    print("\n--- Scanning session time ranges ---")

    # Scan Zephyr
    zephyr_cgm_map = {}
    for folder_path, ecg_path, summary_path in tqdm(valid_zephyr_folders, desc="Scanning Zephyr"):
        summary_df = pd.read_csv(summary_path)
        if summary_df.empty:
            continue
        summary_df['Time'] = pd.to_datetime(summary_df['Time'], format='%d/%m/%Y %H:%M:%S.%f', errors='coerce')
        summary_df = summary_df.dropna(subset=['Time']).sort_values('Time').reset_index(drop=True)
        if summary_df.empty:
            continue
        session_start = summary_df['Time'].min()
        session_end = summary_df['Time'].max()
        for _, row in glucose_df.iterrows():
            idx = int(row['Index'])
            cgm_time = row['Timestamp']
            cgm_start = cgm_time - pd.Timedelta('5min')
            if cgm_start >= session_start and cgm_time <= session_end:
                zephyr_cgm_map[idx] = (folder_path, ecg_path, summary_path)

    # Scan E4
    e4_cgm_map = {}
    for folder_path in tqdm(valid_e4_folders, desc="Scanning E4"):
        eda_path = os.path.join(folder_path, 'EDA.csv')
        if not os.path.exists(eda_path):
            continue
        with open(eda_path, 'r') as f:
            unix_start = float(f.readline().strip().split(',')[0])
            sampling_rate = float(f.readline().strip().split(',')[0])
            num_rows = sum(1 for _ in f)
        start_dt = datetime.fromtimestamp(unix_start)
        end_dt = start_dt + timedelta(seconds=num_rows / sampling_rate)
        for _, row in glucose_df.iterrows():
            idx = int(row['Index'])
            cgm_time = row['Timestamp']
            cgm_start = cgm_time - pd.Timedelta('5min')
            if cgm_start >= start_dt and cgm_time <= end_dt:
                e4_cgm_map[idx] = folder_path

    # Find overlapping segments
    overlap_indices = sorted(list(set(zephyr_cgm_map.keys()) & set(e4_cgm_map.keys())))
    print(f"\nFound {len(overlap_indices)} segments with BOTH Zephyr and E4 data.")

    if args.limit_segments > 0:
        overlap_indices = overlap_indices[:args.limit_segments]
        print(f"Limiting to the first {args.limit_segments} overlapping segments.")

    if not overlap_indices:
        print("ERROR: No overlapping segments found between Zephyr, E4 and CGM!")
        sys.exit(1)

    # ===== Step 4: Extract, Filter, and Save segments =====
    print("\n--- Extracting & filtering 5-minute windows ---")

    needed_zephyr = set(zephyr_cgm_map[idx] for idx in overlap_indices)
    needed_e4 = set(e4_cgm_map[idx] for idx in overlap_indices)

    # Initialize segment storage
    cgm_data_dict = {}
    for idx in overlap_indices:
        row = glucose_df[glucose_df['Index'] == idx].iloc[0]
        cgm_data_dict[idx] = {
            'Index': idx,
            'Timestamp': row['Timestamp'],
            'glucose': float(row['glucose']),
            'ecg_raw': None,
            'ppg_raw': None,
            'eda_raw': None,
            'ecg_filtered': None,
            'ppg_filtered': None,
            'eda_normalized': None,
        }

    # Process Zephyr ECG
    print("\n--- Processing Zephyr ECG ---")
    for idx_folder, (folder_path, ecg_path, summary_path) in enumerate(needed_zephyr):
        print(f"[{idx_folder+1}/{len(needed_zephyr)}] Loading ECG: {os.path.basename(folder_path)}")

        first_row = pd.read_csv(ecg_path, nrows=1)
        first_time_str = first_row['Time'].values[0]
        first_time = pd.to_datetime(first_time_str, format='%d/%m/%Y %H:%M:%S.%f')

        ecg_df = pd.read_csv(ecg_path, usecols=['EcgWaveform'])
        ecg_wave = ecg_df['EcgWaveform'].values.astype(np.float32)

        for idx in overlap_indices:
            if zephyr_cgm_map[idx][0] != folder_path:
                continue
            cgm_time = cgm_data_dict[idx]['Timestamp']
            cgm_start = cgm_time - pd.Timedelta('5min')

            start_sec = (cgm_start - first_time).total_seconds()
            end_sec = (cgm_time - first_time).total_seconds()
            start_i = int(start_sec * 250)
            end_i = int(end_sec * 250)
            start_i = max(0, min(start_i, len(ecg_wave)))
            end_i = max(0, min(end_i, len(ecg_wave)))

            if end_i > start_i:
                raw_ecg = ecg_wave[start_i:end_i]
                # Resize to exactly 75,000 samples
                raw_ecg = resize_signal(raw_ecg, SIGNAL_SPECS['ecg']['num_samples'])
                # Apply bandpass filter (0.5-40Hz)
                filtered_ecg = filter_ecg(raw_ecg, fs=250)

                cgm_data_dict[idx]['ecg_raw'] = raw_ecg
                cgm_data_dict[idx]['ecg_filtered'] = filtered_ecg

    # Process E4 (BVP/PPG and EDA)
    print("\n--- Processing E4 (BVP/PPG + EDA) ---")
    for idx_folder, folder_path in enumerate(needed_e4):
        print(f"[{idx_folder+1}/{len(needed_e4)}] Loading E4: {os.path.basename(folder_path)}")

        for feat, target_key_raw, target_key_proc in [
            ('BVP', 'ppg_raw', 'ppg_filtered'),
            ('EDA', 'eda_raw', 'eda_normalized'),
        ]:
            csv_path = os.path.join(folder_path, f'{feat}.csv')
            if not os.path.exists(csv_path):
                continue
            df = pd.read_csv(csv_path, header=None)
            unix_start_feat = df.loc[0][0]
            sampling_rate_feat = df.loc[1][0]
            data_feat = df.iloc[2:].to_numpy()[:, 0].astype(np.float32)
            start_dt_feat = datetime.fromtimestamp(unix_start_feat)

            for idx in overlap_indices:
                if e4_cgm_map[idx] != folder_path:
                    continue
                cgm_time = cgm_data_dict[idx]['Timestamp']
                cgm_start = cgm_time - pd.Timedelta('5min')
                start_sec = (cgm_start - start_dt_feat).total_seconds()
                end_sec = (cgm_time - start_dt_feat).total_seconds()
                start_i = int(start_sec * sampling_rate_feat)
                end_i = int(end_sec * sampling_rate_feat)
                start_i = max(0, min(start_i, len(data_feat)))
                end_i = max(0, min(end_i, len(data_feat)))

                if end_i > start_i:
                    raw_sig = data_feat[start_i:end_i]

                    if feat == 'BVP':
                        target_samples = SIGNAL_SPECS['ppg']['num_samples']
                        raw_sig = resize_signal(raw_sig, target_samples)
                        processed = filter_ppg(raw_sig, fs=64)
                    else:  # EDA
                        target_samples = SIGNAL_SPECS['eda']['num_samples']
                        raw_sig = resize_signal(raw_sig, target_samples)
                        processed = normalize_eda(raw_sig)

                    cgm_data_dict[idx][target_key_raw] = raw_sig
                    cgm_data_dict[idx][target_key_proc] = processed

    # ===== Step 5: Save segmented pickles =====
    print(f"\n--- Saving segmented files to {out_dir} ---")
    segments_saved = 0
    stats = {'total': 0, 'has_ecg': 0, 'has_ppg': 0, 'has_eda': 0, 'has_all': 0}

    for idx in overlap_indices:
        segment = cgm_data_dict[idx]
        has_ecg = segment['ecg_filtered'] is not None
        has_ppg = segment['ppg_filtered'] is not None
        has_eda = segment['eda_normalized'] is not None

        stats['total'] += 1
        if has_ecg:
            stats['has_ecg'] += 1
        if has_ppg:
            stats['has_ppg'] += 1
        if has_eda:
            stats['has_eda'] += 1
        if has_ecg and has_ppg and has_eda:
            stats['has_all'] += 1

        if has_ecg or has_ppg or has_eda:
            out_path = os.path.join(out_dir, f'{segments_saved}.pkl')
            with open(out_path, 'wb') as f:
                pickle.dump(segment, f)
            segments_saved += 1

    print(f"\n--- Preprocessing Summary ---")
    print(f"Total overlapping CGM segments: {stats['total']}")
    print(f"  With ECG: {stats['has_ecg']}")
    print(f"  With PPG: {stats['has_ppg']}")
    print(f"  With EDA: {stats['has_eda']}")
    print(f"  With ALL modalities: {stats['has_all']}")
    print(f"\nSaved {segments_saved} segmented files to {out_dir}")
    print("Done!")
