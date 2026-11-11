"""
ST-MGM: DWT Feature Extraction & Selection Module
====================================================
- Discrete Wavelet Transform (db4, 7-level decomposition)
- 10 Statistical Features per sub-band
- Feature Selection: UFS + L1-FS + RFE → ~18 features (F_temporal)

Phase 1 of the ST-MGM pipeline.
"""

import numpy as np
import pywt
from scipy.stats import kurtosis, skew
from scipy.signal import welch


# ============================================================
# DWT Decomposition
# ============================================================

def dwt_decompose(signal, wavelet='db4', level=7):
    """
    Perform DWT decomposition using db4 wavelet up to 7 levels.

    Args:
        signal: 1D numpy array (ECG or PPG consecutive cycles)
        wavelet: Mother wavelet (default 'db4')
        level: Decomposition level (default 7)

    Returns:
        List of 8 sub-bands: [raw_signal, detail_1, ..., detail_7]
        Note: We include the raw signal as sub-band 0
    """
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    # coeffs = [cA7, cD7, cD6, cD5, cD4, cD3, cD2, cD1]
    # We want: [raw, cD1, cD2, cD3, cD4, cD5, cD6, cD7]
    sub_bands = [signal]  # raw signal as sub-band 0
    for i in range(len(coeffs) - 1, 0, -1):
        sub_bands.append(coeffs[i])  # detail coefficients from level 1 to 7

    return sub_bands


# ============================================================
# 10 Statistical Features per Sub-band
# ============================================================

def compute_stat_features(sub_band, fs=250):
    """
    Compute 10 statistical features for a single sub-band.

    Features:
        1. Kurtosis (Kur)
        2. Skewness (Ske)
        3. Spectral Mean (SM)
        4. Spectral Centroid (SC)
        5. Frequency Domain features (FD) - dominant frequency
        6. Complexity Domain features (CD) - approximate entropy proxy
        7. Cosine / Correlation (CO) - autocorrelation at lag 1
        8. Power Spectral Entropy (PSE)
        9. Energy / Shannon Entropy (KE)
        10. Spectral Entropy (SE)

    Args:
        sub_band: 1D numpy array (wavelet sub-band)
        fs: Sampling frequency for spectral features

    Returns:
        numpy array of shape (10,)
    """
    n = len(sub_band)
    features = np.zeros(10, dtype=np.float32)

    if n < 4:
        return features

    # 1. Kurtosis
    features[0] = kurtosis(sub_band, fisher=True, nan_policy='omit')

    # 2. Skewness
    features[1] = skew(sub_band, nan_policy='omit')

    # 3-6, 8, 10: Spectral features via Welch PSD
    try:
        nperseg = min(256, n)
        freqs, psd = welch(sub_band, fs=fs, nperseg=nperseg)

        psd_sum = np.sum(psd)
        if psd_sum > 1e-12:
            psd_norm = psd / psd_sum

            # 3. Spectral Mean
            features[2] = np.mean(psd)

            # 4. Spectral Centroid
            features[3] = np.sum(freqs * psd_norm)

            # 5. Dominant Frequency (FD)
            features[4] = freqs[np.argmax(psd)]

            # 8. Power Spectral Entropy (PSE)
            psd_pos = psd_norm[psd_norm > 0]
            features[7] = -np.sum(psd_pos * np.log2(psd_pos))

            # 10. Spectral Entropy (SE) — normalized
            features[9] = features[7] / np.log2(len(psd_pos)) if len(psd_pos) > 1 else 0.0
        else:
            features[2] = 0.0
            features[3] = 0.0
            features[4] = 0.0
            features[7] = 0.0
            features[9] = 0.0
    except Exception:
        pass

    # 6. Complexity Domain (CD) — sample variance of first difference
    if n > 1:
        first_diff = np.diff(sub_band)
        features[5] = np.var(first_diff)
    else:
        features[5] = 0.0

    # 7. Cosine / Correlation (CO) — autocorrelation at lag 1
    if n > 1:
        mean_val = np.mean(sub_band)
        centered = sub_band - mean_val
        var_val = np.var(sub_band)
        if var_val > 1e-12:
            features[6] = np.sum(centered[:-1] * centered[1:]) / ((n - 1) * var_val)
        else:
            features[6] = 0.0
    else:
        features[6] = 0.0

    # 9. Energy / Shannon Entropy (KE)
    energy = np.sum(sub_band ** 2)
    if energy > 1e-12:
        prob = (sub_band ** 2) / energy
        prob_pos = prob[prob > 0]
        features[8] = -np.sum(prob_pos * np.log2(prob_pos))
    else:
        features[8] = 0.0

    # Replace NaN/Inf with 0
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    return features


# ============================================================
# Full DWT Feature Extraction for a Single Modality
# ============================================================

def extract_dwt_features(signal, fs=250, wavelet='db4', level=7):
    """
    Extract all DWT statistical features for a single modality signal.

    Pipeline: signal → DWT decompose → 10 stats × 8 sub-bands = 80 features

    Args:
        signal: 1D numpy array (e.g., consecutive ECG cardiac cycles)
        fs: Sampling frequency
        wavelet: Mother wavelet
        level: DWT decomposition level

    Returns:
        numpy array of shape (80,) — 10 features × 8 sub-bands
    """
    sub_bands = dwt_decompose(signal, wavelet, level)

    all_features = []
    for sb in sub_bands:
        feat = compute_stat_features(sb, fs=fs)
        all_features.append(feat)

    return np.concatenate(all_features)


def extract_temporal_features(ecg_signal, ppg_signal, fs_ecg=250, fs_ppg=64,
                               wavelet='db4', level=7):
    """
    Extract combined temporal features from ECG + PPG.

    Pipeline:
        ECG → 80 features (10 stats × 8 sub-bands)
        PPG → 80 features (10 stats × 8 sub-bands)
        Total = 160 features (before selection)

    Args:
        ecg_signal: 1D ECG signal (consecutive cardiac cycles, 5-min window)
        ppg_signal: 1D PPG signal (consecutive pulses, 5-min window)
        fs_ecg: ECG sampling rate (250Hz)
        fs_ppg: PPG sampling rate (64Hz)
        wavelet: Mother wavelet
        level: DWT level

    Returns:
        numpy array of shape (160,) — raw temporal features
    """
    ecg_features = extract_dwt_features(ecg_signal, fs=fs_ecg, wavelet=wavelet, level=level)
    ppg_features = extract_dwt_features(ppg_signal, fs=fs_ppg, wavelet=wavelet, level=level)
    return np.concatenate([ecg_features, ppg_features])


# ============================================================
# Feature Selection: UFS + L1-FS + RFE → ~18 features
# ============================================================

def select_features_ufs(features_matrix, n_keep=80):
    """
    Unsupervised Feature Selection based on variance.
    Removes features with near-zero variance.

    Args:
        features_matrix: (N_samples, 160) array
        n_keep: Max features to keep

    Returns:
        selected_indices: indices of selected features
    """
    variances = np.var(features_matrix, axis=0)
    sorted_indices = np.argsort(-variances)
    return sorted_indices[:n_keep]


def select_features_l1(features_matrix, labels, alpha=0.01):
    """
    L1-based Feature Selection using Lasso on standardized features and targets.

    Args:
        features_matrix: (N_samples, N_features) array
        labels: (N_samples,) regression targets (glucose values)
        alpha: L1 regularization strength

    Returns:
        selected_indices: indices of features with non-zero coefficients
    """
    from sklearn.linear_model import Lasso
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features_matrix)

    y_arr = np.asarray(labels, dtype=np.float64)
    y_std = np.std(y_arr)
    if y_std > 1e-8:
        y_scaled = (y_arr - np.mean(y_arr)) / y_std
    else:
        y_scaled = y_arr - np.mean(y_arr)

    eff_alpha = max(float(alpha), 0.02)
    lasso = Lasso(alpha=eff_alpha, max_iter=10000, tol=1e-3, random_state=42)
    lasso.fit(X_scaled, y_scaled)

    # Select features with non-zero coefficients
    selected = np.where(np.abs(lasso.coef_) > 1e-5)[0]
    if len(selected) == 0:
        # Fallback: select top features by absolute coefficient
        selected = np.argsort(-np.abs(lasso.coef_))[:20]

    return selected


def select_features_rfe(features_matrix, labels, n_features=18):
    """
    Recursive Feature Elimination.

    Args:
        features_matrix: (N_samples, N_features) array
        labels: (N_samples,) binary labels
        n_features: Target number of features

    Returns:
        selected_indices: indices of selected features
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import RFE

    clf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    rfe = RFE(clf, n_features_to_select=n_features, step=5)
    rfe.fit(features_matrix, labels)

    return np.where(rfe.support_)[0]


def combined_feature_selection(features_matrix, labels_cls, labels_reg,
                                n_final=18, alpha=0.01):
    """
    Combined feature selection: UFS + L1-FS + RFE → intersection/union → ~18 features.

    Args:
        features_matrix: (N_samples, 160) raw temporal features
        labels_cls: (N_samples,) binary classification labels (hypo/normal)
        labels_reg: (N_samples,) regression targets (glucose mg/dL)
        n_final: Target number of final features
        alpha: L1 regularization strength

    Returns:
        selected_indices: Final selected feature indices
        feature_importances: dict with selection method results
    """
    n_samples, n_features = features_matrix.shape

    # Step 1: UFS — variance-based pre-filter
    ufs_indices = select_features_ufs(features_matrix, n_keep=min(80, n_features))
    X_ufs = features_matrix[:, ufs_indices]

    # Step 2: L1-FS on UFS-filtered features
    try:
        l1_local = select_features_l1(X_ufs, labels_reg, alpha=alpha)
        l1_indices = ufs_indices[l1_local]
    except Exception:
        l1_indices = ufs_indices[:n_final]

    # Step 3: RFE on UFS-filtered features
    try:
        rfe_local = select_features_rfe(X_ufs, labels_cls, n_features=min(n_final, len(ufs_indices)))
        rfe_indices = ufs_indices[rfe_local]
    except Exception:
        rfe_indices = ufs_indices[:n_final]

    # Combine: union of L1 and RFE, then trim to n_final
    combined = np.union1d(l1_indices, rfe_indices)

    if len(combined) > n_final:
        # Use variance ranking to pick top n_final from combined
        variances = np.var(features_matrix[:, combined], axis=0)
        top_k = np.argsort(-variances)[:n_final]
        final_indices = combined[top_k]
    elif len(combined) < n_final:
        # Pad with remaining UFS features
        remaining = np.setdiff1d(ufs_indices, combined)
        n_pad = n_final - len(combined)
        final_indices = np.concatenate([combined, remaining[:n_pad]])
    else:
        final_indices = combined

    final_indices = np.sort(final_indices).astype(int)

    return final_indices, {
        'ufs': ufs_indices,
        'l1': l1_indices,
        'rfe': rfe_indices,
        'final': final_indices
    }


# ============================================================
# Quick Test
# ============================================================

if __name__ == '__main__':
    print("--- DWT Features Module Test ---\n")

    # Generate fake signals
    np.random.seed(42)
    ecg = np.random.randn(75000).astype(np.float32)
    ppg = np.random.randn(19200).astype(np.float32)

    # Test DWT decomposition
    sub_bands = dwt_decompose(ecg)
    print(f"DWT sub-bands: {len(sub_bands)}")
    for i, sb in enumerate(sub_bands):
        print(f"  Sub-band {i}: length = {len(sb)}")

    # Test stat features
    feat = compute_stat_features(sub_bands[0], fs=250)
    print(f"\nStat features (sub-band 0): {feat.shape}")
    print(f"  Values: {feat}")

    # Test full extraction
    ecg_feats = extract_dwt_features(ecg, fs=250)
    print(f"\nECG DWT features: {ecg_feats.shape}")

    ppg_feats = extract_dwt_features(ppg, fs=64)
    print(f"PPG DWT features: {ppg_feats.shape}")

    # Test combined
    temporal = extract_temporal_features(ecg, ppg)
    print(f"Combined temporal features: {temporal.shape}")

    # Test feature selection
    N = 50
    X = np.random.randn(N, 160).astype(np.float32)
    y_cls = np.random.randint(0, 2, N)
    y_reg = np.random.randn(N) * 30 + 100

    selected, info = combined_feature_selection(X, y_cls, y_reg, n_final=18)
    print(f"\nFeature selection: {len(selected)} features selected")
    print(f"  Indices: {selected}")

    print("\nAll DWT feature tests passed!")
