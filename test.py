"""
ST-MGM: Phase 5 — Evaluation & Testing
=========================================
Evaluates trained ST-MGM models with:

Classification Metrics:
    - AUC-ROC, F1-Score, Sensitivity, Specificity, Accuracy
    - EER threshold

Regression Metrics:
    - RMSE (Root Mean Squared Error)
    - MAE (Mean Absolute Error)
    - MARD (Mean Absolute Relative Difference)

Usage:
    python test.py --subject c1s01_raw_single
"""

import os
import sys
import json
import pickle
import random
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score,
    f1_score, roc_auc_score, average_precision_score,
    roc_curve, mean_squared_error, mean_absolute_error
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from libs.st_mgm_model import STMGMModel, MultiTaskLoss
from libs.dataloader import STMGMDataset, stmgm_collate_fn


def seed_everything(seed=42):
    """Set all random seeds for deterministic execution."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================
# Metrics
# ============================================================

def find_eer_threshold(targets, probs):
    """
    Find calibrated probability threshold balancing F1-score and Youden's J (TPR - FPR)
    constrained within a prevalence-aware probability quantile range to prevent
    extreme overconfident training thresholds.
    """
    targets = np.asarray(targets).reshape(-1)
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    if len(targets) == 0 or len(np.unique(targets)) < 2 or np.allclose(probs, probs[0]):
        return 0.5
    fpr, tpr, thresholds = roc_curve(targets, probs)
    finite_mask = np.isfinite(thresholds)
    if not np.any(finite_mask):
        return 0.5
    fpr = fpr[finite_mask]
    tpr = tpr[finite_mask]
    thresholds = thresholds[finite_mask]

    n_pos = float(np.sum(targets == 1))
    n_neg = float(np.sum(targets == 0))
    prevalence = float(np.clip(n_pos / max(n_pos + n_neg, 1.0), 0.02, 0.40))

    tp = tpr * n_pos
    fp = fpr * n_neg
    fn = (1.0 - tpr) * n_pos
    f1_scores = (2.0 * tp) / np.maximum(2.0 * tp + fp + fn, 1e-8)
    youden_j = tpr - fpr
    combined_score = 0.5 * f1_scores + 0.5 * youden_j

    q_low = float(np.quantile(probs, np.clip(1.0 - 3.5 * prevalence, 0.50, 0.85)))
    q_high = float(np.quantile(probs, np.clip(1.0 - 0.8 * prevalence, 0.70, 0.95)))
    q_prev = float(np.quantile(probs, np.clip(1.0 - 1.25 * prevalence, 0.65, 0.94)))

    low_bound = float(np.clip(min(q_low, q_high), 0.20, 0.75))
    high_bound = float(np.clip(max(q_low, q_high), max(low_bound + 0.05, 0.30), 0.80))

    valid_mask = (thresholds >= low_bound) & (thresholds <= high_bound)
    if not np.any(valid_mask):
        valid_mask = (thresholds >= 0.20) & (thresholds <= 0.80)

    if np.any(valid_mask):
        valid_indices = np.where(valid_mask)[0]
        best_local = int(np.nanargmax(combined_score[valid_indices]))
        thr_opt = float(thresholds[valid_indices[best_local]])
    else:
        best_idx = int(np.nanargmax(combined_score))
        thr_opt = float(np.clip(thresholds[best_idx], 0.20, 0.80))

    q_prev_clipped = float(np.clip(q_prev, 0.20, 0.80))
    thr = 0.7 * thr_opt + 0.3 * q_prev_clipped
    return float(np.clip(thr, 0.20, 0.80))


def compute_mard(y_true, y_pred):
    """
    Mean Absolute Relative Difference (MARD).
    MARD = mean(|y_pred - y_true| / y_true) * 100%
    """
    mask = y_true > 0
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs(y_pred[mask] - y_true[mask]) / y_true[mask]) * 100)


def compute_parkes_error_grid(y_true, y_pred):
    """
    Compute Parkes Consensus Error Grid zone percentages for Type 1 Diabetes.

    Args:
        y_true: 1D array of reference glucose values (mg/dL)
        y_pred: 1D array of predicted glucose values (mg/dL)

    Returns:
        dict with percentages in zones 'A', 'B', 'C', 'D', 'E', and combined 'AB'.
    """
    ref = np.asarray(y_true, dtype=np.float64).reshape(-1)
    pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    n = len(ref)
    if n == 0:
        return {'A': 0.0, 'B': 0.0, 'C': 0.0, 'D': 0.0, 'E': 0.0, 'AB': 0.0}

    counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0}

    for r, p in zip(ref, pred):
        if p >= r:
            # Above diagonal: check upper boundaries from most severe (E) to A
            if r <= 50.0 and p > np.interp(r, [0.0, 35.0, 50.0], [150.0, 155.0, 550.0]):
                counts['E'] += 1
            elif r <= 125.0 and p > np.interp(
                r, [0.0, 25.0, 50.0, 80.0, 125.0], [100.0, 100.0, 125.0, 215.0, 550.0]
            ):
                counts['D'] += 1
            elif r <= 260.0 and p > np.interp(
                r, [0.0, 30.0, 50.0, 70.0, 260.0], [60.0, 60.0, 80.0, 110.0, 550.0]
            ):
                counts['C'] += 1
            elif r <= 430.0 and p > np.interp(
                r, [0.0, 30.0, 140.0, 280.0, 430.0], [50.0, 50.0, 170.0, 380.0, 550.0]
            ):
                counts['B'] += 1
            else:
                counts['A'] += 1
        else:
            # Below diagonal: check lower boundaries from most severe (D) to A
            if r >= 250.0 and (
                p <= 40.0 or p < np.interp(r, [250.0, 550.0], [40.0, 150.0])
            ):
                counts['D'] += 1
            elif r >= 120.0 and (
                p <= 30.0 or p < np.interp(r, [120.0, 260.0, 550.0], [30.0, 130.0, 250.0])
            ):
                counts['C'] += 1
            elif r >= 50.0 and (
                p <= 30.0 or p < np.interp(
                    r, [50.0, 170.0, 385.0, 550.0], [30.0, 145.0, 300.0, 450.0]
                )
            ):
                counts['B'] += 1
            else:
                counts['A'] += 1

    pct = {k: float(v * 100.0 / n) for k, v in counts.items()}
    pct['AB'] = float((counts['A'] + counts['B']) * 100.0 / n)
    return pct


def evaluate_fold(model, dataloader, device, glucose_mean=0.0, glucose_std=1.0, fixed_threshold=None):
    """Evaluate a single fold and return all classification, regression, and Parkes metrics."""
    model.eval()
    all_cls_preds = []
    all_cls_labels = []
    all_reg_preds = []
    all_reg_labels = []

    with torch.no_grad():
        for batch in dataloader:
            f_temporal = batch['f_temporal'].to(device)
            ecg_beat = batch['ecg_beat'].to(device)
            ppg_pulse = batch['ppg_pulse'].to(device)
            eda = batch['eda'].to(device)
            hypo_label = batch['hypo_label']
            glucose = batch['glucose']

            cls_logits, reg_pred = model(f_temporal, ecg_beat, ppg_pulse, eda)
            cls_prob = torch.sigmoid(cls_logits)

            # Denormalize glucose predictions back to mg/dL
            reg_pred_mgdl = reg_pred * float(glucose_std) + float(glucose_mean)

            all_cls_preds.extend(cls_prob.reshape(-1).cpu().numpy().tolist())
            all_cls_labels.extend(hypo_label.reshape(-1).numpy().tolist())
            all_reg_preds.extend(reg_pred_mgdl.reshape(-1).cpu().numpy().tolist())
            all_reg_labels.extend(glucose.reshape(-1).numpy().tolist())

    cls_preds = np.array(all_cls_preds, dtype=np.float64)
    cls_labels = np.array(all_cls_labels, dtype=np.int64)
    reg_preds = np.array(all_reg_preds, dtype=np.float64)
    reg_labels = np.array(all_reg_labels, dtype=np.float64)

    # Classification threshold: prefer leak-free threshold determined on training fold
    val_eer_threshold = find_eer_threshold(cls_labels, cls_preds)
    if fixed_threshold is not None and np.isfinite(fixed_threshold):
        threshold = float(np.clip(fixed_threshold, 0.20, 0.80))
    else:
        threshold = val_eer_threshold

    cls_binary = (cls_preds >= threshold).astype(int)

    try:
        if len(np.unique(cls_labels)) > 1:
            auc = float(roc_auc_score(cls_labels, cls_preds))
        else:
            auc = 0.5
    except Exception:
        auc = 0.5

    try:
        if len(np.unique(cls_labels)) > 1:
            auprc = float(average_precision_score(cls_labels, cls_preds))
        else:
            auprc = 0.0
    except Exception:
        auprc = 0.0

    acc = accuracy_score(cls_labels, cls_binary)
    sens = recall_score(cls_labels, cls_binary, zero_division=0)
    spec = recall_score(1 - cls_labels, 1 - cls_binary, zero_division=0)
    balanced_acc = 0.5 * (float(sens) + float(spec))
    prec = precision_score(cls_labels, cls_binary, zero_division=0)
    f1 = f1_score(cls_labels, cls_binary, zero_division=0)

    # Regression metrics (in mg/dL)
    rmse = float(np.sqrt(mean_squared_error(reg_labels, reg_preds)))
    mae = float(mean_absolute_error(reg_labels, reg_preds))
    mard = compute_mard(reg_labels, reg_preds)
    try:
        if len(reg_labels) > 1 and np.std(reg_labels) > 1e-8 and np.std(reg_preds) > 1e-8:
            r_val = float(np.corrcoef(reg_labels, reg_preds)[0, 1])
            pearson_r = r_val if np.isfinite(r_val) else 0.0
        else:
            pearson_r = 0.0
    except Exception:
        pearson_r = 0.0
    parkes = compute_parkes_error_grid(reg_labels, reg_preds)

    return {
        # Classification
        'auc': float(auc),
        'auprc': float(auprc),
        'acc': float(acc),
        'balanced_acc': float(balanced_acc),
        'sensitivity': float(sens),
        'specificity': float(spec),
        'precision': float(prec),
        'f1': float(f1),
        'eer_threshold': float(threshold),
        'val_eer_threshold': float(val_eer_threshold),
        # Regression
        'rmse': rmse,
        'mae': mae,
        'mard': mard,
        'pearson_r': float(pearson_r),
        'parkes_zone_a': parkes['A'],
        'parkes_zone_b': parkes['B'],
        'parkes_ab': parkes['AB'],
        # Raw predictions
        'cls_preds': cls_preds.tolist(),
        'cls_labels': cls_labels.tolist(),
        'reg_preds': reg_preds.tolist(),
        'reg_labels': reg_labels.tolist(),
    }


# ============================================================
# Main Evaluation
# ============================================================

def test_subject(subject, data_dir='./data', results_dir='./results', variant='M9'):
    """Evaluate all folds for a subject (supports variant folders)."""
    seed_everything(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    meta_path = os.path.join(data_dir, subject, 'metadata.json')
    segments_path = os.path.join(data_dir, subject, 'segments.pkl')
    temporal_path = os.path.join(data_dir, subject, 'temporal_features.pkl')

    variant_name = variant.upper()
    model_dir = os.path.join(results_dir, subject) if variant_name == 'M9' else os.path.join(results_dir, f"{subject}_{variant_name.lower()}")
    summary_path = os.path.join(model_dir, 'training_summary.json')

    if not os.path.exists(meta_path):
        print(f"ERROR: Metadata not found for {subject}")
        return

    with open(meta_path, 'r') as f:
        metadata = json.load(f)

    with open(segments_path, 'rb') as f:
        segments = pickle.load(f)

    with open(temporal_path, 'rb') as f:
        temporal_data = pickle.load(f)

    training_summary = {}
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            training_summary = json.load(f)

    temporal_raw = temporal_data.get('raw')
    temporal_selected_fallback = temporal_data.get('selected')
    selected_indices_fallback = temporal_data.get('selected_indices')
    fold_selected_map = temporal_data.get(
        'fold_selected_indices',
        metadata.get('fold_selected_indices', {}),
    )
    hypo_threshold = float(metadata.get('hypo_threshold', 70))
    folds = metadata.get('folds', metadata)

    print(f"\n{'='*60}")
    print(f"  ST-MGM Evaluation: {subject}")
    print(f"{'='*60}")

    all_results = {}
    metrics_list = []

    for fold_name, fold_info in folds.items():
        model_path = os.path.join(model_dir, f'{fold_name}.pt')
        if not os.path.exists(model_path):
            print(f"\n  {fold_name}: Model not found at {model_path}, skipping.")
            continue

        print(f"\n--- {fold_name} ---")
        seed_everything(42)

        checkpoint = torch.load(model_path, map_location=device)
        fold_summary = training_summary.get(fold_name, {})

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            fold_meta = checkpoint
        else:
            state_dict = checkpoint
            fold_meta = fold_summary

        # Resolve per-fold selected feature indices
        sel_idx = fold_meta.get(
            'selected_indices',
            fold_info.get('selected_indices', fold_selected_map.get(fold_name, selected_indices_fallback)),
        )
        sel_idx = np.array(sel_idx, dtype=int)

        val_indices = fold_info.get('val', [])
        train_indices = fold_info.get('train', [])
        val_segments = [segments[i] for i in val_indices]

        if temporal_raw is not None:
            val_temporal_raw = temporal_raw[val_indices][:, sel_idx]
            train_temporal_raw = temporal_raw[train_indices][:, sel_idx]
        else:
            val_temporal_raw = temporal_selected_fallback[val_indices]
            train_temporal_raw = temporal_selected_fallback[train_indices]

        # Standardize val_temporal using training fold statistics
        if 'temporal_mean' in fold_meta and 'temporal_std' in fold_meta:
            feat_mean = np.array(fold_meta['temporal_mean'], dtype=np.float32)
            feat_std = np.array(fold_meta['temporal_std'], dtype=np.float32)
        else:
            feat_mean = np.mean(train_temporal_raw, axis=0).astype(np.float32)
            feat_std = np.std(train_temporal_raw, axis=0).astype(np.float32)
        feat_std = np.where(feat_std < 1e-8, 1.0, feat_std).astype(np.float32)

        val_temporal = np.nan_to_num(
            (val_temporal_raw - feat_mean) / feat_std, nan=0.0, posinf=0.0, neginf=0.0
        ).astype(np.float32)

        glucose_mean = float(fold_meta.get('glucose_mean', 0.0))
        glucose_std = float(fold_meta.get('glucose_std', 1.0))
        fixed_threshold = fold_meta.get('eer_threshold', None)

        temporal_dim = val_temporal.shape[1]

        val_dataset = STMGMDataset(
            val_segments, val_temporal, augment=False,
            hypo_threshold=hypo_threshold, verbose=False,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=64, shuffle=False,
            collate_fn=stmgm_collate_fn, num_workers=0,
        )

        fold_variant = fold_meta.get('variant', variant_name)
        model = STMGMModel(temporal_dim=temporal_dim, variant=fold_variant)
        model.set_reg_threshold(
            hypo_threshold=hypo_threshold,
            glucose_mean=glucose_mean,
            glucose_std=glucose_std,
        )
        model = model.to(device)
        model.load_state_dict(state_dict, strict=False)

        results = evaluate_fold(
            model, val_loader, device,
            glucose_mean=glucose_mean,
            glucose_std=glucose_std,
            fixed_threshold=fixed_threshold,
        )

        # Print results
        print(f"  Classification: AUC={results['auc']:.4f}, AUPRC={results['auprc']:.4f}, "
              f"F1={results['f1']:.4f}, BalAcc={results['balanced_acc']:.4f}, "
              f"Sens={results['sensitivity']:.4f}, Spec={results['specificity']:.4f}, "
              f"Acc={results['acc']:.4f}")
        print(f"  Regression:     RMSE={results['rmse']:.2f} mg/dL, "
              f"MAE={results['mae']:.2f} mg/dL, MARD={results['mard']:.2f}%, "
              f"Pearson r={results['pearson_r']:.4f}")
        print(f"  Parkes Grid:    A+B={results['parkes_ab']:.2f}% "
              f"(Zone A={results['parkes_zone_a']:.2f}%, Zone B={results['parkes_zone_b']:.2f}%)")
        print(f"  Threshold:      train_eer={results['eer_threshold']:.4f} "
              f"(val_eer={results['val_eer_threshold']:.4f})")

        # Store without raw predictions for summary
        metrics_list.append({
            'auc': results['auc'],
            'auprc': results['auprc'],
            'f1': results['f1'],
            'balanced_acc': results['balanced_acc'],
            'sensitivity': results['sensitivity'],
            'specificity': results['specificity'],
            'acc': results['acc'],
            'rmse': results['rmse'],
            'mae': results['mae'],
            'mard': results['mard'],
            'pearson_r': results['pearson_r'],
            'parkes_zone_a': results['parkes_zone_a'],
            'parkes_zone_b': results['parkes_zone_b'],
            'parkes_ab': results['parkes_ab'],
        })

        all_results[fold_name] = results

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Print average results
    if metrics_list:
        print(f"\n{'='*60}")
        print(f"  Average Results ({len(metrics_list)} folds)")
        print(f"{'='*60}")

        for metric in ['auc', 'auprc', 'f1', 'balanced_acc', 'sensitivity', 'specificity', 'acc',
                       'rmse', 'mae', 'mard', 'pearson_r', 'parkes_zone_a', 'parkes_zone_b', 'parkes_ab']:
            values = [m[metric] for m in metrics_list]
            if metric in ['rmse', 'mae']:
                unit = ' mg/dL'
            elif metric in ['mard', 'parkes_zone_a', 'parkes_zone_b', 'parkes_ab']:
                unit = '%'
            else:
                unit = ''
            print(f"  {metric.upper():14s}: {np.mean(values):.4f} ± {np.std(values):.4f}{unit}")

    # Save results
    eval_path = os.path.join(model_dir, 'evaluation_results.json')
    # Remove raw predictions from saved results to reduce file size
    save_results = {}
    for fold, res in all_results.items():
        save_results[fold] = {k: v for k, v in res.items()
                              if k not in ['cls_preds', 'cls_labels', 'reg_preds', 'reg_labels']}
    with open(eval_path, 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"\nResults saved to {eval_path}")


# ============================================================
# CLI
# ============================================================

if __name__ == '__main__':
    VALID_VARIANTS = [
        'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',
        'M9_CLS', 'M9_REG', 'M9_NOCOUPLE', 'SINGLE_CLS', 'SINGLE_REG', 'NO_COUPLE',
        'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9',
        'm9_cls', 'm9_reg', 'm9_nocouple', 'single_cls', 'single_reg', 'no_couple',
    ]
    parser = argparse.ArgumentParser(description='ST-MGM: Evaluation')
    parser.add_argument('-s', '--subject', type=str, required=True, help='Subject ID')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--results_dir', type=str, default='./results')
    parser.add_argument('--variant', type=str, default='M9', choices=VALID_VARIANTS,
                        help='Ablation variant folder to evaluate')
    args = parser.parse_args()

    test_subject(args.subject, args.data_dir, args.results_dir, args.variant)
