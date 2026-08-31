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
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score,
    f1_score, roc_auc_score, roc_curve, mean_squared_error, mean_absolute_error
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from libs.st_mgm_model import STMGMModel, MultiTaskLoss
from libs.dataloader import STMGMDataset, stmgm_collate_fn


# ============================================================
# Metrics
# ============================================================

def find_eer_threshold(targets, probs):
    """Find threshold at Equal Error Rate (where FPR == FNR)."""
    fpr, tpr, thresholds = roc_curve(targets, probs)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    return float(thresholds[eer_idx])


def compute_mard(y_true, y_pred):
    """
    Mean Absolute Relative Difference (MARD).
    MARD = mean(|y_pred - y_true| / y_true) * 100%
    """
    mask = y_true > 0
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs(y_pred[mask] - y_true[mask]) / y_true[mask]) * 100)


def evaluate_fold(model, dataloader, device):
    """Evaluate a single fold and return all metrics."""
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

            cls_pred, reg_pred = model(f_temporal, ecg_beat, ppg_pulse, eda)

            all_cls_preds.extend(cls_pred.squeeze().cpu().numpy().tolist())
            all_cls_labels.extend(hypo_label.numpy().tolist())
            all_reg_preds.extend(reg_pred.squeeze().cpu().numpy().tolist())
            all_reg_labels.extend(glucose.numpy().tolist())

    cls_preds = np.array(all_cls_preds)
    cls_labels = np.array(all_cls_labels)
    reg_preds = np.array(all_reg_preds)
    reg_labels = np.array(all_reg_labels)

    # Classification metrics
    try:
        threshold = find_eer_threshold(cls_labels, cls_preds)
    except Exception:
        threshold = 0.5

    cls_binary = (cls_preds >= threshold).astype(int)

    try:
        auc = roc_auc_score(cls_labels, cls_preds)
    except Exception:
        auc = 0.5

    acc = accuracy_score(cls_labels, cls_binary)
    sens = recall_score(cls_labels, cls_binary, zero_division=0)
    spec = recall_score(1 - cls_labels, 1 - cls_binary, zero_division=0)
    prec = precision_score(cls_labels, cls_binary, zero_division=0)
    f1 = f1_score(cls_labels, cls_binary, zero_division=0)

    # Regression metrics
    rmse = float(np.sqrt(mean_squared_error(reg_labels, reg_preds)))
    mae = float(mean_absolute_error(reg_labels, reg_preds))
    mard = compute_mard(reg_labels, reg_preds)

    return {
        # Classification
        'auc': float(auc),
        'acc': float(acc),
        'sensitivity': float(sens),
        'specificity': float(spec),
        'precision': float(prec),
        'f1': float(f1),
        'eer_threshold': threshold,
        # Regression
        'rmse': rmse,
        'mae': mae,
        'mard': mard,
        # Raw predictions
        'cls_preds': cls_preds.tolist(),
        'cls_labels': cls_labels.tolist(),
        'reg_preds': reg_preds.tolist(),
        'reg_labels': reg_labels.tolist(),
    }


# ============================================================
# Main Evaluation
# ============================================================

def test_subject(subject, data_dir='./data', results_dir='./results'):
    """Evaluate all folds for a subject."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    meta_path = os.path.join(data_dir, subject, 'metadata.json')
    segments_path = os.path.join(data_dir, subject, 'segments.pkl')
    temporal_path = os.path.join(data_dir, subject, 'temporal_features.pkl')
    model_dir = os.path.join(results_dir, subject)

    if not os.path.exists(meta_path):
        print(f"ERROR: Metadata not found for {subject}")
        return

    with open(meta_path, 'r') as f:
        metadata = json.load(f)

    with open(segments_path, 'rb') as f:
        segments = pickle.load(f)

    with open(temporal_path, 'rb') as f:
        temporal_data = pickle.load(f)

    temporal_selected = temporal_data['selected']
    temporal_dim = temporal_selected.shape[1]
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

        val_indices = fold_info.get('val', [])
        val_segments = [segments[i] for i in val_indices]
        val_temporal = temporal_selected[val_indices]

        val_dataset = STMGMDataset(val_segments, val_temporal, augment=False, verbose=False)
        val_loader = DataLoader(
            val_dataset, batch_size=64, shuffle=False,
            collate_fn=stmgm_collate_fn, num_workers=0,
        )

        model = STMGMModel(temporal_dim=temporal_dim).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))

        results = evaluate_fold(model, val_loader, device)

        # Print results
        print(f"  Classification: AUC={results['auc']:.4f}, F1={results['f1']:.4f}, "
              f"Sens={results['sensitivity']:.4f}, Spec={results['specificity']:.4f}, "
              f"Acc={results['acc']:.4f}")
        print(f"  Regression:     RMSE={results['rmse']:.2f} mg/dL, "
              f"MAE={results['mae']:.2f} mg/dL, MARD={results['mard']:.2f}%")
        print(f"  EER Threshold:  {results['eer_threshold']:.4f}")

        # Store without raw predictions for summary
        metrics_list.append({
            'auc': results['auc'],
            'f1': results['f1'],
            'sensitivity': results['sensitivity'],
            'specificity': results['specificity'],
            'acc': results['acc'],
            'rmse': results['rmse'],
            'mae': results['mae'],
            'mard': results['mard'],
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

        for metric in ['auc', 'f1', 'sensitivity', 'specificity', 'acc', 'rmse', 'mae', 'mard']:
            values = [m[metric] for m in metrics_list]
            unit = ' mg/dL' if metric in ['rmse', 'mae'] else '%' if metric == 'mard' else ''
            print(f"  {metric.upper():12s}: {np.mean(values):.4f} ± {np.std(values):.4f}{unit}")

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
    parser = argparse.ArgumentParser(description='ST-MGM: Evaluation')
    parser.add_argument('-s', '--subject', type=str, required=True, help='Subject ID')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--results_dir', type=str, default='./results')
    args = parser.parse_args()

    test_subject(args.subject, args.data_dir, args.results_dir)
