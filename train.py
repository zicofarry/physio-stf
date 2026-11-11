"""
ST-MGM: Phase 4 — Multi-Task Training Loop
=============================================
Trains the ST-MGM fusion model with combined loss:
    L = alpha * BCE(classification) + beta * MSE(regression)

Supports:
    - End-to-end training (raw beats/pulses → features → predictions)
    - Fusion-only training (pre-extracted features → predictions)
    - k-fold cross-validation
    - Early stopping with patience
    - Learning rate scheduling (cosine annealing)

Usage:
    python train.py --subject c1s01_raw_single
    python train.py --subject c1s01_raw_single --mode fusion_only
"""

import os
import sys
import gc
import json
import yaml
import pickle
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from libs.st_mgm_model import STMGMModel, STMGMFusionOnly, MultiTaskLoss
from libs.dataloader import STMGMDataset, FeatureDataset, stmgm_collate_fn
from libs.dwt_features import combined_feature_selection


def find_eer_threshold(targets, probs):
    """
    Find optimal probability threshold balancing F1-score and Youden's J (TPR - FPR)
    with finite-value guard.
    """
    targets = np.asarray(targets).reshape(-1)
    probs = np.asarray(probs).reshape(-1)
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
    tp = tpr * n_pos
    fp = fpr * n_neg
    fn = (1.0 - tpr) * n_pos
    f1_scores = (2.0 * tp) / np.maximum(2.0 * tp + fp + fn, 1e-8)
    youden_j = tpr - fpr

    combined_score = 0.5 * f1_scores + 0.5 * youden_j
    best_idx = int(np.nanargmax(combined_score))
    thr = float(thresholds[best_idx])
    return float(np.clip(thr, 1e-6, 1.0 - 1e-6))


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed=42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    """Ensure reproducibility in DataLoader workers."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ============================================================
# Training Functions
# ============================================================

def train_one_epoch_e2e(model, dataloader, optimizer, criterion, device, max_grad_norm=1.0):
    """Train one epoch — end-to-end mode."""
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_reg_loss = 0.0
    n_samples = 0

    for batch in dataloader:
        f_temporal = batch['f_temporal'].to(device)
        ecg_beat = batch['ecg_beat'].to(device)
        ppg_pulse = batch['ppg_pulse'].to(device)
        eda = batch['eda'].to(device)
        hypo_label = batch['hypo_label'].to(device)
        glucose = batch['glucose'].to(device)

        optimizer.zero_grad()
        cls_logits, reg_pred = model(f_temporal, ecg_beat, ppg_pulse, eda)

        loss, cls_loss, reg_loss = criterion(cls_logits, hypo_label, reg_pred, glucose)
        loss.backward()
        if max_grad_norm is not None and max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()

        bs = len(hypo_label)
        total_loss += loss.item() * bs
        total_cls_loss += cls_loss.item() * bs
        total_reg_loss += reg_loss.item() * bs
        n_samples += bs

    return {
        'loss': total_loss / n_samples,
        'cls_loss': total_cls_loss / n_samples,
        'reg_loss': total_reg_loss / n_samples,
    }


def evaluate_e2e(model, dataloader, criterion, device):
    """Evaluate — end-to-end mode."""
    model.eval()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_reg_loss = 0.0
    n_samples = 0

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
            hypo_label = batch['hypo_label'].to(device)
            glucose = batch['glucose'].to(device)

            cls_logits, reg_pred = model(f_temporal, ecg_beat, ppg_pulse, eda)
            loss, cls_loss, reg_loss = criterion(cls_logits, hypo_label, reg_pred, glucose)
            cls_prob = torch.sigmoid(cls_logits)

            bs = len(hypo_label)
            total_loss += loss.item() * bs
            total_cls_loss += cls_loss.item() * bs
            total_reg_loss += reg_loss.item() * bs
            n_samples += bs

            all_cls_preds.extend(cls_prob.reshape(-1).cpu().numpy().tolist())
            all_cls_labels.extend(hypo_label.reshape(-1).cpu().numpy().tolist())
            all_reg_preds.extend(reg_pred.reshape(-1).cpu().numpy().tolist())
            all_reg_labels.extend(glucose.reshape(-1).cpu().numpy().tolist())

    return {
        'loss': total_loss / n_samples,
        'cls_loss': total_cls_loss / n_samples,
        'reg_loss': total_reg_loss / n_samples,
        'cls_preds': np.array(all_cls_preds),
        'cls_labels': np.array(all_cls_labels),
        'reg_preds': np.array(all_reg_preds),
        'reg_labels': np.array(all_reg_labels),
    }


def train_one_epoch_fusion(model, dataloader, optimizer, criterion, device, max_grad_norm=1.0):
    """Train one epoch — fusion-only mode."""
    model.train()
    total_loss = 0.0
    n_samples = 0

    for batch in dataloader:
        f_temporal = batch['f_temporal'].to(device)
        f_morph = batch['f_morph'].to(device)
        f_eda = batch['f_eda'].to(device)
        hypo_label = batch['hypo_label'].to(device)
        glucose = batch['glucose'].to(device)

        optimizer.zero_grad()
        cls_logits, reg_pred = model(f_temporal, f_morph, f_eda)
        loss, _, _ = criterion(cls_logits, hypo_label, reg_pred, glucose)
        loss.backward()
        if max_grad_norm is not None and max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()

        bs = len(hypo_label)
        total_loss += loss.item() * bs
        n_samples += bs

    return {'loss': total_loss / n_samples}


# ============================================================
# Main Training Pipeline
# ============================================================

def load_config():
    """Load training configuration from YAML."""
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'configs', 'st_mgm_config.yaml')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def train_subject(subject, data_dir='./data', out_dir='./results', mode='e2e'):
    """
    Train ST-MGM for a given subject.

    Args:
        subject: Subject ID
        data_dir: Path to feature data
        out_dir: Output directory for models
        mode: 'e2e' (end-to-end) or 'fusion_only'
    """
    set_seed(42)

    config = load_config()
    training_cfg = config.get('training', {})
    dwt_cfg = config.get('dwt', {})
    hypo_threshold = float(config.get('hypo_threshold', 70))
    n_selected_features = int(dwt_cfg.get('selected_features', 18))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Load data
    meta_path = os.path.join(data_dir, subject, 'metadata.json')
    segments_path = os.path.join(data_dir, subject, 'segments.pkl')
    temporal_path = os.path.join(data_dir, subject, 'temporal_features.pkl')

    if not os.path.exists(meta_path):
        print(f"ERROR: Metadata not found at {meta_path}")
        return

    with open(meta_path, 'r') as f:
        metadata = json.load(f)

    with open(segments_path, 'rb') as f:
        segments = pickle.load(f)

    with open(temporal_path, 'rb') as f:
        temporal_data = pickle.load(f)

    temporal_raw = temporal_data.get('raw')
    temporal_selected_fallback = temporal_data.get('selected')
    selected_indices_fallback = temporal_data.get('selected_indices')
    fold_selected_map = temporal_data.get(
        'fold_selected_indices',
        metadata.get('fold_selected_indices', {}),
    )
    hypo_threshold = float(metadata.get('hypo_threshold', hypo_threshold))

    folds = metadata.get('folds', metadata)
    save_dir = os.path.join(out_dir, subject)
    os.makedirs(save_dir, exist_ok=True)

    # Training hyperparameters
    batch_size = training_cfg.get('batch_size', 32)
    epochs = training_cfg.get('epochs', 100)
    patience = training_cfg.get('patience', 20)
    lr = training_cfg.get('lr', 0.001)
    weight_decay = training_cfg.get('weight_decay', 0.0001)
    alpha = training_cfg.get('loss_alpha', 0.5)
    beta = training_cfg.get('loss_beta', 0.5)
    max_grad_norm = float(training_cfg.get('max_grad_norm', 1.0))

    print(f"\nTraining ST-MGM ({mode}) for {subject}")
    print(f"  Target selected features: {n_selected_features}, Folds: {list(folds.keys())}")
    print(f"  LR: {lr}, Batch: {batch_size}, Epochs: {epochs}, Patience: {patience}")
    print(f"  Loss weights: alpha={alpha} (BCEWithLogits), beta={beta} (MSE), grad_clip={max_grad_norm}")

    all_fold_results = {}

    for fold_name, fold_info in folds.items():
        print(f"\n{'='*50}")
        print(f"  Fold {fold_name}")
        print(f"{'='*50}")

        set_seed(42)

        train_indices = fold_info.get('train', [])
        val_indices = fold_info.get('val', [])

        train_segments = [segments[i] for i in train_indices]
        val_segments = [segments[i] for i in val_indices]

        train_glucose = np.array([seg['glucose'] for seg in train_segments], dtype=np.float32)
        train_labels_cls = (train_glucose < hypo_threshold).astype(int)

        # 1. Leak-free per-fold feature selection on train_indices only
        if 'selected_indices' in fold_info and fold_info['selected_indices'] is not None:
            fold_sel_idx = np.array(fold_info['selected_indices'], dtype=int)
        elif fold_name in fold_selected_map:
            fold_sel_idx = np.array(fold_selected_map[fold_name], dtype=int)
        elif temporal_raw is not None:
            try:
                fold_sel_idx, _ = combined_feature_selection(
                    temporal_raw[train_indices],
                    train_labels_cls,
                    train_glucose,
                    n_final=n_selected_features,
                )
            except Exception as e:
                print(f"  WARNING: Per-fold feature selection failed ({e}), using fallback indices")
                fold_sel_idx = np.arange(min(n_selected_features, temporal_raw.shape[1]), dtype=int)
        else:
            fold_sel_idx = np.array(selected_indices_fallback, dtype=int)

        if temporal_raw is not None:
            train_temporal_raw = temporal_raw[train_indices][:, fold_sel_idx]
            val_temporal_raw = temporal_raw[val_indices][:, fold_sel_idx]
        else:
            train_temporal_raw = temporal_selected_fallback[train_indices]
            val_temporal_raw = temporal_selected_fallback[val_indices]

        temporal_dim = train_temporal_raw.shape[1]

        # 2. Fit StandardScaler on train fold f_temporal only, transform train & val
        feat_mean = np.mean(train_temporal_raw, axis=0).astype(np.float32)
        feat_std = np.std(train_temporal_raw, axis=0).astype(np.float32)
        feat_std = np.where(feat_std < 1e-8, 1.0, feat_std).astype(np.float32)

        train_temporal = np.nan_to_num(
            (train_temporal_raw - feat_mean) / feat_std, nan=0.0, posinf=0.0, neginf=0.0
        ).astype(np.float32)
        val_temporal = np.nan_to_num(
            (val_temporal_raw - feat_mean) / feat_std, nan=0.0, posinf=0.0, neginf=0.0
        ).astype(np.float32)

        # 3. Fit target glucose mean/std on train fold only
        glucose_mean = float(np.mean(train_glucose))
        glucose_std = float(np.std(train_glucose))
        if glucose_std < 1e-6:
            glucose_std = 1.0

        # 4. Compute tempered class imbalance pos_weight on train fold
        train_hypo_count = int(train_labels_cls.sum())
        train_normal_count = len(train_labels_cls) - train_hypo_count
        raw_ratio = train_normal_count / max(train_hypo_count, 1)
        pos_weight_val = float(np.clip(np.sqrt(raw_ratio), 1.0, 10.0))
        print(f"  Train glucose: mean={glucose_mean:.2f}, std={glucose_std:.2f} mg/dL | "
              f"pos_weight={pos_weight_val:.2f} (raw_ratio={raw_ratio:.1f})")

        # Create datasets with normalized target glucose
        train_dataset = STMGMDataset(
            train_segments, train_temporal, augment=True,
            hypo_threshold=hypo_threshold,
            glucose_mean=glucose_mean, glucose_std=glucose_std,
            verbose=True,
        )
        train_eval_dataset = STMGMDataset(
            train_segments, train_temporal, augment=False,
            hypo_threshold=hypo_threshold,
            glucose_mean=glucose_mean, glucose_std=glucose_std,
            verbose=False,
        )
        val_dataset = STMGMDataset(
            val_segments, val_temporal, augment=False,
            hypo_threshold=hypo_threshold,
            glucose_mean=glucose_mean, glucose_std=glucose_std,
            verbose=True,
        )

        g = torch.Generator()
        g.manual_seed(42)

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            collate_fn=stmgm_collate_fn,
            num_workers=0, worker_init_fn=seed_worker, generator=g,
        )
        train_eval_loader = DataLoader(
            train_eval_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=stmgm_collate_fn,
            num_workers=0,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=stmgm_collate_fn,
            num_workers=0,
        )

        # Create model & criterion with pos_weight and threshold-aware coupling
        model = STMGMModel(temporal_dim=temporal_dim)
        model.set_reg_threshold(
            hypo_threshold=hypo_threshold,
            glucose_mean=glucose_mean,
            glucose_std=glucose_std,
        )
        model = model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        pos_weight_tensor = torch.tensor([pos_weight_val], dtype=torch.float32, device=device)
        criterion = MultiTaskLoss(alpha=alpha, beta=beta, pos_weight=pos_weight_tensor).to(device)

        best_val_score = float('inf')
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        for epoch in range(1, epochs + 1):
            train_metrics = train_one_epoch_e2e(
                model, train_loader, optimizer, criterion, device, max_grad_norm=max_grad_norm
            )
            val_metrics = evaluate_e2e(model, val_loader, criterion, device)
            scheduler.step()

            try:
                if len(np.unique(val_metrics['cls_labels'])) > 1:
                    val_auc = float(roc_auc_score(val_metrics['cls_labels'], val_metrics['cls_preds']))
                else:
                    val_auc = 0.5
            except Exception:
                val_auc = 0.5

            # Combined validation criterion: normalized regression loss + (1 - AUC)
            val_score = float(val_metrics['reg_loss']) + 0.5 * (1.0 - val_auc)

            if val_score < best_val_score:
                best_val_score = val_score
                best_val_loss = float(val_metrics['loss'])
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
                print(f"  Epoch {epoch:3d}/{epochs}: "
                      f"train_loss={train_metrics['loss']:.4f} "
                      f"(cls={train_metrics['cls_loss']:.4f}, reg={train_metrics['reg_loss']:.4f}) | "
                      f"val_loss={val_metrics['loss']:.4f} "
                      f"(reg={val_metrics['reg_loss']:.4f}, auc={val_auc:.4f})")

            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

        # Compute optimal threshold on training fold using best model (no val leakage)
        train_eer_threshold = 0.5
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            train_eval_res = evaluate_e2e(model, train_eval_loader, criterion, device)
            train_eer_threshold = find_eer_threshold(
                train_eval_res['cls_labels'], train_eval_res['cls_preds']
            )

        # Save best model checkpoint with scaling & fold metadata
        model_path = os.path.join(save_dir, f'{fold_name}.pt')
        if best_model_state is not None:
            checkpoint_payload = {
                'model_state_dict': best_model_state,
                'selected_indices': fold_sel_idx.tolist(),
                'temporal_mean': feat_mean.tolist(),
                'temporal_std': feat_std.tolist(),
                'glucose_mean': glucose_mean,
                'glucose_std': glucose_std,
                'eer_threshold': float(train_eer_threshold),
                'pos_weight': pos_weight_val,
                'temporal_dim': temporal_dim,
            }
            torch.save(checkpoint_payload, model_path)
            print(f"  Saved best model → {model_path} "
                  f"(val_score={best_val_score:.4f}, val_loss={best_val_loss:.4f}, "
                  f"train_thr={train_eer_threshold:.4f})")

        all_fold_results[fold_name] = {
            'best_val_loss': float(best_val_loss),
            'best_val_score': float(best_val_score),
            'epochs_trained': int(epoch),
            'selected_indices': fold_sel_idx.tolist(),
            'temporal_mean': feat_mean.tolist(),
            'temporal_std': feat_std.tolist(),
            'glucose_mean': glucose_mean,
            'glucose_std': glucose_std,
            'eer_threshold': float(train_eer_threshold),
            'pos_weight': pos_weight_val,
        }

        # Cleanup
        del model, optimizer, train_loader, train_eval_loader, val_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # Save training summary
    summary_path = os.path.join(save_dir, 'training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_fold_results, f, indent=2)
    print(f"\nTraining summary saved to {summary_path}")
    print("\nTraining complete!")


# ============================================================
# CLI
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ST-MGM: Multi-Task Training')
    parser.add_argument('-s', '--subject', type=str, required=True, help='Subject ID')
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--out_dir', type=str, default='./results', help='Output directory')
    parser.add_argument('--mode', type=str, default='e2e', choices=['e2e', 'fusion_only'])
    args = parser.parse_args()

    train_subject(args.subject, args.data_dir, args.out_dir, args.mode)
