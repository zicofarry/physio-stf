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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from libs.st_mgm_model import STMGMModel, STMGMFusionOnly, MultiTaskLoss
from libs.dataloader import STMGMDataset, FeatureDataset, stmgm_collate_fn


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

def train_one_epoch_e2e(model, dataloader, optimizer, criterion, device):
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
        cls_pred, reg_pred = model(f_temporal, ecg_beat, ppg_pulse, eda)

        loss, cls_loss, reg_loss = criterion(cls_pred, hypo_label, reg_pred, glucose)
        loss.backward()
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

            cls_pred, reg_pred = model(f_temporal, ecg_beat, ppg_pulse, eda)
            loss, cls_loss, reg_loss = criterion(cls_pred, hypo_label, reg_pred, glucose)

            bs = len(hypo_label)
            total_loss += loss.item() * bs
            total_cls_loss += cls_loss.item() * bs
            total_reg_loss += reg_loss.item() * bs
            n_samples += bs

            all_cls_preds.extend(cls_pred.squeeze().cpu().numpy().tolist())
            all_cls_labels.extend(hypo_label.cpu().numpy().tolist())
            all_reg_preds.extend(reg_pred.squeeze().cpu().numpy().tolist())
            all_reg_labels.extend(glucose.cpu().numpy().tolist())

    return {
        'loss': total_loss / n_samples,
        'cls_loss': total_cls_loss / n_samples,
        'reg_loss': total_reg_loss / n_samples,
        'cls_preds': np.array(all_cls_preds),
        'cls_labels': np.array(all_cls_labels),
        'reg_preds': np.array(all_reg_preds),
        'reg_labels': np.array(all_reg_labels),
    }


def train_one_epoch_fusion(model, dataloader, optimizer, criterion, device):
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
        cls_pred, reg_pred = model(f_temporal, f_morph, f_eda)
        loss, _, _ = criterion(cls_pred, hypo_label, reg_pred, glucose)
        loss.backward()
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

    temporal_selected = temporal_data['selected']
    selected_indices = temporal_data['selected_indices']
    temporal_dim = temporal_selected.shape[1]

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

    print(f"\nTraining ST-MGM ({mode}) for {subject}")
    print(f"  Temporal dim: {temporal_dim}, Folds: {list(folds.keys())}")
    print(f"  LR: {lr}, Batch: {batch_size}, Epochs: {epochs}, Patience: {patience}")
    print(f"  Loss weights: alpha={alpha} (BCE), beta={beta} (MSE)")

    all_fold_results = {}

    for fold_name, fold_info in folds.items():
        print(f"\n{'='*50}")
        print(f"  Fold {fold_name}")
        print(f"{'='*50}")

        set_seed(42)

        train_indices = fold_info.get('train', [])
        val_indices = fold_info.get('val', [])

        # Create datasets
        train_segments = [segments[i] for i in train_indices]
        val_segments = [segments[i] for i in val_indices]
        train_temporal = temporal_selected[train_indices]
        val_temporal = temporal_selected[val_indices]

        train_dataset = STMGMDataset(
            train_segments, train_temporal, augment=True, verbose=True
        )
        val_dataset = STMGMDataset(
            val_segments, val_temporal, augment=False, verbose=True
        )

        g = torch.Generator()
        g.manual_seed(42)

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            collate_fn=stmgm_collate_fn,
            num_workers=0, worker_init_fn=seed_worker, generator=g,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=stmgm_collate_fn,
            num_workers=0,
        )

        # Create model
        model = STMGMModel(temporal_dim=temporal_dim).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = MultiTaskLoss(alpha=alpha, beta=beta)

        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        for epoch in range(1, epochs + 1):
            train_metrics = train_one_epoch_e2e(model, train_loader, optimizer, criterion, device)
            val_metrics = evaluate_e2e(model, val_loader, criterion, device)
            scheduler.step()

            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
                print(f"  Epoch {epoch:3d}/{epochs}: "
                      f"train_loss={train_metrics['loss']:.4f} "
                      f"(cls={train_metrics['cls_loss']:.4f}, reg={train_metrics['reg_loss']:.4f}) | "
                      f"val_loss={val_metrics['loss']:.4f}")

            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

        # Save best model
        model_path = os.path.join(save_dir, f'{fold_name}.pt')
        if best_model_state is not None:
            torch.save(best_model_state, model_path)
            print(f"  Saved best model → {model_path} (val_loss={best_val_loss:.4f})")

        all_fold_results[fold_name] = {
            'best_val_loss': best_val_loss,
            'epochs_trained': epoch,
        }

        # Cleanup
        del model, optimizer, train_loader, val_loader
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
