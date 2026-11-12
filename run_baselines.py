"""
ST-MGM: Benchmark Baseline Models for Table 7
=============================================
Evaluates standard comparative baselines on the same 5-fold CV splits:
1. Support Vector Machine (SVM Regressor + SVC Classifier)
2. Random Forest (RandomForestRegressor + RandomForestClassifier)
3. Quamer et al. [23] (Nature Sci Data 2025):
   Late-fusion architecture (InceptionTime ECG + InceptionTime PPG + BiLSTM EDA)
   combined via Logistic Regression for classification, plus Linear Regression head for continuous BG.

Usage:
    python run_baselines.py --subject c1s01_raw
    python run_baselines.py --subject c1s01_raw --models svm rf quamer
"""

import os
import sys
import gc
import json
import pickle
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    f1_score, recall_score, precision_score,
    mean_squared_error, mean_absolute_error
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test import (
    compute_mard,
    compute_parkes_error_grid,
    compute_sens_at_spec,
    compute_pooled_oof_metrics,
    find_eer_threshold,
    seed_everything,
)
from preprocess_features import build_blocked_splits


# ============================================================
# Quamer et al. Architecture Modules
# ============================================================

class InceptionBlockPlus(nn.Module):
    """InceptionTime block for 1D physiological time-series."""
    def __init__(self, in_channels, out_channels=32):
        super().__init__()
        self.bottleneck = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv10 = nn.Conv1d(out_channels, out_channels, kernel_size=11, padding=5, bias=False)
        self.conv20 = nn.Conv1d(out_channels, out_channels, kernel_size=21, padding=10, bias=False)
        self.conv40 = nn.Conv1d(out_channels, out_channels, kernel_size=41, padding=20, bias=False)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.pool_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels * 4)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        b = self.bottleneck(x)
        c1 = self.conv10(b)
        c2 = self.conv20(b)
        c3 = self.conv40(b)
        p = self.pool_conv(self.pool(x))
        out = torch.cat([c1, c2, c3, p], dim=1)
        out = self.dropout(F.relu(self.bn(out)))
        return out


class InceptionTimeNet(nn.Module):
    """6-block InceptionTime network matching Quamer et al."""
    def __init__(self, in_channels=1, output_dim=32):
        super().__init__()
        self.block1 = InceptionBlockPlus(in_channels, 32)
        self.block2 = InceptionBlockPlus(128, 32)
        self.block3 = InceptionBlockPlus(128, 32)
        self.block4 = InceptionBlockPlus(128, 32)
        self.block5 = InceptionBlockPlus(128, 32)
        self.block6 = InceptionBlockPlus(128, 32)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, output_dim)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = (x - x.mean(dim=-1, keepdim=True)) / x.std(dim=-1, keepdim=True).clamp(min=1e-6)
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2) + x1  # Residual at layer 3
        x4 = self.block4(x3)
        x5 = self.block5(x4)
        x6 = self.block6(x5) + x3  # Residual at layer 6
        pooled = self.gap(x6).squeeze(-1)
        return self.fc(pooled)


class QuamerEDANet(nn.Module):
    """Conv1D + BiLSTM EDA model matching Quamer et al."""
    def __init__(self, output_dim=32):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 4, kernel_size=12, stride=6, padding=3)
        self.bn1 = nn.BatchNorm1d(4)
        self.conv2 = nn.Conv1d(4, 8, kernel_size=8, stride=5, padding=2)
        self.bn2 = nn.BatchNorm1d(8)
        self.lstm = nn.LSTM(input_size=8, hidden_size=16, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(32, output_dim)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.transpose(1, 2)
        _, (h_n, _) = self.lstm(x)
        h = torch.cat([h_n[-2], h_n[-1]], dim=1)
        return self.fc(h)


class QuamerFullLateFusion(nn.Module):
    """Late-fusion multi-modal network from Quamer et al. [23]."""
    def __init__(self):
        super().__init__()
        self.ecg_net = InceptionTimeNet(in_channels=1, output_dim=16)
        self.ppg_net = InceptionTimeNet(in_channels=1, output_dim=16)
        self.eda_net = QuamerEDANet(output_dim=16)
        self.fc_fusion = nn.Linear(48, 16)
        self.cls_head = nn.Linear(16, 1)
        self.reg_head = nn.Linear(16, 1)

    def forward(self, ecg, ppg, eda):
        h_ecg = self.ecg_net(ecg)
        h_ppg = self.ppg_net(ppg)
        h_eda = self.eda_net(eda)
        f = torch.cat([h_ecg, h_ppg, h_eda], dim=1)
        h_f = F.relu(self.fc_fusion(f))
        cls_logits = self.cls_head(h_f)
        reg_val = self.reg_head(h_f)
        return cls_logits, reg_val


# ============================================================
# Baseline Runners
# ============================================================

def evaluate_predictions(
    y_true_reg,
    y_pred_reg,
    y_true_cls,
    y_pred_prob,
    fixed_thr=None,
    train_glucose_mean=None,
    train_glucose_std=None,
):
    """Compute standard metrics package including Training-Mean baseline and variance-normalized metrics."""
    y_true_reg = np.asarray(y_true_reg, dtype=float)
    y_pred_reg = np.asarray(y_pred_reg, dtype=float)
    y_true_cls = np.asarray(y_true_cls, dtype=int)
    y_pred_prob = np.asarray(y_pred_prob, dtype=float)

    if fixed_thr is not None and np.isfinite(fixed_thr):
        thr = float(np.clip(fixed_thr, 0.20, 0.80))
    else:
        thr = find_eer_threshold(y_true_cls, y_pred_prob)

    y_pred_bin = (y_pred_prob >= thr).astype(int)

    try:
        auc = float(roc_auc_score(y_true_cls, y_pred_prob)) if len(np.unique(y_true_cls)) > 1 else 0.5
    except Exception:
        auc = 0.5

    try:
        auprc = float(average_precision_score(y_true_cls, y_pred_prob)) if len(np.unique(y_true_cls)) > 1 else 0.0
    except Exception:
        auprc = 0.0

    sens = float(recall_score(y_true_cls, y_pred_bin, zero_division=0))
    spec = float(recall_score(1 - y_true_cls, 1 - y_pred_bin, zero_division=0))
    sens_at_90spec = compute_sens_at_spec(y_true_cls, y_pred_prob, target_spec=0.90)
    balanced_acc = 0.5 * (sens + spec)

    rmse = float(np.sqrt(mean_squared_error(y_true_reg, y_pred_reg)))
    mae = float(mean_absolute_error(y_true_reg, y_pred_reg))
    mard = compute_mard(y_true_reg, y_pred_reg)
    try:
        if len(y_true_reg) > 1 and np.std(y_true_reg) > 1e-8 and np.std(y_pred_reg) > 1e-8:
            r_val = float(np.corrcoef(y_true_reg, y_pred_reg)[0, 1])
            pearson_r = r_val if np.isfinite(r_val) else 0.0
        else:
            pearson_r = 0.0
    except Exception:
        pearson_r = 0.0
    parkes = compute_parkes_error_grid(y_true_reg, y_pred_reg)

    g_mean = float(train_glucose_mean) if train_glucose_mean is not None else float(np.mean(y_true_reg))
    g_std = float(max(train_glucose_std, 1e-8)) if train_glucose_std is not None else float(max(np.std(y_true_reg), 1e-8))
    mean_preds = np.full_like(y_true_reg, g_mean, dtype=np.float64)
    rmse_mean_pred = float(np.sqrt(mean_squared_error(y_true_reg, mean_preds)))
    mae_mean_pred = float(mean_absolute_error(y_true_reg, mean_preds))
    mard_mean_pred = compute_mard(y_true_reg, mean_preds)
    parkes_mean = compute_parkes_error_grid(y_true_reg, mean_preds)

    r2 = float(1.0 - (rmse ** 2) / (g_std ** 2))
    r2_oos = float(1.0 - (rmse ** 2) / max(rmse_mean_pred ** 2, 1e-8))
    rmse_sd_ratio = float(rmse / g_std)

    return {
        'auc': auc,
        'auprc': auprc,
        'acc': float(accuracy_score(y_true_cls, y_pred_bin)),
        'balanced_acc': balanced_acc,
        'sensitivity': sens,
        'specificity': spec,
        'sens_at_90spec': sens_at_90spec,
        'precision': float(precision_score(y_true_cls, y_pred_bin, zero_division=0)),
        'f1': float(f1_score(y_true_cls, y_pred_bin, zero_division=0)),
        'eer_threshold': thr,
        'rmse': rmse,
        'mae': mae,
        'mard': mard,
        'pearson_r': pearson_r,
        'parkes_zone_a': parkes['A'],
        'parkes_zone_b': parkes['B'],
        'parkes_ab': parkes['AB'],
        'train_glucose_mean': g_mean,
        'train_glucose_std': g_std,
        'rmse_mean_pred': rmse_mean_pred,
        'mae_mean_pred': mae_mean_pred,
        'mard_mean_pred': mard_mean_pred,
        'parkes_ab_mean_pred': float(parkes_mean['AB']),
        'r2': r2,
        'r2_oos': r2_oos,
        'rmse_sd_ratio': rmse_sd_ratio,
        'cls_preds': y_pred_prob.tolist(),
        'cls_binary': y_pred_bin.tolist(),
        'cls_labels': y_true_cls.tolist(),
        'reg_preds': y_pred_reg.tolist(),
        'reg_labels': y_true_reg.tolist(),
        'mean_preds': mean_preds.tolist(),
    }


def run_tabular_baseline(model_type, folds, X_matrix, y_reg_all, y_cls_all):
    """Run SVM or Random Forest across 5 folds."""
    results = {}
    for fold_name, fold_info in folds.items():
        seed_everything(42)
        tr_idx = fold_info['train']
        va_idx = fold_info['val']

        X_tr_raw = np.nan_to_num(X_matrix[tr_idx], nan=0.0, posinf=0.0, neginf=0.0)
        X_va_raw = np.nan_to_num(X_matrix[va_idx], nan=0.0, posinf=0.0, neginf=0.0)

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr_raw)
        X_va = scaler.transform(X_va_raw)

        y_tr_reg = y_reg_all[tr_idx]
        y_va_reg = y_reg_all[va_idx]
        y_tr_cls = y_cls_all[tr_idx]
        y_va_cls = y_cls_all[va_idx]

        g_mean = float(np.mean(y_tr_reg))
        g_std = float(max(np.std(y_tr_reg), 1.0))

        if model_type == 'svm':
            reg = SVR(C=10.0, epsilon=0.1)
            clf = SVC(C=5.0, kernel='rbf', probability=True, class_weight='balanced', random_state=42)
        elif model_type == 'rf':
            reg = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
            clf = RandomForestClassifier(n_estimators=100, max_depth=12, class_weight='balanced', random_state=42, n_jobs=-1)

        reg.fit(X_tr, y_tr_reg)
        pred_reg = reg.predict(X_va)

        if len(np.unique(y_tr_cls)) >= 2:
            clf.fit(X_tr, y_tr_cls)
            prob_cls = clf.predict_proba(X_va)[:, 1]
            tr_prob = clf.predict_proba(X_tr)[:, 1]
            thr = find_eer_threshold(y_tr_cls, tr_prob)
        else:
            # Fallback if a blocked fold has only 1 class in training
            prob_cls = 1.0 / (1.0 + np.exp((pred_reg - 70.0) / 10.0))
            thr = 0.5

        metrics = evaluate_predictions(
            y_va_reg, pred_reg, y_va_cls, prob_cls,
            fixed_thr=thr, train_glucose_mean=g_mean, train_glucose_std=g_std,
        )
        results[fold_name] = metrics

    return results


def run_quamer_baseline(folds, segments, device, epochs=40, lr=0.0005):
    """Train and evaluate Quamer et al. [23] late fusion baseline."""
    class QuamerDataset(Dataset):
        def __init__(self, segs, hypo_thr=70.0):
            self.segs = segs
            self.ecg = [s['ecg_beat'] for s in segs]
            self.ppg = [s['ppg_pulse'] for s in segs]
            self.eda = [s['eda'] for s in segs]
            self.glucose = np.array([s['glucose'] for s in segs], dtype=np.float32)
            self.labels = (self.glucose < hypo_thr).astype(np.float32)

        def __len__(self):
            return len(self.segs)

        def __getitem__(self, i):
            return {
                'ecg': torch.tensor(self.ecg[i], dtype=torch.float32),
                'ppg': torch.tensor(self.ppg[i], dtype=torch.float32),
                'eda': torch.tensor(self.eda[i], dtype=torch.float32),
                'glucose': torch.tensor(self.glucose[i], dtype=torch.float32),
                'label': torch.tensor(self.labels[i], dtype=torch.float32),
            }

    results = {}
    for fold_name, fold_info in folds.items():
        seed_everything(42)
        tr_idx = fold_info['train']
        va_idx = fold_info['val']

        tr_segs = [segments[i] for i in tr_idx]
        va_segs = [segments[i] for i in va_idx]

        tr_ds = QuamerDataset(tr_segs)
        va_ds = QuamerDataset(va_segs)

        g = torch.Generator()
        g.manual_seed(42)
        tr_loader = DataLoader(tr_ds, batch_size=32, shuffle=True, generator=g)
        tr_eval_loader = DataLoader(tr_ds, batch_size=64, shuffle=False)
        va_loader = DataLoader(va_ds, batch_size=64, shuffle=False)

        model = QuamerFullLateFusion().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

        y_tr_cls = tr_ds.labels
        pos_w = float(np.clip(np.sqrt((len(y_tr_cls) - y_tr_cls.sum()) / max(y_tr_cls.sum(), 1)), 1.0, 10.0))
        bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w], device=device))
        mse = nn.MSELoss()

        g_mean = float(np.mean(tr_ds.glucose))
        g_std = float(max(np.std(tr_ds.glucose), 1.0))

        best_score = float('inf')
        best_state = None

        for ep in range(1, epochs + 1):
            model.train()
            for b in tr_loader:
                ecg = b['ecg'].to(device)
                ppg = b['ppg'].to(device)
                eda = b['eda'].to(device)
                lbl = b['label'].to(device).unsqueeze(1)
                glu = ((b['glucose'].to(device) - g_mean) / g_std).unsqueeze(1)

                opt.zero_grad()
                c_pred, r_pred = model(ecg, ppg, eda)
                loss = 0.5 * bce(c_pred, lbl) + 0.5 * mse(r_pred, glu)
                loss.backward()
                opt.step()

            # Simple eval
            model.eval()
            val_loss = 0.0
            n_val = 0
            with torch.no_grad():
                for b in va_loader:
                    ecg = b['ecg'].to(device)
                    ppg = b['ppg'].to(device)
                    eda = b['eda'].to(device)
                    lbl = b['label'].to(device).unsqueeze(1)
                    glu = ((b['glucose'].to(device) - g_mean) / g_std).unsqueeze(1)
                    c_pred, r_pred = model(ecg, ppg, eda)
                    l = 0.5 * bce(c_pred, lbl) + 0.5 * mse(r_pred, glu)
                    val_loss += l.item() * len(lbl)
                    n_val += len(lbl)
            val_loss /= n_val
            if val_loss < best_score:
                best_score = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict(best_state)

        # Calibrate classification threshold on training fold
        model.eval()
        tr_preds_prob = []
        with torch.no_grad():
            for b in tr_eval_loader:
                ecg = b['ecg'].to(device)
                ppg = b['ppg'].to(device)
                eda = b['eda'].to(device)
                c_pred, _ = model(ecg, ppg, eda)
                tr_preds_prob.extend(torch.sigmoid(c_pred).reshape(-1).cpu().numpy().tolist())
        tr_thr = find_eer_threshold(tr_ds.labels, tr_preds_prob)

        # Final predictions on validation set
        va_preds_reg, va_preds_prob = [], []
        with torch.no_grad():
            for b in va_loader:
                ecg = b['ecg'].to(device)
                ppg = b['ppg'].to(device)
                eda = b['eda'].to(device)
                c_pred, r_pred = model(ecg, ppg, eda)
                va_preds_prob.extend(torch.sigmoid(c_pred).reshape(-1).cpu().numpy().tolist())
                r_mgdl = r_pred * g_std + g_mean
                va_preds_reg.extend(r_mgdl.reshape(-1).cpu().numpy().tolist())

        metrics = evaluate_predictions(
            va_ds.glucose, va_preds_reg, va_ds.labels, va_preds_prob,
            fixed_thr=tr_thr, train_glucose_mean=g_mean, train_glucose_std=g_std,
        )
        results[fold_name] = metrics
        del model, opt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    return results


def summarize_results(res_dict):
    keys = [
        'auc', 'auprc', 'acc', 'balanced_acc', 'f1',
        'sensitivity', 'specificity', 'sens_at_90spec',
        'rmse', 'rmse_mean_pred', 'r2', 'r2_oos', 'rmse_sd_ratio',
        'mae', 'mard', 'pearson_r', 'parkes_ab'
    ]
    summary = {}
    for k in keys:
        vals = [v[k] for v in res_dict.values() if isinstance(v, dict) and k in v]
        if vals:
            summary[k] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}
    return summary


# ============================================================
# CLI
# ============================================================

def main():
    seed_everything(42)
    parser = argparse.ArgumentParser(description='ST-MGM: Run Baselines for Table 7')
    parser.add_argument('-s', '--subject', type=str, required=True, help='Subject ID')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--results_dir', type=str, default='./results')
    parser.add_argument('--models', nargs='+', default=['svm', 'rf', 'quamer'], choices=['svm', 'rf', 'quamer'])
    parser.add_argument('--cv_mode', type=str, default='stratified', choices=['stratified', 'blocked'],
                        help='Cross-validation split mode (stratified or blocked)')
    parser.add_argument('--purge_gap', type=int, default=6,
                        help='Temporal purge gap (windows) around validation block in blocked CV')
    args = parser.parse_args()

    subject = args.subject
    cv_mode = args.cv_mode.lower()
    meta_path = os.path.join(args.data_dir, subject, 'metadata.json')
    segments_path = os.path.join(args.data_dir, subject, 'segments.pkl')
    temporal_path = os.path.join(args.data_dir, subject, 'temporal_features.pkl')

    with open(meta_path, 'r') as f:
        metadata = json.load(f)
    with open(segments_path, 'rb') as f:
        segments = pickle.load(f)
    with open(temporal_path, 'rb') as f:
        temporal_data = pickle.load(f)

    hypo_thr = float(metadata.get('hypo_threshold', 70))
    X_temporal = temporal_data.get('raw', temporal_data.get('selected'))
    y_reg = np.array([s['glucose'] for s in segments], dtype=np.float32)
    y_cls = (y_reg < hypo_thr).astype(int)

    if cv_mode == 'blocked':
        if 'blocked_folds' in metadata and isinstance(metadata['blocked_folds'], dict):
            folds = metadata['blocked_folds']
        else:
            raw_blocked_splits = build_blocked_splits(
                segments, y_cls, n_folds=5, purge_gap=args.purge_gap
            )
            folds = {
                f"v{i + 1}": {'train': tr.tolist(), 'val': va.tolist()}
                for i, (tr, va) in enumerate(raw_blocked_splits)
            }
    else:
        folds = metadata.get('stratified_folds', metadata.get('folds', metadata))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n========================================================")
    print(f"  Running Baselines for {subject} (cv_mode={cv_mode}) on {device}")
    print(f"========================================================")

    raw_keys = {'cls_preds', 'cls_binary', 'cls_labels', 'reg_preds', 'reg_labels', 'mean_preds'}

    for m in args.models:
        folder_name = f"{subject}_baseline_{m}" if cv_mode == 'stratified' else f"{subject}_baseline_{m}_blocked"
        out_folder = os.path.join(args.results_dir, folder_name)
        os.makedirs(out_folder, exist_ok=True)
        print(f"\n--- Running Baseline: {m.upper()} ({cv_mode}) ---")

        if m in ['svm', 'rf']:
            res = run_tabular_baseline(m, folds, X_temporal, y_reg, y_cls)
        elif m == 'quamer':
            res = run_quamer_baseline(folds, segments, device)

        pooled_oof = compute_pooled_oof_metrics(res)
        summ = summarize_results(res)
        print(f"  Results for {m.upper()}:")
        print(f"    AUC:        {summ['auc']['mean']:.4f} ± {summ['auc']['std']:.4f}")
        print(f"    AUPRC:      {summ['auprc']['mean']:.4f} ± {summ['auprc']['std']:.4f}")
        print(f"    F1:         {summ['f1']['mean']:.4f} ± {summ['f1']['std']:.4f}")
        print(f"    BalAcc:     {summ['balanced_acc']['mean']:.4f} ± {summ['balanced_acc']['std']:.4f}")
        print(f"    Sens@90Sp:  {summ['sens_at_90spec']['mean']:.4f} ± {summ['sens_at_90spec']['std']:.4f}")
        print(f"    RMSE:       {summ['rmse']['mean']:.2f} ± {summ['rmse']['std']:.2f} mg/dL (R2={summ['r2']['mean']:.4f}, RMSE/SD={summ['rmse_sd_ratio']['mean']:.4f})")
        print(f"    MARD:       {summ['mard']['mean']:.2f} ± {summ['mard']['std']:.2f}%")
        print(f"    Pearson r:  {summ['pearson_r']['mean']:.4f} ± {summ['pearson_r']['std']:.4f}")
        print(f"    Parkes AB:  {summ['parkes_ab']['mean']:.2f} ± {summ['parkes_ab']['std']:.2f}%")

        save_res = {}
        for fold_k, fold_v in res.items():
            save_res[fold_k] = {k: v for k, v in fold_v.items() if k not in raw_keys}
        if pooled_oof:
            save_res['pooled_oof'] = pooled_oof

        out_file = os.path.join(out_folder, 'evaluation_results.json')
        with open(out_file, 'w') as f:
            json.dump(save_res, f, indent=2)
        print(f"  Saved -> {out_file}")


if __name__ == '__main__':
    main()

