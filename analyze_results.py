"""
ST-MGM Phase 6B: Comprehensive Statistical & Multi-Subject Results Analyzer
===========================================================================
Aggregates all evaluation_results.json and training_summary.json across:
  1. Table 5: 10-Subject ST-MGM (M9) Stratified CV + Blocked CV (MJ-2) +
     Training-Mean Predictor comparison, R2, RMSE/SD, Sens@90%Spec (MJ-1, MJ-3),
     and Pooled OOF metrics for rare-event subjects (c1s02_raw, c1s04_raw).
  2. Table 6: Multi-Subject & Single-Subject Ablation Study (M1..M9_NOCOUPLE)
     with paired two-sided Wilcoxon Signed-Rank Tests (p-values) against M9 (MJ-4).
  3. Table 7: Multi-Subject & Single-Subject Baseline Comparison (SVM, RF, Quamer,
     Training-Mean Predictor vs M9) with paired Wilcoxon Signed-Rank Tests (MJ-4).
  4. Table 4: Leak-Free DWT Feature Selection Distribution across 50 Folds.

Outputs:
  - results/phase6b_analysis_summary.md
  - results/phase6b_analysis_summary.json

Usage:
    python analyze_results.py
    python analyze_results.py --results_dir ./results
"""

import os
import json
import argparse
import numpy as np
from scipy.stats import wilcoxon

ALL_SUBJECTS = [
    'c1s01_raw', 'c1s02_raw', 'c1s03_raw', 'c1s04_raw', 'c1s05_raw',
    'c2s01_raw', 'c2s02_raw', 'c2s03_raw', 'c2s04_raw', 'c2s05_raw',
]

ABLATION_VARIANTS = [
    'M1', 'M2', 'M3', 'M6', 'M7', 'M8', 'M9', 'M9_CLS', 'M9_REG', 'M9_NOCOUPLE'
]

BASELINE_MODELS = ['svm', 'rf', 'quamer']

FOLDS = ['v1', 'v2', 'v3', 'v4', 'v5']

METRICS = [
    'auc', 'auprc', 'f1', 'balanced_acc', 'sensitivity', 'specificity',
    'sens_at_90spec', 'acc', 'rmse', 'rmse_mean_pred', 'r2', 'r2_oos',
    'rmse_sd_ratio', 'mae', 'mard', 'pearson_r', 'parkes_ab'
]


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def enrich_fold_metrics(eval_dict, train_summary_dict=None):
    """
    Ensure every fold has r2, rmse_sd_ratio, and rmse_mean_pred even when reading
    Phase 6A evaluation_results.json files before test.py re-evaluation.
    """
    if eval_dict is None:
        return None
    enriched = {}
    for fk in FOLDS:
        if fk not in eval_dict or not isinstance(eval_dict[fk], dict):
            continue
        f_res = dict(eval_dict[fk])
        ts_fold = (train_summary_dict or {}).get(fk, {})
        g_mean = f_res.get('train_glucose_mean', ts_fold.get('glucose_mean'))
        g_std = f_res.get('train_glucose_std', ts_fold.get('glucose_std'))
        rmse = f_res.get('rmse')

        if g_std is not None and float(g_std) > 1e-8 and rmse is not None:
            g_std_f = float(g_std)
            rmse_f = float(rmse)
            f_res.setdefault('train_glucose_std', g_std_f)
            if g_mean is not None:
                f_res.setdefault('train_glucose_mean', float(g_mean))
            f_res.setdefault('rmse_mean_pred', g_std_f)
            f_res.setdefault('r2', float(1.0 - (rmse_f ** 2) / (g_std_f ** 2)))
            f_res.setdefault('r2_oos', float(1.0 - (rmse_f ** 2) / max(f_res.get('rmse_mean_pred', g_std_f) ** 2, 1e-8)))
            f_res.setdefault('rmse_sd_ratio', float(rmse_f / g_std_f))
        enriched[fk] = f_res

    if 'pooled_oof' in eval_dict and isinstance(eval_dict['pooled_oof'], dict):
        enriched['pooled_oof'] = dict(eval_dict['pooled_oof'])
    return enriched


def summarize_subject_folds(enriched_eval):
    """Compute mean and std across v1..v5 for a single subject/model."""
    if not enriched_eval:
        return None
    fold_dicts = [enriched_eval[fk] for fk in FOLDS if fk in enriched_eval]
    if not fold_dicts:
        return None

    out = {'n_folds': len(fold_dicts), 'folds': {}}
    for fk in FOLDS:
        if fk in enriched_eval:
            out['folds'][fk] = enriched_eval[fk]

    for m in METRICS:
        vals = [fd[m] for fd in fold_dicts if m in fd and fd[m] is not None and np.isfinite(fd[m])]
        if vals:
            out[m] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals)), 'values': [float(x) for x in vals]}
    if 'pooled_oof' in enriched_eval:
        out['pooled_oof'] = enriched_eval['pooled_oof']
    return out


def compute_wilcoxon(vals_a, vals_b):
    """
    Compute two-sided paired Wilcoxon Signed-Rank Test between paired arrays vals_a and vals_b.
    Returns dict with statistic, p_value, n_pairs, mean_diff.
    """
    a = np.asarray(vals_a, dtype=np.float64)
    b = np.asarray(vals_b, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    n = int(len(a))
    if n < 5 or np.allclose(a, b):
        return {'statistic': None, 'p_value': 1.0, 'n_pairs': n, 'mean_diff': float(np.mean(a - b)) if n > 0 else 0.0}
    try:
        stat, p = wilcoxon(a, b, zero_method='wilcox', alternative='two-sided')
        return {
            'statistic': float(stat),
            'p_value': float(p),
            'n_pairs': n,
            'mean_diff': float(np.mean(a - b)),
        }
    except Exception:
        return {'statistic': None, 'p_value': 1.0, 'n_pairs': n, 'mean_diff': float(np.mean(a - b))}


def format_p_val(p):
    if p is None or not np.isfinite(p):
        return "-"
    if p < 0.001:
        return f"{p:.4f}**"
    if p < 0.05:
        return f"{p:.4f}*"
    return f"{p:.4f}"


def get_feature_names():
    stats10 = ['Kur', 'Ske', 'SM', 'SC', 'FD', 'CD', 'CO', 'PSE', 'KE', 'SE']
    mods = ['E', 'P']
    names = []
    for m in mods:
        for sb in range(8):
            for st in stats10:
                names.append(f"{st}_{m}{sb}")
    return names


def main():
    parser = argparse.ArgumentParser(description="Analyze ST-MGM Phase 6B Results & Statistical Tests")
    parser.add_argument('--results_dir', type=str, default='./results')
    args = parser.parse_args()
    res_dir = args.results_dir

    summary_data = {
        'm9_stratified': {},
        'm9_blocked': {},
        'ablation_stratified': {},
        'baselines_stratified': {},
        'wilcoxon_ablation': {},
        'wilcoxon_baselines': {},
        'wilcoxon_blocked_vs_stratified': {},
        'feature_selection_50folds': {},
    }

    # 1. Collect M9 Stratified & Blocked across 10 subjects
    for sub in ALL_SUBJECTS:
        m9_eval = load_json(os.path.join(res_dir, sub, 'evaluation_results.json'))
        m9_ts = load_json(os.path.join(res_dir, sub, 'training_summary.json'))
        summ = summarize_subject_folds(enrich_fold_metrics(m9_eval, m9_ts))
        if summ:
            summary_data['m9_stratified'][sub] = summ

        blk_eval = load_json(os.path.join(res_dir, f"{sub}_blocked", 'evaluation_results.json'))
        blk_ts = load_json(os.path.join(res_dir, f"{sub}_blocked", 'training_summary.json'))
        blk_summ = summarize_subject_folds(enrich_fold_metrics(blk_eval, blk_ts))
        if blk_summ:
            summary_data['m9_blocked'][sub] = blk_summ

    # 2. Collect Ablation Variants across all available subjects
    for var in ABLATION_VARIANTS:
        summary_data['ablation_stratified'][var] = {}
        for sub in ALL_SUBJECTS:
            folder = sub if var == 'M9' else f"{sub}_{var.lower()}"
            v_eval = load_json(os.path.join(res_dir, folder, 'evaluation_results.json'))
            v_ts = load_json(os.path.join(res_dir, folder, 'training_summary.json'))
            # Fallback to subject M9 training_summary for glucose_std if needed
            if v_ts is None:
                v_ts = load_json(os.path.join(res_dir, sub, 'training_summary.json'))
            v_summ = summarize_subject_folds(enrich_fold_metrics(v_eval, v_ts))
            if v_summ:
                summary_data['ablation_stratified'][var][sub] = v_summ

    # 3. Collect Baselines across all available subjects
    for bm in BASELINE_MODELS:
        summary_data['baselines_stratified'][bm] = {}
        for sub in ALL_SUBJECTS:
            folder = f"{sub}_baseline_{bm}"
            b_eval = load_json(os.path.join(res_dir, folder, 'evaluation_results.json'))
            m9_ts = load_json(os.path.join(res_dir, sub, 'training_summary.json'))
            b_summ = summarize_subject_folds(enrich_fold_metrics(b_eval, m9_ts))
            if b_summ:
                summary_data['baselines_stratified'][bm][sub] = b_summ

    # 4. Wilcoxon Signed-Rank Tests: M9 vs Ablation Variants
    m9_map = summary_data['ablation_stratified'].get('M9', {})
    for var in ABLATION_VARIANTS:
        if var == 'M9':
            continue
        var_map = summary_data['ablation_stratified'].get(var, {})
        common_subs = [s for s in ALL_SUBJECTS if s in m9_map and s in var_map]
        if not common_subs:
            continue
        summary_data['wilcoxon_ablation'][var] = {'n_subjects': len(common_subs), 'subject_level': {}, 'fold_level': {}}
        for metric in ['auc', 'auprc', 'f1', 'balanced_acc', 'rmse', 'mard', 'pearson_r', 'parkes_ab']:
            sub_a = [m9_map[s][metric]['mean'] for s in common_subs if metric in m9_map[s] and metric in var_map[s]]
            sub_b = [var_map[s][metric]['mean'] for s in common_subs if metric in m9_map[s] and metric in var_map[s]]
            summary_data['wilcoxon_ablation'][var]['subject_level'][metric] = compute_wilcoxon(sub_a, sub_b)

            fold_a, fold_b = [], []
            for s in common_subs:
                if metric in m9_map[s] and metric in var_map[s]:
                    va = m9_map[s][metric]['values']
                    vb = var_map[s][metric]['values']
                    if len(va) == len(vb):
                        fold_a.extend(va)
                        fold_b.extend(vb)
            summary_data['wilcoxon_ablation'][var]['fold_level'][metric] = compute_wilcoxon(fold_a, fold_b)

    # 5. Wilcoxon Signed-Rank Tests: M9 vs Baselines (and Training-Mean Predictor)
    for bm in BASELINE_MODELS:
        b_map = summary_data['baselines_stratified'].get(bm, {})
        common_subs = [s for s in ALL_SUBJECTS if s in m9_map and s in b_map]
        if not common_subs:
            continue
        summary_data['wilcoxon_baselines'][bm] = {'n_subjects': len(common_subs), 'subject_level': {}, 'fold_level': {}}
        for metric in ['auc', 'auprc', 'f1', 'balanced_acc', 'rmse', 'mard', 'pearson_r', 'parkes_ab']:
            sub_a = [m9_map[s][metric]['mean'] for s in common_subs if metric in m9_map[s] and metric in b_map[s]]
            sub_b = [b_map[s][metric]['mean'] for s in common_subs if metric in m9_map[s] and metric in b_map[s]]
            summary_data['wilcoxon_baselines'][bm]['subject_level'][metric] = compute_wilcoxon(sub_a, sub_b)

            fold_a, fold_b = [], []
            for s in common_subs:
                if metric in m9_map[s] and metric in b_map[s]:
                    va = m9_map[s][metric]['values']
                    vb = b_map[s][metric]['values']
                    if len(va) == len(vb):
                        fold_a.extend(va)
                        fold_b.extend(vb)
            summary_data['wilcoxon_baselines'][bm]['fold_level'][metric] = compute_wilcoxon(fold_a, fold_b)

    # M9 RMSE vs Training-Mean Predictor RMSE across subjects and 50 folds
    sub_rmse_m9, sub_rmse_mean = [], []
    fold_rmse_m9, fold_rmse_mean = [], []
    for s in ALL_SUBJECTS:
        if s in m9_map and 'rmse' in m9_map[s] and 'rmse_mean_pred' in m9_map[s]:
            sub_rmse_m9.append(m9_map[s]['rmse']['mean'])
            sub_rmse_mean.append(m9_map[s]['rmse_mean_pred']['mean'])
            fold_rmse_m9.extend(m9_map[s]['rmse']['values'])
            fold_rmse_mean.extend(m9_map[s]['rmse_mean_pred']['values'])
    summary_data['wilcoxon_baselines']['mean_predictor_rmse'] = {
        'subject_level': compute_wilcoxon(sub_rmse_m9, sub_rmse_mean),
        'fold_level': compute_wilcoxon(fold_rmse_m9, fold_rmse_mean),
    }

    # M9 Blocked vs Stratified across 10 subjects and 50 folds
    blk_map = summary_data.get('m9_blocked', {})
    common_blk = [s for s in ALL_SUBJECTS if s in m9_map and s in blk_map]
    if common_blk:
        summary_data['wilcoxon_blocked_vs_stratified'] = {'n_subjects': len(common_blk), 'subject_level': {}, 'fold_level': {}}
        for metric in ['auc', 'auprc', 'f1', 'balanced_acc', 'rmse', 'mard', 'pearson_r', 'parkes_ab']:
            sub_a = [m9_map[s][metric]['mean'] for s in common_blk if metric in m9_map[s] and metric in blk_map[s]]
            sub_b = [blk_map[s][metric]['mean'] for s in common_blk if metric in m9_map[s] and metric in blk_map[s]]
            summary_data['wilcoxon_blocked_vs_stratified']['subject_level'][metric] = compute_wilcoxon(sub_a, sub_b)

            fold_a, fold_b = [], []
            for s in common_blk:
                if metric in m9_map[s] and metric in blk_map[s]:
                    va = m9_map[s][metric]['values']
                    vb = blk_map[s][metric]['values']
                    if len(va) == len(vb):
                        fold_a.extend(va)
                        fold_b.extend(vb)
            summary_data['wilcoxon_blocked_vs_stratified']['fold_level'][metric] = compute_wilcoxon(fold_a, fold_b)

    # 6. Feature Selection Distribution across 50 Folds
    feat_names = get_feature_names()
    stat_counts = {'Kur': 0, 'Ske': 0, 'SM': 0, 'SC': 0, 'FD': 0, 'CD': 0, 'CO': 0, 'PSE': 0, 'KE': 0, 'SE': 0}
    mod_counts = {'ECG': 0, 'PPG': 0}
    idx_counts = {}
    total_folds_counted = 0

    for sub in ALL_SUBJECTS:
        ts = load_json(os.path.join(res_dir, sub, 'training_summary.json'))
        if not ts:
            continue
        for fk in FOLDS:
            if fk in ts and 'selected_indices' in ts[fk]:
                total_folds_counted += 1
                for idx in ts[fk]['selected_indices']:
                    idx_int = int(idx)
                    idx_counts[idx_int] = idx_counts.get(idx_int, 0) + 1
                    if 0 <= idx_int < len(feat_names):
                        fname = feat_names[idx_int]
                        st = fname.split('_')[0]
                        if st in stat_counts:
                            stat_counts[st] += 1
                        if idx_int < 80:
                            mod_counts['ECG'] += 1
                        else:
                            mod_counts['PPG'] += 1

    top_features = sorted(idx_counts.items(), key=lambda x: (-x[1], x[0]))[:20]
    summary_data['feature_selection_50folds'] = {
        'total_folds': total_folds_counted,
        'total_selections': sum(stat_counts.values()),
        'statistic_counts': stat_counts,
        'modality_counts': mod_counts,
        'top_20_features': [
            {'index': idx, 'name': feat_names[idx] if idx < len(feat_names) else f"feat_{idx}", 'count': cnt}
            for idx, cnt in top_features
        ],
    }

    # 7. Build Markdown Report
    md = [
        "# ST-MGM Phase 6B Statistical & Multi-Subject Evaluation Summary",
        "",
        "## 1. Per-Subject ST-MGM (`M9`) Performance — Stratified 5-Fold CV & Training-Mean Baseline Comparison",
        "",
        "| Subject | AUC | AUPRC | F1 | BalAcc | Sens@90%Spec | RMSE (mg/dL) | Mean-Pred RMSE | $R^2$ | RMSE/SD | MARD (%) | Pearson $r$ | Parkes A+B (%) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for s in ALL_SUBJECTS:
        if s not in summary_data['m9_stratified']:
            continue
        d = summary_data['m9_stratified'][s]
        s_At90 = f"{d['sens_at_90spec']['mean']:.4f}" if 'sens_at_90spec' in d else "-"
        r_mp = f"{d['rmse_mean_pred']['mean']:.2f}" if 'rmse_mean_pred' in d else "-"
        r2_v = f"{d['r2']['mean']:.4f}" if 'r2' in d else "-"
        rsd_v = f"{d['rmse_sd_ratio']['mean']:.4f}" if 'rmse_sd_ratio' in d else "-"
        md.append(
            f"| `{s}` | {d['auc']['mean']:.4f}±{d['auc']['std']:.4f} | "
            f"{d['auprc']['mean']:.4f}±{d['auprc']['std']:.4f} | "
            f"{d['f1']['mean']:.4f}±{d['f1']['std']:.4f} | "
            f"{d['balanced_acc']['mean']:.4f}±{d['balanced_acc']['std']:.4f} | "
            f"{s_At90} | {d['rmse']['mean']:.2f}±{d['rmse']['std']:.2f} | "
            f"{r_mp} | {r2_v} | {rsd_v} | "
            f"{d['mard']['mean']:.2f}±{d['mard']['std']:.2f} | "
            f"{d['pearson_r']['mean']:.4f}±{d['pearson_r']['std']:.4f} | "
            f"{d['parkes_ab']['mean']:.2f}±{d['parkes_ab']['std']:.2f} |"
        )

    if summary_data['m9_stratified']:
        subs_present = [summary_data['m9_stratified'][s] for s in ALL_SUBJECTS if s in summary_data['m9_stratified']]
        def avg_m(k):
            vals = [x[k]['mean'] for x in subs_present if k in x]
            return f"{np.mean(vals):.4f}±{np.std(vals):.4f}" if vals else "-"
        def avg_m2(k):
            vals = [x[k]['mean'] for x in subs_present if k in x]
            return f"{np.mean(vals):.2f}±{np.std(vals):.2f}" if vals else "-"

        w_mp = summary_data['wilcoxon_baselines'].get('mean_predictor_rmse', {})
        p_sub_mp = format_p_val(w_mp.get('subject_level', {}).get('p_value'))
        p_fold_mp = format_p_val(w_mp.get('fold_level', {}).get('p_value'))

        md.append(
            f"| **10-Subj Mean±SD** | **{avg_m('auc')}** | **{avg_m('auprc')}** | "
            f"**{avg_m('f1')}** | **{avg_m('balanced_acc')}** | **{avg_m('sens_at_90spec')}** | "
            f"**{avg_m2('rmse')}** | **{avg_m2('rmse_mean_pred')}** | **{avg_m('r2')}** | "
            f"**{avg_m('rmse_sd_ratio')}** | **{avg_m2('mard')}** | **{avg_m('pearson_r')}** | "
            f"**{avg_m2('parkes_ab')}** |"
        )
        md.append("")
        md.append(
            f"- **Wilcoxon Signed-Rank Test (`M9` RMSE vs. Training-Mean Predictor RMSE)**: "
            f"Subject-level ($N={len(subs_present)}$) $p = {p_sub_mp}$, "
            f"Fold-level ($N={len(subs_present)*5}$) $p = {p_fold_mp}$."
        )

    # Blocked CV Section (if present)
    if summary_data['m9_blocked']:
        md.extend([
            "",
            "## 2. Blocked / Chronological 5-Fold CV (`M9_blocked`, MJ-2)",
            "",
            "| Subject | Blocked AUC | Blocked AUPRC | Blocked F1 | Blocked BalAcc | Blocked RMSE (mg/dL) | Blocked $R^2$ | Blocked MARD (%) | Blocked Pearson $r$ | Blocked Parkes A+B (%) |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])
        for s in ALL_SUBJECTS:
            if s not in summary_data['m9_blocked']:
                continue
            d = summary_data['m9_blocked'][s]
            md.append(
                f"| `{s}` | {d['auc']['mean']:.4f}±{d['auc']['std']:.4f} | "
                f"{d['auprc']['mean']:.4f}±{d['auprc']['std']:.4f} | "
                f"{d['f1']['mean']:.4f}±{d['f1']['std']:.4f} | "
                f"{d['balanced_acc']['mean']:.4f}±{d['balanced_acc']['std']:.4f} | "
                f"{d['rmse']['mean']:.2f}±{d['rmse']['std']:.2f} | "
                f"{d.get('r2', {}).get('mean', 0.0):.4f} | "
                f"{d['mard']['mean']:.2f}±{d['mard']['std']:.2f} | "
                f"{d['pearson_r']['mean']:.4f}±{d['pearson_r']['std']:.4f} | "
                f"{d['parkes_ab']['mean']:.2f}±{d['parkes_ab']['std']:.2f} |"
            )

    # Ablation Table Section
    md.extend([
        "",
        "## 3. Ablation Study Summary & Wilcoxon Signed-Rank Tests vs. `M9` (Table 6, MJ-4)",
        "",
        "| Variant | $N_{\\text{subj}}$ | AUC | AUPRC | F1 | BalAcc | RMSE (mg/dL) | $R^2$ | MARD (%) | Pearson $r$ | Parkes A+B (%) | $p_{\\text{AUC}}$ (Fold/Subj) | $p_{\\text{RMSE}}$ (Fold/Subj) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    for var in ABLATION_VARIANTS:
        v_map = summary_data['ablation_stratified'].get(var, {})
        if not v_map:
            continue
        subs_v = [v_map[s] for s in ALL_SUBJECTS if s in v_map]
        n_s = len(subs_v)
        def vm(k, dec=4):
            if n_s == 1 and k in subs_v[0]:
                return f"{subs_v[0][k]['mean']:.{dec}f}±{subs_v[0][k]['std']:.{dec}f}"
            vals = [x[k]['mean'] for x in subs_v if k in x]
            if not vals:
                return "-"
            return f"{np.mean(vals):.{dec}f}±{np.std(vals):.{dec}f}"

        w_info = summary_data['wilcoxon_ablation'].get(var, {})
        p_auc_f = format_p_val(w_info.get('fold_level', {}).get('auc', {}).get('p_value'))
        p_auc_s = format_p_val(w_info.get('subject_level', {}).get('auc', {}).get('p_value'))
        p_rmse_f = format_p_val(w_info.get('fold_level', {}).get('rmse', {}).get('p_value'))
        p_rmse_s = format_p_val(w_info.get('subject_level', {}).get('rmse', {}).get('p_value'))
        p_auc_str = "-" if var == 'M9' else f"{p_auc_f} / {p_auc_s}"
        p_rmse_str = "-" if var == 'M9' else f"{p_rmse_f} / {p_rmse_s}"

        if var == 'M9' and 'c1s01_raw' in v_map and n_s > 1:
            c1 = v_map['c1s01_raw']
            def c1m(k, dec=4):
                return f"{c1[k]['mean']:.{dec}f}±{c1[k]['std']:.{dec}f}" if k in c1 else "-"
            md.append(
                f"| **M9 (`c1s01`)** | 1 | {c1m('auc', 4)} | {c1m('auprc', 4)} | {c1m('f1', 4)} | "
                f"{c1m('balanced_acc', 4)} | {c1m('rmse', 2)} | {c1m('r2', 4)} | "
                f"{c1m('mard', 2)} | {c1m('pearson_r', 4)} | {c1m('parkes_ab', 2)} | - | - |"
            )

        md.append(
            f"| **{var}** | {n_s} | {vm('auc', 4)} | {vm('auprc', 4)} | {vm('f1', 4)} | "
            f"{vm('balanced_acc', 4)} | {vm('rmse', 2)} | {vm('r2', 4)} | "
            f"{vm('mard', 2)} | {vm('pearson_r', 4)} | {vm('parkes_ab', 2)} | "
            f"{p_auc_str} | {p_rmse_str} |"
        )

    # Baselines Table Section
    md.extend([
        "",
        "## 4. Baseline Comparison Summary & Wilcoxon Signed-Rank Tests vs. `M9` (Table 7, MJ-4)",
        "",
        "| Model | $N_{\\text{subj}}$ | AUC | AUPRC | F1 | BalAcc | RMSE (mg/dL) | $R^2$ | MARD (%) | Pearson $r$ | Parkes A+B (%) | $p_{\\text{AUC}}$ (Fold/Subj) | $p_{\\text{RMSE}}$ (Fold/Subj) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    for bm in BASELINE_MODELS:
        b_map = summary_data['baselines_stratified'].get(bm, {})
        if not b_map:
            continue
        subs_b = [b_map[s] for s in ALL_SUBJECTS if s in b_map]
        n_s = len(subs_b)
        def bm_fmt(k, dec=4):
            if n_s == 1 and k in subs_b[0]:
                return f"{subs_b[0][k]['mean']:.{dec}f}±{subs_b[0][k]['std']:.{dec}f}"
            vals = [x[k]['mean'] for x in subs_b if k in x]
            if not vals:
                return "-"
            return f"{np.mean(vals):.{dec}f}±{np.std(vals):.{dec}f}"

        w_info = summary_data['wilcoxon_baselines'].get(bm, {})
        p_auc_f = format_p_val(w_info.get('fold_level', {}).get('auc', {}).get('p_value'))
        p_auc_s = format_p_val(w_info.get('subject_level', {}).get('auc', {}).get('p_value'))
        p_rmse_f = format_p_val(w_info.get('fold_level', {}).get('rmse', {}).get('p_value'))
        p_rmse_s = format_p_val(w_info.get('subject_level', {}).get('rmse', {}).get('p_value'))

        md.append(
            f"| **{bm.upper()}** | {n_s} | {bm_fmt('auc', 4)} | {bm_fmt('auprc', 4)} | {bm_fmt('f1', 4)} | "
            f"{bm_fmt('balanced_acc', 4)} | {bm_fmt('rmse', 2)} | {bm_fmt('r2', 4)} | "
            f"{bm_fmt('mard', 2)} | {bm_fmt('pearson_r', 4)} | {bm_fmt('parkes_ab', 2)} | "
            f"{p_auc_f} / {p_auc_s} | {p_rmse_f} / {p_rmse_s} |"
        )

    # Feature Selection Section
    fs = summary_data['feature_selection_50folds']
    md.extend([
        "",
        f"## 5. DWT Feature Selection Distribution ({fs['total_folds']} Folds, {fs['total_selections']} Selections)",
        "",
        "| Statistical Operator | Selected Count | Share (%) |",
        "| :--- | :---: | :---: |",
    ])
    tot_sel = max(fs['total_selections'], 1)
    for st_name, cnt in fs['statistic_counts'].items():
        md.append(f"| `{st_name}` | {cnt} / {tot_sel} | {cnt * 100.0 / tot_sel:.2f}% |")

    out_md_path = os.path.join(res_dir, 'phase6b_analysis_summary.md')
    out_json_path = os.path.join(res_dir, 'phase6b_analysis_summary.json')
    with open(out_md_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(md) + "\n")
    with open(out_json_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2)

    print("\n".join(md))
    print(f"\nSaved Markdown report -> {out_md_path}")
    print(f"Saved JSON summary    -> {out_json_path}")


if __name__ == '__main__':
    main()

