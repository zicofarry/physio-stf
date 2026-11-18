"""
Generate publication-quality figures for the ST-MGM manuscript from Phase 6B results.

Outputs:
    results/figures/fig2_dwt_feature_selection.png
    results/figures/fig3_multisubject_performance.png
    results/figures/fig4_ablation_and_baselines.png
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9.5,
    'axes.titlesize': 10.5,
    'axes.titleweight': 'bold',
    'axes.labelsize': 9.5,
    'axes.labelweight': 'bold',
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 8.5,
    'legend.fontsize': 8.2,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

SUBJECTS = [
    'c1s01_raw', 'c1s02_raw', 'c1s03_raw', 'c1s04_raw', 'c1s05_raw',
    'c2s01_raw', 'c2s02_raw', 'c2s03_raw', 'c2s04_raw', 'c2s05_raw',
]
SUB_LABELS = [s.replace('_raw', '') for s in SUBJECTS]
FOLDS = ['v1', 'v2', 'v3', 'v4', 'v5']
STATS10 = ['Kur', 'Ske', 'SM', 'SC', 'FD', 'CD', 'CO', 'PSE', 'KE', 'SE']
STAT_FULL = {
    'CO': 'Lag-1 Autocorrelation (CO)',
    'SC': 'Spectral Centroid (SC)',
    'SE': 'Norm. Spectral Entropy (SE)',
    'PSE': 'Power Spectral Entropy (PSE)',
    'FD': 'Dominant Frequency (FD)',
    'KE': 'Energy Shannon Entropy (KE)',
    'CD': 'Complexity Domain Var. (CD)',
    'Ske': 'Skewness (Ske)',
    'SM': 'Spectral Mean (SM)',
    'Kur': 'Kurtosis (Kur)',
}


def generate_fig2(results_dir, out_dir):
    """Fig. 2: 50-fold DWT feature selection distribution by operator, modality, and sub-band."""
    true_names = [f"{st}_{m}{sb}" for m in ['E', 'P'] for sb in range(8) for st in STATS10]
    ecg_stat_cnt = {s: 0 for s in STATS10}
    ppg_stat_cnt = {s: 0 for s in STATS10}
    feat_cnt = {}

    for sub in SUBJECTS:
        ts_path = os.path.join(results_dir, sub, 'training_summary.json')
        with open(ts_path, 'r', encoding='utf-8') as f:
            ts = json.load(f)
        for fk in FOLDS:
            for idx in ts[fk]['selected_indices']:
                idx = int(idx)
                st = STATS10[idx % 10]
                if idx < 80:
                    ecg_stat_cnt[st] += 1
                else:
                    ppg_stat_cnt[st] += 1
                fname = true_names[idx]
                feat_cnt[fname] = feat_cnt.get(fname, 0) + 1

    sorted_ops = sorted(STATS10, key=lambda s: ecg_stat_cnt[s] + ppg_stat_cnt[s])
    op_labels = [STAT_FULL[s] for s in sorted_ops]
    ecg_vals = np.array([ecg_stat_cnt[s] for s in sorted_ops])
    ppg_vals = np.array([ppg_stat_cnt[s] for s in sorted_ops])
    totals = ecg_vals + ppg_vals

    top15 = sorted(feat_cnt.items(), key=lambda x: (-x[1], x[0]))[:15][::-1]
    top_names = [x[0] for x in top15]
    top_counts = [x[1] for x in top15]
    top_colors = ['#1f4e78' if '_E' in n else '#d95f02' for n in top_names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.4), gridspec_kw={'width_ratios': [1.18, 1.0]})

    y_pos = np.arange(len(sorted_ops))
    ax1.barh(y_pos, ecg_vals, color='#1f4e78', edgecolor='white', height=0.68, label='ECG Sub-bands (55.4%)')
    ax1.barh(y_pos, ppg_vals, left=ecg_vals, color='#d95f02', edgecolor='white', height=0.68, label='PPG Sub-bands (44.6%)')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(op_labels)
    ax1.set_xlabel('Selection Count Across 50 Folds (900 Total)')
    ax1.set_title('(a) Selected DWT Operators by Modality', loc='left', pad=8)
    ax1.set_xlim(0, 182)
    ax1.grid(axis='x', linestyle='--', alpha=0.35)
    ax1.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9)

    for i, (tot, e_v, p_v) in enumerate(zip(totals, ecg_vals, ppg_vals)):
        pct = tot / 9.0
        ax1.text(tot + 2.5, i, f"{tot} ({pct:.1f}%)", va='center', ha='left', fontsize=8.0, fontweight='bold')

    y_top = np.arange(len(top_names))
    ax2.barh(y_top, top_counts, color=top_colors, edgecolor='white', height=0.68)
    ax2.set_yticks(y_top)
    ax2.set_yticklabels(top_names)
    ax2.set_xlabel('Selection Frequency (Out of 50 Folds)')
    ax2.set_title('(b) Top 15 Recurring Sub-Band Descriptors', loc='left', pad=8)
    ax2.set_xlim(0, 34)
    ax2.axvline(25, color='gray', linestyle=':', linewidth=0.9, alpha=0.6)
    ax2.grid(axis='x', linestyle='--', alpha=0.35)

    for i, cnt in enumerate(top_counts):
        ax2.text(cnt + 0.5, i, f"{cnt}/50 ({cnt*2}%)", va='center', ha='left', fontsize=8.0)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1f4e78', label='ECG Sub-band (E0–E7)'),
        Patch(facecolor='#d95f02', label='PPG Sub-band (P0–P7)'),
    ]
    ax2.legend(handles=legend_elements, loc='lower right', frameon=True, facecolor='white', framealpha=0.9)

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'fig2_dwt_feature_selection.png')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    return out_path


def generate_fig3(summary_data, out_dir):
    """Fig. 3: 10-Subject RMSE vs Mean-Predictor, Parkes Error Grid Zones, & Stratified vs Blocked CV."""
    m9_s = summary_data['m9_stratified']
    m9_b = summary_data['m9_blocked']

    rmse_m9 = [m9_s[s]['rmse']['mean'] for s in SUBJECTS]
    rmse_m9_sd = [m9_s[s]['rmse']['std'] for s in SUBJECTS]
    rmse_mp = [m9_s[s]['rmse_mean_pred']['mean'] for s in SUBJECTS]
    r2_vals = [m9_s[s]['r2']['mean'] for s in SUBJECTS]

    auc_strat = [m9_s[s]['auc']['mean'] for s in SUBJECTS]
    auc_strat_sd = [m9_s[s]['auc']['std'] for s in SUBJECTS]
    auc_blk = [m9_b[s]['auc']['mean'] for s in SUBJECTS]
    auc_blk_sd = [m9_b[s]['auc']['std'] for s in SUBJECTS]

    zone_a = [np.mean([m9_s[s]['folds'][fk]['parkes_zone_a'] for fk in FOLDS]) for s in SUBJECTS]
    zone_b = [np.mean([m9_s[s]['folds'][fk]['parkes_zone_b'] for fk in FOLDS]) for s in SUBJECTS]
    mp_ab = [np.mean([m9_s[s]['folds'][fk]['parkes_ab_mean_pred'] for fk in FOLDS]) for s in SUBJECTS]

    labels_ext = SUB_LABELS + ['Cohort\nMean']
    rmse_m9.append(float(np.mean(rmse_m9)))
    rmse_m9_sd.append(float(np.std(rmse_m9[:-1])))
    rmse_mp.append(float(np.mean(rmse_mp)))
    r2_vals.append(float(np.mean(r2_vals)))

    auc_strat.append(float(np.mean(auc_strat)))
    auc_strat_sd.append(float(np.std(auc_strat[:-1])))
    auc_blk.append(float(np.mean(auc_blk)))
    auc_blk_sd.append(float(np.std(auc_blk[:-1])))

    zone_a.append(float(np.mean(zone_a)))
    zone_b.append(float(np.mean(zone_b)))
    mp_ab.append(float(np.mean(mp_ab)))

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    x = np.arange(len(labels_ext))
    w = 0.36

    # Panel (a): RMSE comparison
    ax = axes[0]
    ax.bar(x - w/2, rmse_mp, width=w, color='#b0bec5', edgecolor='#455a64', linewidth=0.6, label='Training-Mean Predictor')
    ax.bar(x + w/2, rmse_m9, width=w, yerr=rmse_m9_sd, capsize=2.2, color='#1f4e78', edgecolor='#0d253f', linewidth=0.6, label='ST-MGM (M9, p = 0.0020*)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels_ext, rotation=40, ha='right')
    ax.set_ylabel('RMSE (mg/dL)')
    ax.set_title('(a) Continuous BG RMSE vs. Mean Baseline', loc='left', pad=8)
    ax.set_ylim(0, 72)
    ax.grid(axis='y', linestyle='--', alpha=0.35)
    ax.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9)
    for i, r2 in enumerate(r2_vals):
        top_y = max(rmse_mp[i], rmse_m9[i] + rmse_m9_sd[i]) + 1.2
        ax.text(i, top_y, f".{int(round(r2*100)):02d}", ha='center', va='bottom', fontsize=6.8, color='#1f4e78', fontweight='bold')
    ax.text(0.98, 0.93, "Numbers above bars: $R^2$", transform=ax.transAxes, ha='right', va='top', fontsize=7.5, style='italic', color='#1f4e78')

    # Panel (b): Parkes Zones A + B
    ax = axes[1]
    ax.bar(x, zone_a, width=0.55, color='#2b83ba', edgecolor='white', label='Parkes Zone A (58.5%)')
    ax.bar(x, zone_b, bottom=zone_a, width=0.55, color='#abdda4', edgecolor='white', label='Parkes Zone B (36.9%)')
    ax.plot(x, mp_ab, color='#d7191c', marker='o', markersize=4, linewidth=1.3, linestyle='--', label='Mean-Pred Zones A+B (93.1%)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels_ext, rotation=40, ha='right')
    ax.set_ylabel('Predictions in Clinical Zones (%)')
    ax.set_title('(b) Parkes Error Grid Zones A+B (95.4%)', loc='left', pad=8)
    ax.set_ylim(40, 104)
    ax.grid(axis='y', linestyle='--', alpha=0.35)
    ax.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9)

    # Panel (c): Stratified CV vs Chronological Blocked CV AUC
    ax = axes[2]
    ax.bar(x - w/2, auc_strat, width=w, yerr=auc_strat_sd, capsize=2.2, color='#1f4e78', edgecolor='#0d253f', linewidth=0.6, label='Stratified 5-Fold CV (0.774)')
    ax.bar(x + w/2, auc_blk, width=w, yerr=auc_blk_sd, capsize=2.2, color='#e66101', edgecolor='#7f3b08', linewidth=0.6, label='Blocked 5-Fold CV (0.565)')
    ax.axhline(0.5, color='black', linestyle=':', linewidth=1.0, label='Chance Level (0.50)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels_ext, rotation=40, ha='right')
    ax.set_ylabel('Hypoglycemia AUC-ROC')
    ax.set_title('(c) Stratified vs. Chronological Blocked CV', loc='left', pad=8)
    ax.set_ylim(0.3, 1.08)
    ax.grid(axis='y', linestyle='--', alpha=0.35)
    ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'fig3_multisubject_performance.png')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    return out_path


def generate_fig4(summary_data, out_dir):
    """Fig. 4: 10-Subject Ablation Study (M1..M9) and Baseline Comparison."""
    abl = summary_data['ablation_stratified']
    base = summary_data['baselines_stratified']

    variants = ['M1', 'M2', 'M3', 'M6', 'M7', 'M8', 'M9_CLS', 'M9_REG', 'M9_NOCOUPLE', 'M9']
    var_labels = [
        'M1\n(DWT)', 'M2\n(ResNet)', 'M3\n(EDA)', 'M6\n(No EDA)', 'M7\n(No DWT)',
        'M8\n(No ResNet)', 'M9_CLS\n(Cls Only)', 'M9_REG\n(Reg Only)', 'M9_NOCPL\n(No Cpl)', 'M9\n(Full Ours)'
    ]

    auc_v = [np.mean([abl[v][s]['auc']['mean'] for s in SUBJECTS]) * 100 for v in variants]
    bal_v = [np.mean([abl[v][s]['balanced_acc']['mean'] for s in SUBJECTS]) * 100 for v in variants]
    ab_v = [np.mean([abl[v][s]['parkes_ab']['mean'] for s in SUBJECTS]) for v in variants]
    mard_v = [np.mean([abl[v][s]['mard']['mean'] for s in SUBJECTS]) for v in variants]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.4), gridspec_kw={'width_ratios': [1.3, 1.0]})

    x1 = np.arange(len(variants))
    w1 = 0.27
    ax1.bar(x1 - w1, auc_v, width=w1, color='#1f4e78', label='AUC-ROC (%)')
    ax1.bar(x1, bal_v, width=w1, color='#2b83ba', label='Balanced Acc (%)')
    ax1.bar(x1 + w1, ab_v, width=w1, color='#66c2a5', label='Parkes Zones A+B (%)')
    ax1_r = ax1.twinx()
    ax1_r.plot(x1, mard_v, color='#d7191c', marker='s', markersize=5, linewidth=1.6, label='MARD (%, right axis)')
    ax1_r.spines['top'].set_visible(False)
    ax1_r.set_ylabel('MARD (%) — Lower is Better', color='#d7191c', fontweight='bold')
    ax1_r.tick_params(axis='y', labelcolor='#d7191c')
    ax1_r.set_ylim(18, 31)

    ax1.set_xticks(x1)
    ax1.set_xticklabels(var_labels, fontsize=7.8)
    ax1.set_ylabel('Classification / Clinical Grid Metric (%)')
    ax1.set_ylim(40, 102)
    ax1.set_title('(a) 10-Subject Ablation Study Across 10 Model Variants (50 Folds)', loc='left', pad=8)
    ax1.grid(axis='y', linestyle='--', alpha=0.35)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_r.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower left', ncol=2, frameon=True, facecolor='white', framealpha=0.9, fontsize=7.8)

    # Panel (b): Baselines comparison highlighting Sensitivity / Balanced Acc / F1 vs MARD
    b_models = ['RBF-SVM', 'Random\nForest', 'Quamer\net al. [23]', 'ST-MGM\n(M6)', 'ST-MGM\n(M9 Ours)']
    sens_b = [
        np.mean([base['svm'][s]['sensitivity']['mean'] for s in SUBJECTS]) * 100,
        np.mean([base['rf'][s]['sensitivity']['mean'] for s in SUBJECTS]) * 100,
        np.mean([base['quamer'][s]['sensitivity']['mean'] for s in SUBJECTS]) * 100,
        np.mean([abl['M6'][s]['sensitivity']['mean'] for s in SUBJECTS]) * 100,
        np.mean([abl['M9'][s]['sensitivity']['mean'] for s in SUBJECTS]) * 100,
    ]
    bal_b = [
        np.mean([base['svm'][s]['balanced_acc']['mean'] for s in SUBJECTS]) * 100,
        np.mean([base['rf'][s]['balanced_acc']['mean'] for s in SUBJECTS]) * 100,
        np.mean([base['quamer'][s]['balanced_acc']['mean'] for s in SUBJECTS]) * 100,
        np.mean([abl['M6'][s]['balanced_acc']['mean'] for s in SUBJECTS]) * 100,
        np.mean([abl['M9'][s]['balanced_acc']['mean'] for s in SUBJECTS]) * 100,
    ]
    f1_b = [
        np.mean([base['svm'][s]['f1']['mean'] for s in SUBJECTS]) * 100,
        np.mean([base['rf'][s]['f1']['mean'] for s in SUBJECTS]) * 100,
        np.mean([base['quamer'][s]['f1']['mean'] for s in SUBJECTS]) * 100,
        np.mean([abl['M6'][s]['f1']['mean'] for s in SUBJECTS]) * 100,
        np.mean([abl['M9'][s]['f1']['mean'] for s in SUBJECTS]) * 100,
    ]
    ab_b = [
        np.mean([base['svm'][s]['parkes_ab']['mean'] for s in SUBJECTS]),
        np.mean([base['rf'][s]['parkes_ab']['mean'] for s in SUBJECTS]),
        np.mean([base['quamer'][s]['parkes_ab']['mean'] for s in SUBJECTS]),
        np.mean([abl['M6'][s]['parkes_ab']['mean'] for s in SUBJECTS]),
        np.mean([abl['M9'][s]['parkes_ab']['mean'] for s in SUBJECTS]),
    ]

    x2 = np.arange(len(b_models))
    w2 = 0.25
    ax2.bar(x2 - w2, sens_b, width=w2, color='#e66101', label='Hypoglycemia Sensitivity (%)')
    ax2.bar(x2, f1_b, width=w2, color='#5e3c99', label='F1-Score (×100)')
    ax2.bar(x2 + w2, bal_b, width=w2, color='#1f4e78', label='Balanced Accuracy (%)')

    for i in range(len(b_models)):
        ax2.text(x2[i] - w2, sens_b[i] + 1.0, f"{sens_b[i]:.1f}%", ha='center', va='bottom', fontsize=7.2, fontweight='bold', color='#b34800')
        ax2.text(x2[i], f1_b[i] + 1.0, f"{f1_b[i]:.1f}", ha='center', va='bottom', fontsize=7.2, color='#5e3c99')
        ax2.text(x2[i] + w2, bal_b[i] + 1.0, f"{bal_b[i]:.1f}%", ha='center', va='bottom', fontsize=7.2, color='#1f4e78')

    ax2.set_xticks(x2)
    ax2.set_xticklabels(b_models, fontsize=8.3)
    ax2.set_ylabel('Minority-Class Detection Metric (%)')
    ax2.set_ylim(0, 74)
    ax2.set_title('(b) Minority-Class Alarm Comparison vs. Baselines (10 Subjects)', loc='left', pad=8)
    ax2.grid(axis='y', linestyle='--', alpha=0.35)
    ax2.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9, fontsize=8.0)

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'fig4_ablation_and_baselines.png')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    return out_path


def main():
    results_dir = './results'
    out_dir = os.path.join(results_dir, 'figures')
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(results_dir, 'phase6b_analysis_summary.json'), 'r', encoding='utf-8') as f:
        summary_data = json.load(f)

    f2 = generate_fig2(results_dir, out_dir)
    f3 = generate_fig3(summary_data, out_dir)
    f4 = generate_fig4(summary_data, out_dir)
    print(f"Generated Fig. 2 -> {f2}")
    print(f"Generated Fig. 3 -> {f3}")
    print(f"Generated Fig. 4 -> {f4}")


if __name__ == '__main__':
    main()
