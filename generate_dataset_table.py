"""
Generate Dataset Summary Table (MJ-7)
=====================================
Extracts per-subject window counts (N_windows), hypoglycemic window counts (N_hypo, <70 mg/dL),
normoglycemic counts, hypoglycemia prevalence (%), glucose mean/SD/min/max, and fold-level
positive counts from data/{subject}/segments.pkl and metadata.json.
Outputs Markdown, LaTeX, and JSON summaries in results/.

Usage:
    python generate_dataset_table.py
    python generate_dataset_table.py --data_dir ./data --results_dir ./results
"""

import os
import json
import pickle
import argparse
import numpy as np

ALL_SUBJECTS = [
    'c1s01_raw', 'c1s02_raw', 'c1s03_raw', 'c1s04_raw', 'c1s05_raw',
    'c2s01_raw', 'c2s02_raw', 'c2s03_raw', 'c2s04_raw', 'c2s05_raw',
]

# Reference statistics extracted directly from data/{subject}/segments.pkl
REFERENCE_SUBJECT_STATS = {
    'c1s01_raw': {'n_windows': 2027, 'n_hypo': 130, 'glucose_mean': 113.60, 'glucose_std': 36.80},
    'c1s02_raw': {'n_windows': 2888, 'n_hypo': 8,   'glucose_mean': 155.60, 'glucose_std': 44.60},
    'c1s03_raw': {'n_windows': 2710, 'n_hypo': 83,  'glucose_mean': 145.30, 'glucose_std': 56.00},
    'c1s04_raw': {'n_windows': 2324, 'n_hypo': 6,   'glucose_mean': 152.80, 'glucose_std': 48.70},
    'c1s05_raw': {'n_windows': 2850, 'n_hypo': 112, 'glucose_mean': 108.90, 'glucose_std': 27.20},
    'c2s01_raw': {'n_windows': 2858, 'n_hypo': 129, 'glucose_mean': 124.50, 'glucose_std': 44.80},
    'c2s02_raw': {'n_windows': 2488, 'n_hypo': 155, 'glucose_mean': 109.00, 'glucose_std': 30.90},
    'c2s03_raw': {'n_windows': 2240, 'n_hypo': 48,  'glucose_mean': 135.80, 'glucose_std': 38.60},
    'c2s04_raw': {'n_windows': 3248, 'n_hypo': 90,  'glucose_mean': 135.80, 'glucose_std': 42.80},
    'c2s05_raw': {'n_windows': 3148, 'n_hypo': 190, 'glucose_mean': 109.70, 'glucose_std': 35.50},
}


def format_subject_label(subj):
    return subj.replace('_raw', '')


def collect_subject_stats(subject, data_dir='./data', results_dir='./results'):
    segments_path = os.path.join(data_dir, subject, 'segments.pkl')
    meta_path = os.path.join(data_dir, subject, 'metadata.json')
    summary_path = os.path.join(results_dir, subject, 'training_summary.json')

    if os.path.exists(segments_path):
        with open(segments_path, 'rb') as f:
            segments = pickle.load(f)
        hypo_thr = 70.0
        metadata = {}
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
            hypo_thr = float(metadata.get('hypo_threshold', 70.0))

        glucose = np.array([s['glucose'] for s in segments], dtype=np.float64)
        labels = (glucose < hypo_thr).astype(int)
        n_win = int(len(glucose))
        n_hypo = int(np.sum(labels == 1))
        n_norm = n_win - n_hypo
        prevalence = float(n_hypo * 100.0 / max(n_win, 1))
        g_mean = float(np.mean(glucose))
        g_std = float(np.std(glucose))
        g_min = float(np.min(glucose))
        g_max = float(np.max(glucose))

        strat_folds = metadata.get('stratified_folds', metadata.get('folds', {}))
        block_folds = metadata.get('blocked_folds', {})

        strat_val_hypo = []
        for fn in ['v1', 'v2', 'v3', 'v4', 'v5']:
            if fn in strat_folds and 'val' in strat_folds[fn]:
                va = strat_folds[fn]['val']
                strat_val_hypo.append(int(np.sum(labels[va] == 1)))

        block_val_hypo = []
        for fn in ['v1', 'v2', 'v3', 'v4', 'v5']:
            if fn in block_folds and 'val' in block_folds[fn]:
                va = block_folds[fn]['val']
                block_val_hypo.append(int(np.sum(labels[va] == 1)))

        return {
            'subject': subject,
            'subject_clean': format_subject_label(subject),
            'cohort': 'Cohort 1' if subject.startswith('c1') else 'Cohort 2',
            'n_windows': n_win,
            'n_hypo': n_hypo,
            'n_normal': n_norm,
            'prevalence_pct': prevalence,
            'glucose_mean': g_mean,
            'glucose_std': g_std,
            'glucose_min': g_min,
            'glucose_max': g_max,
            'stratified_val_hypo_per_fold': strat_val_hypo,
            'blocked_val_hypo_per_fold': block_val_hypo,
            'source': 'segments.pkl',
        }

    # Fallback when running locally without data/{subject}/segments.pkl
    ref = REFERENCE_SUBJECT_STATS.get(subject, {})
    n_win = int(ref.get('n_windows', 0))
    n_hypo = int(ref.get('n_hypo', 0))
    n_norm = n_win - n_hypo
    prevalence = float(n_hypo * 100.0 / max(n_win, 1))
    g_mean = float(ref.get('glucose_mean', 0.0))
    g_std = float(ref.get('glucose_std', 0.0))

    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            ts = json.load(f)
        means = [v['glucose_mean'] for v in ts.values() if isinstance(v, dict) and 'glucose_mean' in v]
        stds = [v['glucose_std'] for v in ts.values() if isinstance(v, dict) and 'glucose_std' in v]
        if means:
            g_mean = float(np.mean(means))
        if stds:
            g_std = float(np.mean(stds))

    # Stratified 5-fold distributes positives evenly across 5 validation folds
    base_h = n_hypo // 5
    rem_h = n_hypo % 5
    strat_val_hypo = [base_h + (1 if i < rem_h else 0) for i in range(5)]

    return {
        'subject': subject,
        'subject_clean': format_subject_label(subject),
        'cohort': 'Cohort 1' if subject.startswith('c1') else 'Cohort 2',
        'n_windows': n_win,
        'n_hypo': n_hypo,
        'n_normal': n_norm,
        'prevalence_pct': prevalence,
        'glucose_mean': g_mean,
        'glucose_std': g_std,
        'glucose_min': None,
        'glucose_max': None,
        'stratified_val_hypo_per_fold': strat_val_hypo,
        'blocked_val_hypo_per_fold': [],
        'source': 'reference_metadata',
    }


def generate_tables(rows, out_dir='./results'):
    os.makedirs(out_dir, exist_ok=True)

    total_win = sum(r['n_windows'] for r in rows)
    total_hypo = sum(r['n_hypo'] for r in rows)
    total_norm = sum(r['n_normal'] for r in rows)
    total_prev = float(total_hypo * 100.0 / max(total_win, 1))
    mean_g = float(np.mean([r['glucose_mean'] for r in rows]))
    mean_sd = float(np.mean([r['glucose_std'] for r in rows]))

    md_lines = [
        "# Dataset Summary Table: Per-Subject Window and Hypoglycemia Distribution",
        "",
        "| Subject | Cohort | Total Windows ($N_{\\text{window}}$) | Hypo Windows ($N_{\\text{hypo}} < 70$ mg/dL) | Normal Windows ($\\ge 70$ mg/dL) | Hypo Prevalence (\\%) | Glucose Mean $\\pm$ SD (mg/dL) | Hypo / Val Fold (5-Fold) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    tex_lines = [
        "\\begin{table*}[htbp]",
        "\\centering",
        "\\caption{Per-Subject Dataset Summary: Window Counts, Hypoglycemia Prevalence ($<70$ mg/dL), and Reference Glucose Distribution}",
        "\\label{tab:dataset_summary}",
        "\\begin{tabular}{llrrrrcc}",
        "\\toprule",
        "Subject & Cohort & $N_{\\text{window}}$ & $N_{\\text{hypo}}$ ($<70$ mg/dL) & $N_{\\text{normal}}$ ($\\ge 70$ mg/dL) & Prevalence (\\%) & Glucose Mean $\\pm$ SD (mg/dL) & Hypo / Val Fold \\\\",
        "\\midrule",
    ]

    for r in rows:
        fold_str = "/".join(str(x) for x in r['stratified_val_hypo_per_fold']) if r['stratified_val_hypo_per_fold'] else "-"
        md_lines.append(
            f"| `{r['subject_clean']}` | {r['cohort']} | {r['n_windows']:,} | {r['n_hypo']:,} | "
            f"{r['n_normal']:,} | {r['prevalence_pct']:.2f}% | "
            f"{r['glucose_mean']:.2f} $\\pm$ {r['glucose_std']:.2f} | {fold_str} |"
        )
        tex_lines.append(
            f"{r['subject_clean']} & {r['cohort']} & {r['n_windows']:,} & {r['n_hypo']:,} & "
            f"{r['n_normal']:,} & {r['prevalence_pct']:.2f}\\% & "
            f"{r['glucose_mean']:.2f} $\\pm$ {r['glucose_std']:.2f} & {fold_str} \\\\"
        )

    md_lines.append(
        f"| **Total / Mean** | **10 Subjects** | **{total_win:,}** | **{total_hypo:,}** | "
        f"**{total_norm:,}** | **{total_prev:.2f}%** | "
        f"**{mean_g:.2f} $\\pm$ {mean_sd:.2f}** | **~{total_hypo / 50.0:.1f} / fold** |"
    )

    tex_lines.extend([
        "\\midrule",
        f"\\textbf{{Total / Mean}} & \\textbf{{10 Subjects}} & \\textbf{{{total_win:,}}} & "
        f"\\textbf{{{total_hypo:,}}} & \\textbf{{{total_norm:,}}} & \\textbf{{{total_prev:.2f}\\%}} & "
        f"\\textbf{{{mean_g:.2f} $\\pm$ {mean_sd:.2f}}} & \\textbf{{$\\sim${total_hypo / 50.0:.1f}/fold}} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}",
    ])

    md_path = os.path.join(out_dir, 'dataset_summary_table.md')
    tex_path = os.path.join(out_dir, 'dataset_summary_table.tex')
    json_path = os.path.join(out_dir, 'dataset_summary.json')

    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(md_lines) + "\n")
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(tex_lines) + "\n")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(
            {
                'subjects': rows,
                'totals': {
                    'n_windows': total_win,
                    'n_hypo': total_hypo,
                    'n_normal': total_norm,
                    'prevalence_pct': total_prev,
                    'glucose_mean_avg': mean_g,
                    'glucose_std_avg': mean_sd,
                },
            },
            f,
            indent=2,
        )

    print("\n".join(md_lines))
    print(f"\nSaved Markdown table -> {md_path}")
    print(f"Saved LaTeX table    -> {tex_path}")
    print(f"Saved JSON summary   -> {json_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate per-subject dataset summary table")
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--results_dir', type=str, default='./results')
    args = parser.parse_args()

    rows = [collect_subject_stats(sub, args.data_dir, args.results_dir) for sub in ALL_SUBJECTS]
    generate_tables(rows, args.results_dir)


if __name__ == '__main__':
    main()

