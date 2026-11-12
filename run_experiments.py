"""
ST-MGM: Master Experiment Runner (Batch Execution)
==================================================
Automates multi-subject training, modular ablation studies, and baseline benchmarks
with GPU memory cleanup and temperature guard intervals for HPC servers.

Usage:
    # Run full ST-MGM (M9) for all 10 subjects
    python run_experiments.py --mode stmgm --subjects all

    # Run key ablation variants (M1, M2, M3, M6, M7, M8) for c1s01_raw
    python run_experiments.py --mode ablation --subjects c1s01_raw --variants M1 M2 M3 M6 M7 M8

    # Run all baselines (SVM, RF, Quamer) for all subjects
    python run_experiments.py --mode baselines --subjects all

    # Run complete benchmark pipeline
    python run_experiments.py --mode full --subjects all
"""

import os
import sys
import time
import argparse
import subprocess

ALL_SUBJECTS = [
    'c1s01_raw', 'c1s02_raw', 'c1s03_raw', 'c1s04_raw', 'c1s05_raw',
    'c2s01_raw', 'c2s02_raw', 'c2s03_raw', 'c2s04_raw', 'c2s05_raw'
]

CORE_ABLATION_VARIANTS = ['M1', 'M2', 'M3', 'M6', 'M7', 'M8']


def run_command(cmd, desc=""):
    print(f"\n>>> [{time.strftime('%Y-%m-%d %H:%M:%S')}] {desc}")
    print(f"    CMD: {' '.join(cmd)}")
    t0 = time.time()
    res = subprocess.run(cmd)
    elapsed = time.time() - t0
    if res.returncode != 0:
        print(f"!!! ERROR: Command failed with exit code {res.returncode} ({elapsed:.1f}s)")
        return False
    print(f"--- Finished successfully in {elapsed:.1f}s ---")
    return True


def cooldown(seconds=60):
    if seconds > 0:
        print(f"    [Cooldown] Pausing {seconds}s for GPU cooling and cache flush...")
        time.sleep(seconds)


def main():
    parser = argparse.ArgumentParser(description="ST-MGM: Master Experiment Runner")
    parser.add_argument('--mode', type=str, required=True,
                        choices=['stmgm', 'ablation', 'baselines', 'full', 'preprocess_all'],
                        help="Experiment mode")
    parser.add_argument('--subjects', nargs='+', default=['all'],
                        help="List of subjects or 'all'")
    parser.add_argument('--variants', nargs='+', default=CORE_ABLATION_VARIANTS,
                        help="Ablation variants to run")
    parser.add_argument('--cooldown_sec', type=int, default=30,
                        help="Cooldown seconds between heavy runs")
    parser.add_argument('--reextract', action='store_true',
                        help="Run preprocess_features.py for target subjects before training")
    args = parser.parse_args()

    subjects = ALL_SUBJECTS if 'all' in args.subjects else args.subjects

    print("========================================================")
    print("  ST-MGM Experiment Runner")
    print(f"  Mode:      {args.mode}")
    print(f"  Subjects:  {subjects}")
    print(f"  Reextract: {args.reextract}")
    print("========================================================")

    # 1. Preprocess raw data if requested
    if args.reextract or args.mode in ['preprocess_all', 'full']:
        for sub in subjects:
            run_command([sys.executable, 'preprocess_features.py', '--subject', sub],
                        desc=f"Feature extraction for {sub}")

    # 2. Train and evaluate Full ST-MGM (M9)
    if args.mode in ['stmgm', 'full']:
        for sub in subjects:
            ok1 = run_command([sys.executable, 'train.py', '--subject', sub, '--variant', 'M9'],
                              desc=f"Training ST-MGM (M9) for {sub}")
            if ok1:
                run_command([sys.executable, 'test.py', '--subject', sub, '--variant', 'M9'],
                            desc=f"Evaluating ST-MGM (M9) for {sub}")
            cooldown(args.cooldown_sec)

    # 3. Ablation Study Variants (Table 6)
    if args.mode in ['ablation', 'full']:
        for sub in subjects:
            for var in args.variants:
                ok = run_command([sys.executable, 'train.py', '--subject', sub, '--variant', var],
                                 desc=f"Training Ablation {var} for {sub}")
                if ok:
                    run_command([sys.executable, 'test.py', '--subject', sub, '--variant', var],
                                desc=f"Evaluating Ablation {var} for {sub}")
                cooldown(args.cooldown_sec // 2)

    # 4. Baselines Comparison (Table 7)
    if args.mode in ['baselines', 'full']:
        for sub in subjects:
            run_command([sys.executable, 'run_baselines.py', '--subject', sub, '--models', 'svm', 'rf', 'quamer'],
                        desc=f"Running Baselines (SVM, RF, Quamer) for {sub}")
            cooldown(args.cooldown_sec)

    print("\n========================================================")
    print("  All requested experiments completed successfully!")
    print("========================================================")


if __name__ == '__main__':
    main()

