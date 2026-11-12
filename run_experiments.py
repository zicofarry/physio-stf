"""
ST-MGM: Master Experiment Runner (Batch Execution)
==================================================
Automates multi-subject training, modular ablation studies, blocked CV evaluation,
and baseline benchmarks with GPU memory cleanup and cooldown intervals for HPC servers.

Usage:
    # Backfill timestamps & blocked CV splits for all 10 subjects (fast cache mode)
    python run_experiments.py --mode preprocess_all --subjects all

    # Re-evaluate existing M9 checkpoints (adds R2, RMSE/SD, Sens@90Spec, Pooled OOF)
    python run_experiments.py --mode eval_only --subjects all

    # Run Blocked / Chronological 5-fold CV for all 10 subjects (MJ-2)
    python run_experiments.py --mode blocked_stmgm --subjects all

    # Run full ablation suite across all 10 subjects (MJ-4)
    python run_experiments.py --mode ablation --subjects all --skip_existing

    # Run all baselines (SVM, RF, Quamer) across all 10 subjects (MJ-4)
    python run_experiments.py --mode baselines --subjects all

    # Generate dataset summary table & statistical analysis (Wilcoxon tests)
    python run_experiments.py --mode analyze
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

ALL_ABLATION_VARIANTS = [
    'M1', 'M2', 'M3', 'M6', 'M7', 'M8', 'M9', 'M9_CLS', 'M9_REG', 'M9_NOCOUPLE'
]


def has_all_checkpoints(results_dir, subject, variant='M9', cv_mode='stratified'):
    """Check whether v1.pt..v5.pt already exist for the target subject/variant/cv_mode."""
    var_up = variant.upper()
    folder = subject if var_up == 'M9' else f"{subject}_{var_up.lower()}"
    if cv_mode.lower() == 'blocked':
        folder = f"{folder}_blocked"
    target_dir = os.path.join(results_dir, folder)
    return all(os.path.exists(os.path.join(target_dir, f"v{i}.pt")) for i in range(1, 6))


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
    os.makedirs('logs', exist_ok=True)

    parser = argparse.ArgumentParser(description="ST-MGM: Master Experiment Runner")
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=[
            'stmgm',
            'eval_only',
            'blocked_stmgm',
            'ablation',
            'baselines',
            'analyze',
            'full',
            'preprocess_all',
        ],
        help="Experiment mode",
    )
    parser.add_argument('--subjects', nargs='+', default=['all'],
                        help="List of subjects or 'all'")
    parser.add_argument('--variants', nargs='+', default=ALL_ABLATION_VARIANTS,
                        help="Ablation variants to run")
    parser.add_argument('--models', nargs='+', default=['svm', 'rf', 'quamer'],
                        choices=['svm', 'rf', 'quamer'],
                        help="Baseline models to run")
    parser.add_argument('--cv_mode', type=str, default='stratified',
                        choices=['stratified', 'blocked'],
                        help="Cross-validation split mode (stratified or blocked)")
    parser.add_argument('--purge_gap', type=int, default=6,
                        help="Temporal purge gap (windows) around validation block in blocked CV")
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'],
                        help="Compute device for training/evaluation (default: auto)")
    parser.add_argument('--cooldown_sec', type=int, default=20,
                        help="Cooldown seconds between heavy runs")
    parser.add_argument('--reextract', action='store_true',
                        help="Run preprocess_features.py for target subjects before training")
    parser.add_argument('--force_extract', action='store_true',
                        help="Force full DWT/NeuroKit2 feature re-extraction in preprocess_features.py")
    parser.add_argument('--skip_existing', action='store_true',
                        help="Skip train.py if all 5 fold .pt checkpoints already exist and run test.py only")
    args = parser.parse_args()

    subjects = ALL_SUBJECTS if 'all' in args.subjects else args.subjects
    cv_mode = 'blocked' if args.mode == 'blocked_stmgm' else args.cv_mode

    print("========================================================")
    print("  ST-MGM Experiment Runner")
    print(f"  Mode:          {args.mode}")
    print(f"  CV Mode:       {cv_mode} (purge_gap={args.purge_gap})")
    print(f"  Subjects:      {subjects}")
    print(f"  Skip Existing: {args.skip_existing}")
    print(f"  Reextract:     {args.reextract}")
    print("========================================================")

    # 1. Preprocess raw data / backfill timestamps & blocked splits if requested
    if args.reextract or args.mode in ['preprocess_all', 'full']:
        for sub in subjects:
            cmd = [
                sys.executable, 'preprocess_features.py',
                '--subject', sub,
                '--cv_mode', cv_mode,
                '--purge_gap', str(args.purge_gap),
            ]
            if args.force_extract:
                cmd.append('--force_extract')
            run_command(cmd, desc=f"Preprocessing / Split Metadata Update for {sub}")

    # 2. Evaluate-only mode: refresh evaluation_results.json on existing checkpoints
    if args.mode == 'eval_only':
        for sub in subjects:
            for var in args.variants:
                if has_all_checkpoints('./results', sub, variant=var, cv_mode=cv_mode):
                    run_command(
                        [
                            sys.executable, 'test.py',
                            '--subject', sub,
                            '--variant', var,
                            '--cv_mode', cv_mode,
                            '--purge_gap', str(args.purge_gap),
                        ],
                        desc=f"Evaluating existing {var} ({cv_mode}) for {sub}",
                    )

    # 3. Train and evaluate Full ST-MGM (M9) in stratified or blocked CV
    if args.mode in ['stmgm', 'blocked_stmgm', 'full']:
        for sub in subjects:
            if args.skip_existing and has_all_checkpoints('./results', sub, variant='M9', cv_mode=cv_mode):
                print(f"\n[Skip Train] Checkpoints already exist for {sub} (M9, {cv_mode}), running test.py only.")
                ok1 = True
            else:
                ok1 = run_command(
                    [
                        sys.executable, 'train.py',
                        '--subject', sub,
                        '--variant', 'M9',
                        '--cv_mode', cv_mode,
                        '--purge_gap', str(args.purge_gap),
                    ],
                    desc=f"Training ST-MGM (M9, {cv_mode}) for {sub}",
                )
            if ok1:
                run_command(
                    [
                        sys.executable, 'test.py',
                        '--subject', sub,
                        '--variant', 'M9',
                        '--cv_mode', cv_mode,
                        '--purge_gap', str(args.purge_gap),
                    ],
                    desc=f"Evaluating ST-MGM (M9, {cv_mode}) for {sub}",
                )
            cooldown(args.cooldown_sec)

    # 4. Ablation Study Variants (Table 6)
    if args.mode in ['ablation', 'full']:
        for sub in subjects:
            for var in args.variants:
                already_trained = has_all_checkpoints('./results', sub, variant=var, cv_mode=cv_mode)
                if (args.skip_existing or var.upper() == 'M9') and already_trained:
                    print(f"\n[Skip Train] Checkpoints already exist for {sub} ({var}, {cv_mode}), running test.py only.")
                    ok = True
                else:
                    ok = run_command(
                        [
                            sys.executable, 'train.py',
                            '--subject', sub,
                            '--variant', var,
                            '--cv_mode', cv_mode,
                            '--purge_gap', str(args.purge_gap),
                        ],
                        desc=f"Training Ablation {var} ({cv_mode}) for {sub}",
                    )
                if ok:
                    run_command(
                        [
                            sys.executable, 'test.py',
                            '--subject', sub,
                            '--variant', var,
                            '--cv_mode', cv_mode,
                            '--purge_gap', str(args.purge_gap),
                        ],
                        desc=f"Evaluating Ablation {var} ({cv_mode}) for {sub}",
                    )
                if not already_trained:
                    cooldown(max(args.cooldown_sec // 2, 5))

    # 5. Baselines Comparison (Table 7)
    if args.mode in ['baselines', 'full']:
        for sub in subjects:
            cmd = [
                sys.executable, 'run_baselines.py',
                '--subject', sub,
                '--models', *args.models,
                '--cv_mode', cv_mode,
                '--purge_gap', str(args.purge_gap),
                '--device', args.device,
            ]
            if args.skip_existing:
                cmd.append('--skip_existing')
            run_command(cmd, desc=f"Running Baselines ({', '.join(args.models)}, {cv_mode}, device={args.device}) for {sub}")
            cooldown(args.cooldown_sec)

    # 6. Dataset Table & Statistical Analysis
    if args.mode in ['analyze', 'full']:
        if os.path.exists('generate_dataset_table.py'):
            run_command([sys.executable, 'generate_dataset_table.py'],
                        desc="Generating Dataset Summary Table (MJ-7)")
        if os.path.exists('analyze_results.py'):
            run_command([sys.executable, 'analyze_results.py'],
                        desc="Running Statistical Analysis & Wilcoxon Signed-Rank Tests")

    print("\n========================================================")
    print("  All requested experiments completed successfully!")
    print("========================================================")


if __name__ == '__main__':
    main()

