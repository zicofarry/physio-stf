"""
ST-MGM: Master Pipeline Runner
==================================
Runs the entire ST-MGM pipeline from raw data to evaluation.

Phases:
    1. Raw Preprocessing: ECG/PPG/EDA filtering & segmentation
    2. Feature Extraction: DWT, beat segmentation, feature selection
    3. Training: Multi-task training (BCE + MSE) with k-fold CV
    4. Evaluation: Classification (AUC, F1) & Regression (RMSE, MARD) metrics

Usage:
    python run_all.py --subject c1s01_raw_single
    python run_all.py --subject c1s01_raw_single --skip_preprocess
    python run_all.py --subject c1s01_raw_single --limit_segments 10
"""

import os
import sys
import time
import argparse
import subprocess


def run_step(description, command, cwd=None):
    """Run a pipeline step and check for errors."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"  Command: {command}")
    print(f"{'='*60}\n")

    start = time.time()
    result = subprocess.run(command, shell=True, cwd=cwd)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\n  ERROR: Step failed with return code {result.returncode}")
        print(f"  Elapsed: {elapsed:.1f}s")
        return False

    print(f"\n  Step completed in {elapsed:.1f}s")
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ST-MGM: Full Pipeline')
    parser.add_argument('--subject', type=str, required=True, help='Subject ID')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to raw data (default: auto-detect)')
    parser.add_argument('--skip_preprocess', action='store_true',
                        help='Skip Phase 1 and 2 (use existing processed data)')
    parser.add_argument('--skip_train', action='store_true',
                        help='Skip Phase 3 (use existing trained models)')
    parser.add_argument('--limit_segments', type=int, default=0,
                        help='Limit to first N segments (default: 0 for all)')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Auto-detect data path
    if args.data_path is None:
        candidates = [
            os.path.join(script_dir, 'dataset', args.subject),
            os.path.join(script_dir, 'dataset', f'{args.subject}_raw'),
            os.path.join(script_dir, 'dataset', f'{args.subject}_raw_single'),
        ]
        for c in candidates:
            if os.path.exists(c):
                args.data_path = os.path.abspath(c)
                break

        if args.data_path is None and not args.skip_preprocess:
            print(f"ERROR: Could not find raw data for {args.subject}.")
            print(f"Searched: {candidates}")
            print("Use --data_path to specify manually, or --skip_preprocess if already processed.")
            sys.exit(1)

    total_start = time.time()

    print(f"\n{'#'*60}")
    print(f"  ST-MGM: Full Pipeline for {args.subject}")
    print(f"  Raw data: {args.data_path}")
    print(f"{'#'*60}")

    success = True

    # Phase 1: Raw Preprocessing
    if not args.skip_preprocess:
        limit_arg = f" --limit_segments {args.limit_segments}" if args.limit_segments > 0 else ""
        success = run_step(
            "Phase 1: Raw Data Preprocessing (ECG/PPG/EDA Filter + 5-min Windows)",
            f'python preprocess_raw.py --subject {args.subject} --data_path "{args.data_path}"{limit_arg}',
            cwd=script_dir
        )
        if not success:
            sys.exit(1)

        # Phase 2: Feature Extraction
        success = run_step(
            "Phase 2: Feature Extraction (DWT + Beat Segmentation + Feature Selection)",
            f'python preprocess_features.py --subject {args.subject}',
            cwd=script_dir
        )
        if not success:
            sys.exit(1)
    else:
        print("\n  Skipping Phase 1+2 (preprocessing)")

    # Phase 3: Training
    if not args.skip_train:
        success = run_step(
            "Phase 3: Multi-Task Training (BCE + MSE, k-fold CV)",
            f'python train.py -s {args.subject}',
            cwd=script_dir
        )
        if not success:
            print("  WARNING: Training failed!")
    else:
        print("\n  Skipping Phase 3 (training)")

    # Phase 4: Evaluation
    success = run_step(
        "Phase 4: Evaluation (AUC, F1, RMSE, MARD)",
        f'python test.py -s {args.subject}',
        cwd=script_dir
    )

    total_elapsed = time.time() - total_start
    hours = int(total_elapsed // 3600)
    mins = int((total_elapsed % 3600) // 60)
    secs = int(total_elapsed % 60)

    print(f"\n{'#'*60}")
    print(f"  Pipeline Complete!")
    print(f"  Subject: {args.subject}")
    print(f"  Total time: {hours}h {mins}m {secs}s")
    print(f"{'#'*60}")
