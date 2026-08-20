"""Top-level master script to run Experiments A, B, and C and compile the final report."""

import os
import sys
import argparse
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from experiments.run_experiment_a import run_experiment_a
from experiments.run_experiment_b import run_experiment_b
from experiments.run_experiment_c import run_experiment_c
from generate_report import generate_markdown_report


def main():
    parser = argparse.ArgumentParser(description="Master Execution Pipeline for HIGHT Neural Cryptanalysis")
    parser.add_argument("--samples", type=int, default=10000, help="Number of dataset samples per experiment (default: 10000)")
    parser.add_argument("--results-dir", type=str, default="results", help="Target results directory (default: results)")
    parser.add_argument("--quick", action="store_true", help="Quick execution mode with reduced parameters")
    args = parser.parse_args()

    start_time = time.time()
    print("=" * 80)
    print("STARTING FULL REPRODUCIBILITY PIPELINE: HIGHT NEURAL CRYPTANALYSIS")
    print(f"Configurations: samples={args.samples}, results_dir='{args.results_dir}', quick_mode={args.quick}")
    print("=" * 80)

    # 1. Run Experiment A
    print("\n>>> [1/3] EXECUTING EXPERIMENT A...")
    run_experiment_a(
        n_samples=args.samples,
        data_path=os.path.join("data", "exp_a_data.npz"),
        results_dir=args.results_dir,
        quick_mode=args.quick
    )

    # 2. Run Experiment B
    print("\n>>> [2/3] EXECUTING EXPERIMENT B...")
    run_experiment_b(
        n_samples=args.samples,
        pairs_per_sample=4,
        data_path=os.path.join("data", "exp_b_data.npz"),
        results_dir=args.results_dir,
        quick_mode=args.quick
    )

    # 3. Run Experiment C
    print("\n>>> [3/3] EXECUTING EXPERIMENT C...")
    run_experiment_c(
        n_samples=args.samples,
        data_path=os.path.join("data", "exp_c_data.npz"),
        results_dir=args.results_dir,
        quick_mode=args.quick
    )

    # 4. Generate Synthesis Report
    print("\n>>> [4/4] GENERATING SYNTHESIS REPORT...")
    report_text = generate_markdown_report(
        results_dir=args.results_dir,
        output_path=os.path.join(args.results_dir, "report.md")
    )

    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"FULL PIPELINE EXECUTION COMPLETED IN {total_time:.1f}s!")
    print(f"CSVs, plots, and markdown report are located in: {args.results_dir}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
