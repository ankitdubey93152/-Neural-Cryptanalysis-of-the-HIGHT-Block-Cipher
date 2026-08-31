"""Experiment C: Cipher Identification Distinguisher (HIGHT vs SM4).

Evaluates 1D CNN, 2D CNN (Dense=2, 3), and MLP models discriminating between HIGHT and SM4 under fixed difference Δ.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.seeding import set_global_seed
from data.generators import generate_experiment_c, bits, save_dataset, load_dataset
from models.architectures import build_mlp, build_cnn_1d, build_cnn_2d, prepare_input
from training.train_eval import run_kfold, summarize_run, plot_training_curves


def run_experiment_c(
    n_samples: int = 10000,
    data_path: str = "data/exp_c_data.npz",
    results_dir: str = "results",
    quick_mode: bool = False
) -> pd.DataFrame:
    """Execute Experiment C pipeline across all model configurations."""
    set_global_seed(42)
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 70)
    print("EXPERIMENT C: CIPHER IDENTIFICATION (HIGHT VS SM4)")
    print("=" * 70)

    # 1. Dataset Generation / Loading
    if os.path.exists(data_path):
        print(f"Loading cached Experiment C dataset from {data_path}...")
        X_raw, y = load_dataset(data_path)
        if len(y) > n_samples:
            X_raw, y = X_raw[:n_samples], y[:n_samples]
    else:
        print(f"Generating {n_samples} samples for Experiment C...")
        X_raw, y = generate_experiment_c(n_samples=n_samples, show_progress=True)
        save_dataset(data_path, X_raw, y)
        print(f"Dataset saved to {data_path}")

    X_bits = bits(X_raw)
    print(f"Data ready: X_bits shape={X_bits.shape}, labels={y.shape} (Balance: {np.mean(y):.1%} HIGHT)")

    folds = 2 if quick_mode else 5
    epochs = 2 if quick_mode else 5

    # 2. Define Model Configurations
    configs = [
        {
            "name": "1D CNN",
            "model_type": "cnn_1d",
            "builder": build_cnn_1d,
            "kwargs": {"input_shape": (64, 2), "n_dense_layers": 2},
            "folds": folds,
            "epochs": epochs,
            "batch_size": 256,
        },
        {
            "name": "2D CNN (Dense=2)",
            "model_type": "cnn_2d",
            "builder": build_cnn_2d,
            "kwargs": {"input_shape": (2, 64, 1), "n_dense_layers": 2, "kernel_size": 3},
            "folds": folds,
            "epochs": epochs,
            "batch_size": 256,
        },
        {
            "name": "2D CNN (Dense=3)",
            "model_type": "cnn_2d",
            "builder": build_cnn_2d,
            "kwargs": {"input_shape": (2, 64, 1), "n_dense_layers": 3, "kernel_size": 3},
            "folds": folds,
            "epochs": epochs,
            "batch_size": 256,
        },
        {
            "name": "MLP (Dense=2)",
            "model_type": "mlp",
            "builder": build_mlp,
            "kwargs": {"input_dim": 128, "n_dense_layers": 2, "units": 640},
            "folds": folds,
            "epochs": epochs,
            "batch_size": 256,
        },
        {
            "name": "MLP (Dense=3)",
            "model_type": "mlp",
            "builder": build_mlp,
            "kwargs": {"input_dim": 128, "n_dense_layers": 3, "units": 640},
            "folds": folds,
            "epochs": epochs,
            "batch_size": 256,
        },
    ]

    results_list = []

    for cfg in configs:
        dense_layers = cfg["kwargs"].get("n_dense_layers", 2)
        print(f"\n---> Running {cfg['name']} (Dense={dense_layers}, Folds={cfg['folds']}, Epochs={cfg['epochs']})...")
        X_in = prepare_input(X_bits, cfg["model_type"], "c")

        res = run_kfold(
            model_builder=cfg["builder"],
            X=X_in,
            y=y,
            n_folds=cfg["folds"],
            epochs=cfg["epochs"],
            batch_size=cfg["batch_size"],
            model_name=cfg["name"],
            **cfg["kwargs"]
        )

        row = summarize_run(res)
        results_list.append(row)

        plot_file = os.path.join(results_dir, f"exp_c_{cfg['name'].lower().replace(' ', '_').replace('=', '')}_curves.png")
        plot_training_curves(res, save_path=plot_file, title_prefix="[Exp C]")

    df = pd.DataFrame(results_list)
    csv_path = os.path.join(results_dir, "table_experiment_c.csv")
    df.to_csv(csv_path, index=False)
    print("\n" + "=" * 70)
    print("EXPERIMENT C RESULTS TABLE:")
    print("=" * 70)
    print(df.to_string(index=False))
    print(f"\nSaved results table to: {csv_path}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Experiment C (Cipher Identification Distinguisher)")
    parser.add_argument("--samples", type=int, default=10000, help="Number of samples to generate")
    parser.add_argument("--data-path", type=str, default="data/exp_c_data.npz", help="Path to cache dataset")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory to save CSVs and plots")
    parser.add_argument("--quick", action="store_true", help="Run with reduced parameters for fast test")
    args = parser.parse_args()

    run_experiment_c(
        n_samples=args.samples,
        data_path=args.data_path,
        results_dir=args.results_dir,
        quick_mode=args.quick
    )
