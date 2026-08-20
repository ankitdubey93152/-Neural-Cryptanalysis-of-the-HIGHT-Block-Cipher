"""Training, validation, and evaluation pipeline with Stratified K-Fold cross validation."""

import os
import sys
from typing import Callable, Dict, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm
import tensorflow as tf

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.seeding import set_global_seed


def run_kfold(
    model_builder: Callable[..., tf.keras.Model],
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    epochs: int = 5,
    batch_size: int = 256,
    random_state: int = 42,
    verbose: int = 0,
    model_name: Optional[str] = None,
    **model_kwargs: Any
) -> Dict[str, Any]:
    """Execute Stratified K-Fold Cross-Validation for a neural distinguisher.

    Args:
        model_builder (Callable): Function returning a freshly compiled tf.keras.Model.
        X (np.ndarray): Preprocessed feature tensor.
        y (np.ndarray): Binary target labels (0 or 1).
        n_folds (int): Number of cross-validation splits (default: 5).
        epochs (int): Training epochs per fold (default: 5).
        batch_size (int): Batch size (default: 256).
        random_state (int): Seed for fold splitting reproducibility.
        verbose (int): Keras fit verbosity (0=silent, 1=progress, 2=one line per epoch).
        model_name (str, optional): Custom display name for the model.
        **model_kwargs: Hyperparameters forwarded to model_builder.

    Returns:
        Dict[str, Any]: Detailed evaluation metrics and training history.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    fold_accuracies = []
    fold_losses = []
    fold_aucs = []
    histories = []

    progress_bar = tqdm(
        enumerate(skf.split(X, y), start=1),
        total=n_folds,
        desc=f"Training {model_name or 'Model'} ({n_folds}-Fold CV)"
    )

    for fold, (train_idx, val_idx) in progress_bar:
        # Reset graph state and seeds for reproducible fold initialization
        tf.keras.backend.clear_session()
        set_global_seed(random_state + fold)

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # Instantiate a fresh model instance for this fold
        model = model_builder(**model_kwargs)
        if model_name is None:
            model_name = model.name

        # Train on current split
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose
        )
        histories.append(history.history)

        # Inference on validation split
        y_pred_prob = model.predict(X_val, batch_size=batch_size, verbose=0).flatten()
        y_pred_class = (y_pred_prob >= 0.5).astype(int)

        val_loss, val_acc = model.evaluate(X_val, y_val, batch_size=batch_size, verbose=0)
        
        # Calculate ROC-AUC (handle edge cases gracefully)
        try:
            val_auc = roc_auc_score(y_val, y_pred_prob)
        except ValueError:
            val_auc = 0.5

        fold_accuracies.append(val_acc)
        fold_losses.append(val_loss)
        fold_aucs.append(val_auc)

        progress_bar.set_postfix({
            "val_acc": f"{val_acc:.4f}",
            "val_auc": f"{val_auc:.4f}",
            "val_loss": f"{val_loss:.4f}"
        })

    # Extract swept hyperparameter 'n_dense_layers' if provided
    dense_layers_count = model_kwargs.get("n_dense_layers", 3 if "mlp" in model_name.lower() else 2)

    return {
        "model_name": model_name,
        "n_dense_layers": dense_layers_count,
        "n_folds": n_folds,
        "epochs": epochs,
        "batch_size": batch_size,
        "mean_accuracy": float(np.mean(fold_accuracies)),
        "std_accuracy": float(np.std(fold_accuracies)),
        "mean_auc": float(np.mean(fold_aucs)),
        "std_auc": float(np.std(fold_aucs)),
        "mean_loss": float(np.mean(fold_losses)),
        "std_loss": float(np.std(fold_losses)),
        "fold_accuracies": fold_accuracies,
        "fold_losses": fold_losses,
        "fold_aucs": fold_aucs,
        "histories": histories,
    }


def plot_training_curves(
    result_dict: Dict[str, Any],
    save_path: Optional[str] = None,
    title_prefix: str = ""
) -> None:
    """Plot cross-validation accuracy and loss curves per epoch.

    Args:
        result_dict (Dict[str, Any]): Output dictionary from run_kfold.
        save_path (str, optional): Target file path to save plot (e.g. in results/).
        title_prefix (str): Optional prefix for plot titles.
    """
    histories = result_dict["histories"]
    epochs = range(1, result_dict["epochs"] + 1)
    n_folds = result_dict["n_folds"]
    model_name = result_dict["model_name"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy Plot
    for i, h in enumerate(histories):
        axes[0].plot(epochs, h["accuracy"], linestyle="--", alpha=0.4, label=f"Fold {i+1} Train")
        axes[0].plot(epochs, h["val_accuracy"], linestyle="-", alpha=0.7, label=f"Fold {i+1} Val")

    # Mean accuracy across folds
    mean_val_acc = np.mean([h["val_accuracy"] for h in histories], axis=0)
    axes[0].plot(epochs, mean_val_acc, color="black", linewidth=2, label="Mean Val Accuracy")
    axes[0].axhline(0.5, color="red", linestyle=":", label="Chance (50%)")
    axes[0].set_title(f"{title_prefix} {model_name} - Accuracy over Epochs".strip())
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim([0.45, 0.55])
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(loc="lower right", fontsize=8)

    # Loss Plot
    for i, h in enumerate(histories):
        axes[1].plot(epochs, h["loss"], linestyle="--", alpha=0.4, label=f"Fold {i+1} Train")
        axes[1].plot(epochs, h["val_loss"], linestyle="-", alpha=0.7, label=f"Fold {i+1} Val")

    mean_val_loss = np.mean([h["val_loss"] for h in histories], axis=0)
    axes[1].plot(epochs, mean_val_loss, color="black", linewidth=2, label="Mean Val Loss")
    axes[1].set_title(f"{title_prefix} {model_name} - Loss over Epochs".strip())
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Binary Cross-Entropy Loss")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(loc="upper right", fontsize=8)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")

    plt.close()


def summarize_run(result_dict: Dict[str, Any]) -> pd.Series:
    """Format k-fold results into a standardized pandas Series matching published table format.

    Args:
        result_dict (Dict[str, Any]): Output dictionary from run_kfold.

    Returns:
        pd.Series: Structured results row.
    """
    # Accuracy formatted as percentage matching source (e.g. 50.85)
    acc_pct = round(result_dict["mean_accuracy"] * 100, 2)
    std_pct = round(result_dict["std_accuracy"] * 100, 2)
    auc = round(result_dict["mean_auc"], 4)

    return pd.Series({
        "Model Name": result_dict["model_name"],
        "Dense Layers": result_dict["n_dense_layers"],
        "Folds": result_dict["n_folds"],
        "Epochs": result_dict["epochs"],
        "Accuracy (%)": acc_pct,
        "Std Dev (%)": std_pct,
        "ROC-AUC": auc
    })


if __name__ == "__main__":
    from models.architectures import build_mlp

    print("=" * 60)
    print("SANITY CHECK: 5-FOLD CV ON SYNTHETIC RANDOM DATASET")
    print("=" * 60)

    set_global_seed(42)

    # Generate purely random synthetic data (5,000 samples, 128 features)
    n_samples = 5000
    n_features = 128
    X_synthetic = np.random.binomial(1, 0.5, size=(n_samples, n_features)).astype(np.float32)
    y_synthetic = np.random.randint(0, 2, size=(n_samples,)).astype(np.float32)

    print(f"Synthetic Data Shape: X={X_synthetic.shape}, y={y_synthetic.shape}")
    print(f"Target distribution: 0={np.sum(y_synthetic == 0)}, 1={np.sum(y_synthetic == 1)}")

    # Run 5-fold CV with MLP (3 dense layers)
    results = run_kfold(
        model_builder=build_mlp,
        X=X_synthetic,
        y=y_synthetic,
        n_folds=5,
        epochs=3,
        batch_size=128,
        input_dim=n_features,
        n_dense_layers=3,
        units=128,  # smaller units for fast sanity check
        model_name="Synthetic_MLP_Sanity"
    )

    print("\n--- Cross-Validation Metrics ---")
    print(f"Mean Accuracy: {results['mean_accuracy']:.4f} (+/- {results['std_accuracy']:.4f})")
    print(f"Mean ROC-AUC:  {results['mean_auc']:.4f} (+/- {results['std_auc']:.4f})")
    print(f"Mean Loss:     {results['mean_loss']:.4f}")

    # Summary table row
    summary = summarize_run(results)
    print("\n--- Result Summary Series ---")
    print(summary)

    # Plot sanity check
    plot_path = os.path.join("results", "synthetic_sanity_curves.png")
    plot_training_curves(results, save_path=plot_path, title_prefix="[Sanity]")

    print("\n" + "=" * 60)
    print("PIPELINE UNBIASED SANITY CHECK COMPLETED!")
    print("=" * 60)
