"""Unit tests for training and evaluation pipeline."""

import os
import tempfile
import numpy as np
import pytest
from training.train_eval import run_kfold, summarize_run, plot_training_curves
from models.architectures import build_mlp


def test_run_kfold_synthetic():
    """Verify run_kfold executes k-fold cross validation and returns valid metrics."""
    n_samples = 200
    n_features = 32
    X = np.random.binomial(1, 0.5, size=(n_samples, n_features)).astype(np.float32)
    y = np.random.randint(0, 2, size=(n_samples,)).astype(np.float32)

    results = run_kfold(
        model_builder=build_mlp,
        X=X,
        y=y,
        n_folds=2,
        epochs=2,
        batch_size=64,
        input_dim=n_features,
        n_dense_layers=2,
        units=32,
        model_name="TestMLP"
    )

    assert "mean_accuracy" in results
    assert "mean_auc" in results
    assert "histories" in results
    assert len(results["fold_accuracies"]) == 2
    assert 0.0 <= results["mean_accuracy"] <= 1.0
    assert 0.0 <= results["mean_auc"] <= 1.0


def test_summarize_run():
    """Verify summarize_run formats results into standard pandas Series."""
    fake_results = {
        "model_name": "MLP",
        "n_dense_layers": 3,
        "n_folds": 5,
        "epochs": 10,
        "mean_accuracy": 0.5085,
        "std_accuracy": 0.0042,
        "mean_auc": 0.5012,
        "std_auc": 0.003,
        "mean_loss": 0.6931,
        "std_loss": 0.001
    }

    summary = summarize_run(fake_results)
    assert summary["Model Name"] == "MLP"
    assert summary["Dense Layers"] == 3
    assert summary["Folds"] == 5
    assert summary["Epochs"] == 10
    assert summary["Accuracy (%)"] == 50.85
    assert summary["Std Dev (%)"] == 0.42
    assert summary["ROC-AUC"] == 0.5012


def test_plot_training_curves():
    """Verify training curves plot generation and saving."""
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "test_curve.png")
        fake_results = {
            "model_name": "TestModel",
            "n_folds": 2,
            "epochs": 3,
            "histories": [
                {"accuracy": [0.5, 0.51, 0.52], "val_accuracy": [0.49, 0.50, 0.50], "loss": [0.7, 0.69, 0.69], "val_loss": [0.7, 0.7, 0.7]},
                {"accuracy": [0.5, 0.50, 0.51], "val_accuracy": [0.50, 0.50, 0.51], "loss": [0.7, 0.69, 0.69], "val_loss": [0.7, 0.7, 0.7]}
            ]
        }

        plot_training_curves(fake_results, save_path=save_path)
        assert os.path.exists(save_path)
