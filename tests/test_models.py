"""Unit tests for neural distinguisher model architectures and input shapers."""

import pytest
import numpy as np
from models.architectures import (
    build_mlp,
    build_cnn_1d,
    build_cnn_2d,
    prepare_input
)


def test_build_mlp_compilation_and_forward_pass():
    """Verify MLP model compiles and outputs valid sigmoid probabilities."""
    model = build_mlp(input_dim=128, n_dense_layers=3, units=640)
    assert model.output_shape == (None, 1)

    dummy_input = np.random.binomial(1, 0.5, size=(8, 128)).astype(np.float32)
    preds = model.predict(dummy_input, verbose=0)
    assert preds.shape == (8, 1)
    assert np.all((preds >= 0.0) & (preds <= 1.0))


def test_build_mlp_layer_sweep():
    """Verify MLP can be constructed with varying dense layer counts (2 and 3)."""
    m2 = build_mlp(input_dim=128, n_dense_layers=2, units=640)
    m3 = build_mlp(input_dim=128, n_dense_layers=3, units=640)
    assert len(m2.layers) < len(m3.layers)


def test_build_cnn_1d_compilation_and_forward_pass():
    """Verify 1D CNN model compiles and performs forward pass."""
    model = build_cnn_1d(input_shape=(64, 2), n_dense_layers=2)
    assert model.output_shape == (None, 1)

    dummy_input = np.random.binomial(1, 0.5, size=(8, 64, 2)).astype(np.float32)
    preds = model.predict(dummy_input, verbose=0)
    assert preds.shape == (8, 1)
    assert np.all((preds >= 0.0) & (preds <= 1.0))


def test_build_cnn_2d_compilation_and_forward_pass():
    """Verify 2D CNN model compiles with 3x3 filter size and performs forward pass."""
    # Test Exp A/C shape: (2, 64, 1)
    model_ac = build_cnn_2d(input_shape=(2, 64, 1), n_dense_layers=2, kernel_size=3)
    assert model_ac.output_shape == (None, 1)

    dummy_ac = np.random.binomial(1, 0.5, size=(8, 2, 64, 1)).astype(np.float32)
    preds_ac = model_ac.predict(dummy_ac, verbose=0)
    assert preds_ac.shape == (8, 1)

    # Test Exp B shape: (4, 128, 1)
    model_b = build_cnn_2d(input_shape=(4, 128, 1), n_dense_layers=2, kernel_size=3)
    dummy_b = np.random.binomial(1, 0.5, size=(8, 4, 128, 1)).astype(np.float32)
    preds_b = model_b.predict(dummy_b, verbose=0)
    assert preds_b.shape == (8, 1)


def test_prepare_input_shapes():
    """Verify prepare_input produces exact expected shapes for all experiments."""
    N = 10
    # Exp A & C bit shape: (N, 128)
    X_bits_ac = np.random.randint(0, 2, size=(N, 128), dtype=np.uint8)

    assert prepare_input(X_bits_ac, "mlp", "a").shape == (N, 128)
    assert prepare_input(X_bits_ac, "cnn_1d", "a").shape == (N, 64, 2)
    assert prepare_input(X_bits_ac, "cnn_2d", "a").shape == (N, 2, 64, 1)

    # Exp B bit shape: (N, 4, 128)
    X_bits_b = np.random.randint(0, 2, size=(N, 4, 128), dtype=np.uint8)

    assert prepare_input(X_bits_b, "mlp", "b").shape == (N, 512)
    assert prepare_input(X_bits_b, "cnn_1d", "b").shape == (N, 64, 8)
    assert prepare_input(X_bits_b, "cnn_2d", "b").shape == (N, 4, 128, 1)
