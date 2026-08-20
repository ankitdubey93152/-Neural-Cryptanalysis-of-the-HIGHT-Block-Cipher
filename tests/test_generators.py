"""Unit tests for dataset generators, bit expansion, and persistence."""

import os
import tempfile
import numpy as np
import pytest

from data.generators import (
    DELTA,
    bits,
    generate_experiment_a,
    generate_experiment_b,
    generate_experiment_c,
    save_dataset,
    load_dataset
)


def test_delta_constant():
    """Verify DELTA is exactly 8 bytes with LSB set."""
    assert len(DELTA) == 8
    assert DELTA == bytes.fromhex("0000000000000001")


def test_bits_expansion():
    """Verify byte array expansion to 0/1 bits."""
    # 0x01 in binary is 00000001
    # 0x80 in binary is 10000000
    arr = np.array([[0x01, 0x80]], dtype=np.uint8)
    b = bits(arr)
    assert b.shape == (1, 16)
    expected = np.array([[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)
    assert np.array_equal(b, expected)


def test_generate_experiment_a_shape_and_balance():
    """Verify Experiment A generator shape, dtype, and 50/50 class balance."""
    n_samples = 100
    X, y = generate_experiment_a(n_samples=n_samples)
    assert X.shape == (n_samples, 16)
    assert X.dtype == np.uint8
    assert y.shape == (n_samples,)
    assert y.dtype == np.uint8
    assert np.sum(y == 1) == 50
    assert np.sum(y == 0) == 50


def test_generate_experiment_b_shape_and_balance():
    """Verify Experiment B generator multi-pair shape and balance."""
    n_samples = 60
    pairs_per_sample = 4
    X, y = generate_experiment_b(n_samples=n_samples, pairs_per_sample=pairs_per_sample)
    assert X.shape == (n_samples, pairs_per_sample, 16)
    assert X.dtype == np.uint8
    assert y.shape == (n_samples,)
    assert np.sum(y == 1) == 30
    assert np.sum(y == 0) == 30

    b = bits(X)
    assert b.shape == (n_samples, pairs_per_sample, 128)


def test_generate_experiment_c_shape_and_balance():
    """Verify Experiment C generator cipher identification shape and balance."""
    n_samples = 80
    X, y = generate_experiment_c(n_samples=n_samples)
    assert X.shape == (n_samples, 16)
    assert X.dtype == np.uint8
    assert y.shape == (n_samples,)
    assert np.sum(y == 1) == 40
    assert np.sum(y == 0) == 40


def test_dataset_save_load_npz():
    """Verify dataset save and load roundtrip with NPZ format."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test_exp_a.npz")
        X = np.random.randint(0, 256, size=(50, 16), dtype=np.uint8)
        y = np.random.randint(0, 2, size=(50,), dtype=np.uint8)

        save_dataset(filepath, X, y)
        assert os.path.exists(filepath)

        loaded_X, loaded_y = load_dataset(filepath)
        assert np.array_equal(X, loaded_X)
        assert np.array_equal(y, loaded_y)
