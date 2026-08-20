"""Dataset generation routines for neural cryptanalysis experiments.

Implements data generation for:
- Experiment A: Single ciphertext pair differential distinguisher (Fixed Δ vs random).
- Experiment B: Bundled multi-pair differential distinguisher (4 pairs under same key).
- Experiment C: Cipher identification distinguisher (HIGHT vs SM4 given fixed Δ).

Also provides bit-expansion utility and NPZ dataset persistence.
"""

import os
import sys
from typing import Tuple, Optional
import numpy as np
from tqdm import tqdm

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ciphers.hight import hight_encrypt_with_subkeys, generate_key_schedule
from ciphers.sm4 import SM4Cipher


# Fixed 64-bit input difference: flip only the least significant bit (LSB)
DELTA = bytes.fromhex("0000000000000001")
DELTA_ARR = np.array([0, 0, 0, 0, 0, 0, 0, 1], dtype=np.uint8)


def bits(byte_array: np.ndarray) -> np.ndarray:
    """Expand a uint8 byte array into an array of individual bits (0 or 1).

    Args:
        byte_array (np.ndarray): NumPy array with dtype=np.uint8.

    Returns:
        np.ndarray: Array with same prefix shape and unpacked 8x bits on the last axis.
    """
    if byte_array.dtype != np.uint8:
        byte_array = byte_array.astype(np.uint8)
    return np.unpackbits(byte_array, axis=-1)


def generate_experiment_a(
    n_samples: int,
    delta: bytes = DELTA,
    show_progress: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate balanced dataset for Experiment A (Single-pair differential distinguisher).

    For each sample:
      - A fresh random 128-bit key is generated.
      - P1 is a random 64-bit plaintext.
      - If Y=1: P2 = P1 XOR delta.
      - If Y=0: P2 = P1 XOR (fresh random 64-bit non-zero difference).
      - C1, C2 = HIGHT(P1, key), HIGHT(P2, key).

    Args:
        n_samples (int): Total number of samples (will be balanced 50/50).
        delta (bytes): 8-byte fixed difference.
        show_progress (bool): Whether to display tqdm progress bar.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - X: (n_samples, 16) uint8 array representing (C1 || C2).
            - y: (n_samples,) uint8 array of binary labels (0 or 1).
    """
    delta_bytes = np.frombuffer(delta, dtype=np.uint8)
    half = n_samples // 2
    n_samples = half * 2  # ensure even count

    y = np.zeros(n_samples, dtype=np.uint8)
    y[:half] = 1

    # Shuffle labels
    perm = np.random.permutation(n_samples)
    y = y[perm]

    X = np.zeros((n_samples, 16), dtype=np.uint8)

    iterator = range(n_samples)
    if show_progress:
        iterator = tqdm(iterator, desc="Generating Exp A data")

    # Generate random keys and plaintexts in bulk
    keys = np.random.randint(0, 256, size=(n_samples, 16), dtype=np.uint8)
    p1s = np.random.randint(0, 256, size=(n_samples, 8), dtype=np.uint8)
    random_diffs = np.random.randint(0, 256, size=(n_samples, 8), dtype=np.uint8)

    for i in iterator:
        key_bytes = bytes(keys[i])
        WK, SK = generate_key_schedule(key_bytes)

        p1 = p1s[i]
        if y[i] == 1:
            p2 = p1 ^ delta_bytes
        else:
            p2 = p1 ^ random_diffs[i]

        c1 = hight_encrypt_with_subkeys(bytes(p1), WK, SK)
        c2 = hight_encrypt_with_subkeys(bytes(p2), WK, SK)

        X[i, :8] = np.frombuffer(c1, dtype=np.uint8)
        X[i, 8:] = np.frombuffer(c2, dtype=np.uint8)

    return X, y


def generate_experiment_b(
    n_samples: int,
    pairs_per_sample: int = 4,
    delta: bytes = DELTA,
    show_progress: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate balanced dataset for Experiment B (Bundled multi-pair differential distinguisher).

    For each sample:
      - A fresh random 128-bit key is generated.
      - 4 plaintext pairs are generated under the SAME key.
      - If Y=1: all 4 pairs have P2_k = P1_k XOR delta.
      - If Y=0: all 4 pairs have P2_k = P1_k XOR (fresh random diff_k).
      - All pairs encrypted with HIGHT under the shared key.

    Args:
        n_samples (int): Total number of bundled samples.
        pairs_per_sample (int): Number of ciphertext pairs per sample (default: 4).
        delta (bytes): 8-byte fixed difference.
        show_progress (bool): Whether to display progress bar.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - X: (n_samples, pairs_per_sample, 16) uint8 array representing bundled pairs.
            - y: (n_samples,) uint8 array of binary labels (0 or 1).
    """
    delta_bytes = np.frombuffer(delta, dtype=np.uint8)
    half = n_samples // 2
    n_samples = half * 2

    y = np.zeros(n_samples, dtype=np.uint8)
    y[:half] = 1

    perm = np.random.permutation(n_samples)
    y = y[perm]

    X = np.zeros((n_samples, pairs_per_sample, 16), dtype=np.uint8)

    iterator = range(n_samples)
    if show_progress:
        iterator = tqdm(iterator, desc="Generating Exp B data")

    keys = np.random.randint(0, 256, size=(n_samples, 16), dtype=np.uint8)
    p1s = np.random.randint(0, 256, size=(n_samples, pairs_per_sample, 8), dtype=np.uint8)
    random_diffs = np.random.randint(0, 256, size=(n_samples, pairs_per_sample, 8), dtype=np.uint8)

    for i in iterator:
        key_bytes = bytes(keys[i])
        WK, SK = generate_key_schedule(key_bytes)

        p1_bundle = p1s[i]
        if y[i] == 1:
            p2_bundle = p1_bundle ^ delta_bytes
        else:
            p2_bundle = p1_bundle ^ random_diffs[i]

        for k in range(pairs_per_sample):
            c1 = hight_encrypt_with_subkeys(bytes(p1_bundle[k]), WK, SK)
            c2 = hight_encrypt_with_subkeys(bytes(p2_bundle[k]), WK, SK)
            X[i, k, :8] = np.frombuffer(c1, dtype=np.uint8)
            X[i, k, 8:] = np.frombuffer(c2, dtype=np.uint8)

    return X, y


def generate_experiment_c(
    n_samples: int,
    delta: bytes = DELTA,
    show_progress: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate balanced dataset for Experiment C (Cipher Identification: HIGHT vs SM4).

    For each sample:
      - Fixed difference Δ is applied: P2 = P1 XOR delta.
      - If Y=1: Encrypted using HIGHT under a fresh random 128-bit key.
      - If Y=0: Encrypted using SM4 (truncated to first 8 bytes) under a fresh random 128-bit key.

    Args:
        n_samples (int): Total number of samples.
        delta (bytes): 8-byte fixed difference.
        show_progress (bool): Whether to display progress bar.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - X: (n_samples, 16) uint8 array representing (C1 || C2).
            - y: (n_samples,) uint8 array of binary labels (1=HIGHT, 0=SM4).
    """
    delta_bytes = np.frombuffer(delta, dtype=np.uint8)
    half = n_samples // 2
    n_samples = half * 2

    y = np.zeros(n_samples, dtype=np.uint8)
    y[:half] = 1

    perm = np.random.permutation(n_samples)
    y = y[perm]

    X = np.zeros((n_samples, 16), dtype=np.uint8)

    iterator = range(n_samples)
    if show_progress:
        iterator = tqdm(iterator, desc="Generating Exp C data")

    keys = np.random.randint(0, 256, size=(n_samples, 16), dtype=np.uint8)
    p1s = np.random.randint(0, 256, size=(n_samples, 8), dtype=np.uint8)

    for i in iterator:
        key_bytes = bytes(keys[i])
        p1 = p1s[i]
        p2 = p1 ^ delta_bytes

        if y[i] == 1:
            # HIGHT encryption
            WK, SK = generate_key_schedule(key_bytes)
            c1 = hight_encrypt_with_subkeys(bytes(p1), WK, SK)
            c2 = hight_encrypt_with_subkeys(bytes(p2), WK, SK)
            X[i, :8] = np.frombuffer(c1, dtype=np.uint8)
            X[i, 8:] = np.frombuffer(c2, dtype=np.uint8)
        else:
            # SM4 encryption (zero-pad 8-byte plaintexts to 16 bytes, truncate 16-byte ciphertexts to 8 bytes)
            sm4_cipher = SM4Cipher(key_bytes)
            p1_16 = bytes(p1) + b"\x00" * 8
            p2_16 = bytes(p2) + b"\x00" * 8
            ct1_16 = sm4_cipher.encrypt(p1_16)
            ct2_16 = sm4_cipher.encrypt(p2_16)
            X[i, :8] = np.frombuffer(ct1_16[:8], dtype=np.uint8)
            X[i, 8:] = np.frombuffer(ct2_16[:8], dtype=np.uint8)

    return X, y


def save_dataset(filepath: str, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
    """Save dataset arrays to a compressed NumPy .npz file.

    Args:
        filepath (str): Output filepath (.npz extension).
        X (np.ndarray): Feature array.
        y (np.ndarray): Label array.
        **kwargs: Additional metadata arrays or parameters to store.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    np.savez_compressed(filepath, X=X, y=y, **kwargs)


def load_dataset(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load dataset arrays from a compressed NumPy .npz file.

    Args:
        filepath (str): Path to .npz file.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (X, y) arrays.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found: {filepath}")
    data = np.load(filepath)
    return data["X"], data["y"]


if __name__ == "__main__":
    from utils.seeding import set_global_seed

    set_global_seed(42)
    sample_size = 10000

    print("=" * 60)
    print("SANITY CHECK: GENERATING 10,000 SAMPLES FOR EXPERIMENTS A, B, C")
    print("=" * 60)

    # Experiment A
    print("\n--- Generating Experiment A (Single Pair) ---")
    Xa, ya = generate_experiment_a(sample_size, show_progress=False)
    print(f"X shape: {Xa.shape}, dtype: {Xa.dtype}")
    print(f"y shape: {ya.shape}, dtype: {ya.dtype}")
    print(f"Class balance: Y=1: {np.sum(ya == 1)} ({np.mean(ya == 1):.2%}), Y=0: {np.sum(ya == 0)} ({np.mean(ya == 0):.2%})")
    print("Example Row 0 (hex):", bytes(Xa[0]).hex(), "Label:", ya[0])
    print("Example Row 1 (hex):", bytes(Xa[1]).hex(), "Label:", ya[1])
    Xa_bits = bits(Xa)
    print(f"Bits shape: {Xa_bits.shape}, sample bits row 0 [:16]:", Xa_bits[0, :16])

    # Experiment B
    print("\n--- Generating Experiment B (4 Bundled Pairs) ---")
    Xb, yb = generate_experiment_b(sample_size, pairs_per_sample=4, show_progress=False)
    print(f"X shape: {Xb.shape}, dtype: {Xb.dtype}")
    print(f"y shape: {yb.shape}, dtype: {yb.dtype}")
    print(f"Class balance: Y=1: {np.sum(yb == 1)} ({np.mean(yb == 1):.2%}), Y=0: {np.sum(yb == 0)} ({np.mean(yb == 0):.2%})")
    print("Example Row 0 Pair 0 (hex):", bytes(Xb[0, 0]).hex(), "Label:", yb[0])
    Xb_bits = bits(Xb)
    print(f"Bits shape: {Xb_bits.shape}")

    # Experiment C
    print("\n--- Generating Experiment C (Cipher ID: HIGHT vs SM4) ---")
    Xc, yc = generate_experiment_c(sample_size, show_progress=False)
    print(f"X shape: {Xc.shape}, dtype: {Xc.dtype}")
    print(f"y shape: {yc.shape}, dtype: {yc.dtype}")
    print(f"Class balance: Y=1 (HIGHT): {np.sum(yc == 1)} ({np.mean(yc == 1):.2%}), Y=0 (SM4): {np.sum(yc == 0)} ({np.mean(yc == 0):.2%})")
    print("Example Row 0 (hex):", bytes(Xc[0]).hex(), "Label:", yc[0])
    print("Example Row 1 (hex):", bytes(Xc[1]).hex(), "Label:", yc[1])
    Xc_bits = bits(Xc)
    print(f"Bits shape: {Xc_bits.shape}")

    print("\n" + "=" * 60)
    print("ALL GENERATORS SANITY CHECK PASSED SUCCESSFULLY!")
    print("=" * 60)
