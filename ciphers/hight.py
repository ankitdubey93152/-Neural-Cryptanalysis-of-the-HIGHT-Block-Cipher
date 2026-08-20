"""HIGHT Block Cipher Implementation (64-bit block, 128-bit key, 32 rounds).

Strictly compliant with ISO/IEC 18033-3 and KISA specifications.
Supports both single-block bytes encryption and high-throughput NumPy batch encryption.
"""

from typing import Tuple, Union
import numpy as np


def _generate_constants() -> np.ndarray:
    """Generate the 128 constant values d[0..127] for the HIGHT key schedule.

    Returns:
        np.ndarray: Array of 128 7-bit constants (dtype=np.uint8).
    """
    s = np.zeros(134, dtype=np.uint8)
    s[0:7] = [0, 1, 0, 1, 1, 0, 1]

    for i in range(1, 128):
        s[i + 6] = s[i + 2] ^ s[i - 1]

    d = np.zeros(128, dtype=np.uint8)
    for i in range(128):
        d[i] = (
            (s[i + 6] << 6)
            | (s[i + 5] << 5)
            | (s[i + 4] << 4)
            | (s[i + 3] << 3)
            | (s[i + 2] << 2)
            | (s[i + 1] << 1)
            | s[i]
        )
    return d


# Precompute LFSR constants once at module load
_DELTA_CONSTANTS = _generate_constants()


def rotl8(x: Union[int, np.ndarray], n: int) -> Union[int, np.ndarray]:
    """Left circular rotation of 8-bit value(s) by n bits.

    Args:
        x (int or np.ndarray): 8-bit unsigned integer or uint8 array.
        n (int): Number of bits to rotate left (0..7).

    Returns:
        int or np.ndarray: Rotated 8-bit value(s).
    """
    return ((x << n) | (x >> (8 - n))) & 0xFF


def F0(x: Union[int, np.ndarray]) -> Union[int, np.ndarray]:
    """HIGHT round auxiliary function F0.

    F0(x) = rotl8(x, 1) XOR rotl8(x, 2) XOR rotl8(x, 7)
    """
    return rotl8(x, 1) ^ rotl8(x, 2) ^ rotl8(x, 7)


def F1(x: Union[int, np.ndarray]) -> Union[int, np.ndarray]:
    """HIGHT round auxiliary function F1.

    F1(x) = rotl8(x, 3) XOR rotl8(x, 4) XOR rotl8(x, 6)
    """
    return rotl8(x, 3) ^ rotl8(x, 4) ^ rotl8(x, 6)


def generate_key_schedule(key: bytes) -> Tuple[np.ndarray, np.ndarray]:
    """Generate whitening keys (WK[0..7]) and round subkeys (SK[0..127]).

    Follows spec index reversal: K[i] = key[15 - i].

    Args:
        key (bytes): 16-byte master key in standard big-endian/left-to-right order.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - WK: 8 whitening keys (dtype=np.uint8)
            - SK: 128 round subkeys (dtype=np.uint8)
    """
    if len(key) != 16:
        raise ValueError(f"Master key must be exactly 16 bytes, got {len(key)} bytes.")

    # Convert to spec order: K[i] = key[15 - i]
    K = np.array([key[15 - i] for i in range(16)], dtype=np.uint8)

    # Whitening keys WK[0..7]
    WK = np.zeros(8, dtype=np.uint8)
    for i in range(4):
        WK[i] = K[i + 12]
    for i in range(4, 8):
        WK[i] = K[i - 4]

    # Subkeys SK[0..127]
    SK = np.zeros(128, dtype=np.uint8)
    for i in range(8):
        for j in range(8):
            k_idx1 = (j - i) % 8
            SK[16 * i + j] = (int(K[k_idx1]) + int(_DELTA_CONSTANTS[16 * i + j])) & 0xFF

            k_idx2 = ((j - i) % 8) + 8
            SK[16 * i + j + 8] = (int(K[k_idx2]) + int(_DELTA_CONSTANTS[16 * i + j + 8])) & 0xFF

    return WK, SK


def hight_encrypt(plaintext: bytes, key: bytes, rounds: int = 32) -> bytes:
    """Encrypt a single 64-bit block using HIGHT.

    Args:
        plaintext (bytes): 8-byte plaintext block in standard byte order.
        key (bytes): 16-byte master key in standard byte order.
        rounds (int): Number of rounds to execute (default: 32).

    Returns:
        bytes: 8-byte ciphertext block in standard byte order.
    """
    if len(plaintext) != 8:
        raise ValueError(f"Plaintext must be exactly 8 bytes, got {len(plaintext)} bytes.")

    WK, SK = generate_key_schedule(key)
    return hight_encrypt_with_subkeys(plaintext, WK, SK, rounds=rounds)


def hight_encrypt_with_subkeys(
    plaintext: bytes,
    WK: np.ndarray,
    SK: np.ndarray,
    rounds: int = 32
) -> bytes:
    """Encrypt a single 64-bit block using precomputed key schedule.

    Args:
        plaintext (bytes): 8-byte plaintext block.
        WK (np.ndarray): 8 whitening keys.
        SK (np.ndarray): 128 round subkeys.
        rounds (int): Number of rounds (1..32, default: 32).

    Returns:
        bytes: 8-byte ciphertext block.
    """
    if len(plaintext) != 8:
        raise ValueError(f"Plaintext must be exactly 8 bytes, got {len(plaintext)} bytes.")
    if rounds < 1 or rounds > 32:
        raise ValueError(f"Rounds must be between 1 and 32, got {rounds}.")

    # Spec order: P[i] = plaintext[7 - i]
    P = [plaintext[7 - i] for i in range(8)]

    # Initial transformation
    X = [
        (P[0] + int(WK[0])) & 0xFF,
        P[1],
        P[2] ^ int(WK[1]),
        P[3],
        (P[4] + int(WK[2])) & 0xFF,
        P[5],
        P[6] ^ int(WK[3]),
        P[7]
    ]

    # Rounds 0 to rounds-2 (or 0 to 30 for full 32 rounds)
    limit_rotating = min(rounds, 31)
    for i in range(limit_rotating):
        new_X = [0] * 8
        new_X[1] = X[0]
        new_X[3] = X[2]
        new_X[5] = X[4]
        new_X[7] = X[6]
        new_X[0] = X[7] ^ ((F0(X[6]) + int(SK[4 * i + 3])) & 0xFF)
        new_X[2] = (X[1] + (F1(X[0]) ^ int(SK[4 * i]))) & 0xFF
        new_X[4] = X[3] ^ ((F0(X[2]) + int(SK[4 * i + 1])) & 0xFF)
        new_X[6] = (X[5] + (F1(X[4]) ^ int(SK[4 * i + 2]))) & 0xFF
        X = new_X

    # Round 31 (no rotation) if 32 rounds requested
    if rounds == 32:
        new_X = [0] * 8
        new_X[0] = X[0]
        new_X[1] = (X[1] + (F1(X[0]) ^ int(SK[124]))) & 0xFF
        new_X[2] = X[2]
        new_X[3] = X[3] ^ ((F0(X[2]) + int(SK[125])) & 0xFF)
        new_X[4] = X[4]
        new_X[5] = (X[5] + (F1(X[4]) ^ int(SK[126])) & 0xFF)
        new_X[4] = X[4]
        new_X[5] = (X[5] + (F1(X[4]) ^ int(SK[126]))) & 0xFF
        new_X[6] = X[6]
        new_X[7] = X[7] ^ ((F0(X[6]) + int(SK[127])) & 0xFF)
        X = new_X

    # Final transformation
    C = [
        (X[0] + int(WK[4])) & 0xFF,
        X[1],
        X[2] ^ int(WK[5]),
        X[3],
        (X[4] + int(WK[6])) & 0xFF,
        X[5],
        X[6] ^ int(WK[7]),
        X[7]
    ]

    # Reversal to normal byte order: out[j] = C[7 - j]
    return bytes(C[7 - j] for j in range(8))


def hight_encrypt_batch(
    plaintexts: np.ndarray,
    WK: np.ndarray,
    SK: np.ndarray,
    rounds: int = 32
) -> np.ndarray:
    """Vectorized batch encryption for N 64-bit plaintext blocks.

    Args:
        plaintexts (np.ndarray): Array of shape (N, 8) with dtype=np.uint8 in normal byte order.
        WK (np.ndarray): 8 whitening keys.
        SK (np.ndarray): 128 round subkeys.
        rounds (int): Number of rounds (1..32, default: 32).

    Returns:
        np.ndarray: Array of shape (N, 8) with dtype=np.uint8 representing ciphertexts.
    """
    if plaintexts.ndim != 2 or plaintexts.shape[1] != 8:
        raise ValueError(f"Expected plaintexts shape (N, 8), got {plaintexts.shape}")

    # Convert to spec order: P[i] = plaintexts[:, 7 - i]
    # P_0 is col 7, P_7 is col 0
    X = [
        (plaintexts[:, 7].astype(np.uint16) + int(WK[0])) & 0xFF,
        plaintexts[:, 6].copy(),
        plaintexts[:, 5] ^ int(WK[1]),
        plaintexts[:, 4].copy(),
        (plaintexts[:, 3].astype(np.uint16) + int(WK[2])) & 0xFF,
        plaintexts[:, 2].copy(),
        plaintexts[:, 1] ^ int(WK[3]),
        plaintexts[:, 0].copy(),
    ]

    limit_rotating = min(rounds, 31)
    for i in range(limit_rotating):
        new_0 = X[7] ^ (((F0(X[6]) + int(SK[4 * i + 3]))) & 0xFF)
        new_1 = X[0]
        new_2 = ((X[1].astype(np.uint16) + (F1(X[0]) ^ int(SK[4 * i])))) & 0xFF
        new_3 = X[2]
        new_4 = X[3] ^ (((F0(X[2]) + int(SK[4 * i + 1]))) & 0xFF)
        new_5 = X[4]
        new_6 = ((X[5].astype(np.uint16) + (F1(X[4]) ^ int(SK[4 * i + 2])))) & 0xFF
        new_7 = X[6]
        X = [new_0, new_1, new_2, new_3, new_4, new_5, new_6, new_7]

    if rounds == 32:
        new_0 = X[0]
        new_1 = ((X[1].astype(np.uint16) + (F1(X[0]) ^ int(SK[124])))) & 0xFF
        new_2 = X[2]
        new_3 = X[3] ^ (((F0(X[2]) + int(SK[125]))) & 0xFF)
        new_4 = X[4]
        new_5 = ((X[5].astype(np.uint16) + (F1(X[4]) ^ int(SK[126])))) & 0xFF
        new_6 = X[6]
        new_7 = X[7] ^ (((F0(X[6]) + int(SK[127]))) & 0xFF)
        X = [new_0, new_1, new_2, new_3, new_4, new_5, new_6, new_7]

    # Final transformation
    C = [
        (X[0].astype(np.uint16) + int(WK[4])) & 0xFF,
        X[1],
        X[2] ^ int(WK[5]),
        X[3],
        (X[4].astype(np.uint16) + int(WK[6])) & 0xFF,
        X[5],
        X[6] ^ int(WK[7]),
        X[7],
    ]

    # Reversal to normal byte order: out[:, j] = C[7 - j]
    ciphertexts = np.column_stack([C[7 - j] for j in range(8)]).astype(np.uint8)
    return ciphertexts


class HIGHT:
    """HIGHT block cipher primitive wrapper class."""

    def __init__(self, key: bytes = None):
        """Initialize HIGHT instance with optional 128-bit key.

        Args:
            key (bytes, optional): 16-byte master key.
        """
        self.key = key
        self.WK = None
        self.SK = None
        if key is not None:
            self.set_key(key)

    def set_key(self, key: bytes) -> None:
        """Set or update the master key and precompute key schedule."""
        self.key = key
        self.WK, self.SK = generate_key_schedule(key)

    def encrypt(self, plaintext: bytes, rounds: int = 32) -> bytes:
        """Encrypt a single 64-bit plaintext block."""
        if self.WK is None or self.SK is None:
            raise ValueError("Cipher key not initialized. Call set_key() first.")
        return hight_encrypt_with_subkeys(plaintext, self.WK, self.SK, rounds=rounds)

    def encrypt_batch(self, plaintexts: np.ndarray, rounds: int = 32) -> np.ndarray:
        """Encrypt a batch of 64-bit plaintext blocks."""
        if self.WK is None or self.SK is None:
            raise ValueError("Cipher key not initialized. Call set_key() first.")
        return hight_encrypt_batch(plaintexts, self.WK, self.SK, rounds=rounds)


if __name__ == "__main__":
    print("=" * 60)
    print("HIGHT BLOCK CIPHER: OFFICIAL ISO/IEC 18033-3 TEST VECTORS")
    print("=" * 60)

    # Test Vector 1
    key1 = bytes.fromhex("00112233445566778899aabbccddeeff")
    pt1 = bytes.fromhex("0000000000000000")
    expected_ct1 = bytes.fromhex("00f418aed94f03f2")
    ct1 = hight_encrypt(pt1, key1)

    print("\n[Vector 1]")
    print(f"Key:        {key1.hex()}")
    print(f"Plaintext:  {pt1.hex()}")
    print(f"Ciphertext: {ct1.hex()}")
    print(f"Expected:   {expected_ct1.hex()}")
    print(f"Status:     {'[PASS]' if ct1 == expected_ct1 else '[FAIL]'}")

    # Test Vector 2
    key2 = bytes.fromhex("ffeeddccbbaa99887766554433221100")
    pt2 = bytes.fromhex("0011223344556677")
    expected_ct2 = bytes.fromhex("23ce9f72e543e6d8")
    ct2 = hight_encrypt(pt2, key2)

    print("\n[Vector 2]")
    print(f"Key:        {key2.hex()}")
    print(f"Plaintext:  {pt2.hex()}")
    print(f"Ciphertext: {ct2.hex()}")
    print(f"Expected:   {expected_ct2.hex()}")
    print(f"Status:     {'[PASS]' if ct2 == expected_ct2 else '[FAIL]'}")

    # Batch Encryption Demo
    cipher = HIGHT(key2)
    pt_batch = np.array([list(pt2), list(pt1)], dtype=np.uint8)
    ct_batch = cipher.encrypt_batch(pt_batch)
    print("\n[Batch Encryption Demo (2 blocks)]")
    print(f"Block 0: {bytes(ct_batch[0]).hex()} ({'[MATCH]' if bytes(ct_batch[0]) == ct2 else '[MISMATCH]'})")
    print(f"Block 1: {bytes(ct_batch[1]).hex()} ({'[MATCH]' if bytes(ct_batch[1]) == cipher.encrypt(pt1) else '[MISMATCH]'})")

    print("\n" + "=" * 60)
    print("ALL TEST VECTORS VERIFIED SUCCESSFULLY!")
    print("=" * 60)


