"""Unit tests and official ISO/IEC 18033-3 test vectors for HIGHT block cipher."""

import pytest
import numpy as np
from ciphers.hight import (
    HIGHT,
    hight_encrypt,
    generate_key_schedule,
    hight_encrypt_batch,
    _DELTA_CONSTANTS
)


def test_hight_test_vector_1():
    """Verify HIGHT encryption against official test vector 1."""
    key = bytes.fromhex("00112233445566778899aabbccddeeff")
    pt = bytes.fromhex("0000000000000000")
    expected_ct = bytes.fromhex("00f418aed94f03f2")

    ct = hight_encrypt(pt, key)
    assert ct == expected_ct, f"Vector 1 failed: got {ct.hex()}, expected {expected_ct.hex()}"


def test_hight_test_vector_2():
    """Verify HIGHT encryption against official test vector 2."""
    key = bytes.fromhex("ffeeddccbbaa99887766554433221100")
    pt = bytes.fromhex("0011223344556677")
    expected_ct = bytes.fromhex("23ce9f72e543e6d8")

    ct = hight_encrypt(pt, key)
    assert ct == expected_ct, f"Vector 2 failed: got {ct.hex()}, expected {expected_ct.hex()}"


def test_hight_class_interface():
    """Verify HIGHT object-oriented interface."""
    key1 = bytes.fromhex("00112233445566778899aabbccddeeff")
    pt1 = bytes.fromhex("0000000000000000")
    expected_ct1 = bytes.fromhex("00f418aed94f03f2")

    cipher = HIGHT(key1)
    assert cipher.encrypt(pt1) == expected_ct1

    key2 = bytes.fromhex("ffeeddccbbaa99887766554433221100")
    pt2 = bytes.fromhex("0011223344556677")
    expected_ct2 = bytes.fromhex("23ce9f72e543e6d8")

    cipher.set_key(key2)
    assert cipher.encrypt(pt2) == expected_ct2


def test_hight_batch_encryption():
    """Verify vectorized batch encryption matches single block encryption."""
    key = bytes.fromhex("ffeeddccbbaa99887766554433221100")
    cipher = HIGHT(key)

    pts = [
        bytes.fromhex("0011223344556677"),
        bytes.fromhex("0000000000000000"),
        bytes.fromhex("ffffffffffffffff"),
        bytes.fromhex("0123456789abcdef"),
    ]

    expected_cts = [cipher.encrypt(pt) for pt in pts]

    pt_batch = np.array([[b for b in pt] for pt in pts], dtype=np.uint8)
    ct_batch = cipher.encrypt_batch(pt_batch)

    for i, expected_ct in enumerate(expected_cts):
        assert bytes(ct_batch[i].tolist()) == expected_ct
