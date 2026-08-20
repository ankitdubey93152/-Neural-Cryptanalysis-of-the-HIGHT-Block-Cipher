"""Unit tests and official GB/T 32907-2016 test vectors for SM4 block cipher."""

import pytest
from ciphers.sm4 import SM4Cipher, sm4_encrypt


def test_sm4_official_test_vector():
    """Verify SM4 encryption against official standard test vector.

    Key:       01 23 45 67 89 ab cd ef fe dc ba 98 76 54 32 10
    Plaintext: 01 23 45 67 89 ab cd ef fe dc ba 98 76 54 32 10
    Expected:  68 1e df 34 d2 06 96 5e 86 b3 e9 4f 53 6e 42 46
    """
    key = bytes.fromhex("0123456789abcdeffedcba9876543210")
    pt = bytes.fromhex("0123456789abcdeffedcba9876543210")
    expected_ct = bytes.fromhex("681edf34d206965e86b3e94f536e4246")

    ct = sm4_encrypt(pt, key)
    assert ct == expected_ct, f"SM4 vector failed: got {ct.hex()}, expected {expected_ct.hex()}"


def test_sm4_class_interface():
    """Verify SM4Cipher class wrapper."""
    key = bytes.fromhex("0123456789abcdeffedcba9876543210")
    pt = bytes.fromhex("0123456789abcdeffedcba9876543210")
    expected_ct = bytes.fromhex("681edf34d206965e86b3e94f536e4246")

    cipher = SM4Cipher(key)
    assert cipher.encrypt(pt) == expected_ct
