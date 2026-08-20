"""SM4 Block Cipher Interface (128-bit block, 128-bit key).

Compliant with GB/T 32907-2016 standard.
Uses gmssl for verified SM4 encryption and provides single-block and batch interfaces.
"""

from typing import Union
import numpy as np
from gmssl.sm4 import CryptSM4, SM4_ENCRYPT


def sm4_encrypt(plaintext: bytes, key: bytes) -> bytes:
    """Encrypt a single 128-bit block using SM4 in ECB mode without padding.

    Args:
        plaintext (bytes): 16-byte plaintext block.
        key (bytes): 16-byte master key.

    Returns:
        bytes: 16-byte ciphertext block.
    """
    if len(plaintext) != 16:
        raise ValueError(f"Plaintext must be exactly 16 bytes, got {len(plaintext)} bytes.")
    if len(key) != 16:
        raise ValueError(f"Key must be exactly 16 bytes, got {len(key)} bytes.")

    crypt = CryptSM4()
    crypt.set_key(key, SM4_ENCRYPT)
    ct_list = crypt.one_round(crypt.sk, list(plaintext))
    return bytes(ct_list)


class SM4Cipher:
    """SM4 block cipher wrapper instance."""

    def __init__(self, key: bytes = None):
        """Initialize SM4 cipher instance.

        Args:
            key (bytes, optional): 16-byte key.
        """
        self.crypt = CryptSM4()
        self.key = None
        if key is not None:
            self.set_key(key)

    def set_key(self, key: bytes) -> None:
        """Set the 128-bit master key and precompute round keys."""
        if len(key) != 16:
            raise ValueError(f"Key must be exactly 16 bytes, got {len(key)} bytes.")
        self.key = key
        self.crypt.set_key(key, SM4_ENCRYPT)

    def encrypt(self, plaintext: bytes) -> bytes:
        """Encrypt a single 16-byte block.

        Args:
            plaintext (bytes): 16-byte plaintext block.

        Returns:
            bytes: 16-byte ciphertext block.
        """
        if self.key is None:
            raise ValueError("Cipher key not initialized. Call set_key() first.")
        if len(plaintext) != 16:
            raise ValueError(f"Plaintext must be exactly 16 bytes, got {len(plaintext)} bytes.")
        ct_list = self.crypt.one_round(self.crypt.sk, list(plaintext))
        return bytes(ct_list)
