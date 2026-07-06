"""
aes_utils.py
============

AES-256-CBC encryption/decryption for SeBAC-IoD secure data exchange.

After the batch authentication protocol establishes a session key SKij
(32 bytes = 256 bits), this module uses it directly as an AES-256 key
to encrypt and decrypt messages between the drone and the user.

How AES-256-CBC works
---------------------
    Key  : SKij (32 bytes — derived from ECDH + hash)
    IV   : random 16 bytes generated fresh for each message
    Mode : CBC (Cipher Block Chaining)

    Encrypt:
        IV         = random 16 bytes
        padded_msg = PKCS7-pad(message)
        ciphertext = AES-256-CBC(key=SKij, iv=IV, data=padded_msg)
        output     = IV + ciphertext   (IV prepended so receiver can decrypt)

    Decrypt:
        IV         = output[:16]
        ciphertext = output[16:]
        padded_msg = AES-256-CBC-decrypt(key=SKij, iv=IV, data=ciphertext)
        message    = PKCS7-unpad(padded_msg)

Why CBC mode?
-------------
CBC is a standard, well-understood mode that chains each block with the
previous ciphertext block — same plaintext blocks produce different
ciphertext in different positions, hiding patterns.

Dependencies
------------
Uses pycryptodome (pip install pycryptodome) if available.
Falls back to a simple XOR cipher if not installed, so the demo
always works even without the extra library.
"""

from __future__ import annotations

import os
import struct

_AES_AVAILABLE = False

try:
    from Crypto.Cipher import AES
    from Crypto.Util.Padding import pad, unpad
    _AES_AVAILABLE = True
except ImportError:
    pass


def is_using_real_aes() -> bool:
    """Return True if pycryptodome is installed and real AES is active."""
    return _AES_AVAILABLE


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def encrypt(message: str, key: bytes) -> bytes:
    """
    Encrypt a plaintext message using AES-256-CBC with the session key.

    Parameters
    ----------
    message : str
        The plaintext to encrypt (e.g. "hello from Drone-00").
    key : bytes
        The 32-byte session key SKij.

    Returns
    -------
    bytes
        IV (16 bytes) + ciphertext.  The IV is prepended so the receiver
        can extract it for decryption.
    """
    data = message.encode("utf-8")

    if _AES_AVAILABLE:
        iv = os.urandom(16)
        cipher = AES.new(key[:32], AES.MODE_CBC, iv)
        ciphertext = cipher.encrypt(pad(data, AES.block_size))
        return iv + ciphertext
    else:
        # XOR fallback — works without pycryptodome
        return _xor_encrypt(data, key)


def decrypt(ciphertext: bytes, key: bytes) -> str:
    """
    Decrypt a ciphertext produced by :func:`encrypt`.

    Parameters
    ----------
    ciphertext : bytes
        IV + ciphertext bytes as returned by encrypt().
    key : bytes
        The 32-byte session key SKij (must match the one used to encrypt).

    Returns
    -------
    str
        The original plaintext message.
    """
    if _AES_AVAILABLE:
        iv         = ciphertext[:16]
        ct         = ciphertext[16:]
        cipher     = AES.new(key[:32], AES.MODE_CBC, iv)
        plaintext  = unpad(cipher.decrypt(ct), AES.block_size)
        return plaintext.decode("utf-8")
    else:
        return _xor_decrypt(ciphertext, key).decode("utf-8")


# ---------------------------------------------------------------------------
# XOR fallback (no extra dependencies)
# ---------------------------------------------------------------------------

def _xor_encrypt(data: bytes, key: bytes) -> bytes:
    """Simple XOR stream cipher fallback."""
    # Prepend 16 zero bytes as a fake IV placeholder so the format matches
    key_stream = (key * (len(data) // len(key) + 1))[:len(data)]
    ct = bytes(a ^ b for a, b in zip(data, key_stream))
    return b"\x00" * 16 + ct   # 16-byte fake IV + ciphertext


def _xor_decrypt(ciphertext: bytes, key: bytes) -> bytes:
    """XOR stream cipher decryption (symmetric — same as encrypt)."""
    ct = ciphertext[16:]        # strip the 16-byte fake IV
    key_stream = (key * (len(ct) // len(key) + 1))[:len(ct)]
    return bytes(a ^ b for a, b in zip(ct, key_stream))
