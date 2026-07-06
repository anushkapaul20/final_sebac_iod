"""
crypto.py
=========

Cryptographic primitive layer for the SeBAC-IoD simulation.

This module wraps the low-level cryptography so that the protocol entities
(ControlServer, User, Drone) can use clean, paper-like calls such as::

    s, Ppub = gen_keypair()          # Ppub = s . Q   (server master key)
    kj, Kj  = gen_keypair()          # Kj   = kj . Q   (drone public point)
    h       = hash_to_int(a, b, c)   # H(a || b || c)

Design choices
--------------
* **Curve**: NIST P-256 (a.k.a. ``secp256r1`` / ``prime256v1``) via the
  ``ecdsa`` library.  The paper works over an elliptic curve E(Fp) with a
  base point Q of prime order n; P-256 gives us exactly that with a
  well-reviewed standard curve, satisfying the "standard curve" requirement.
* **Hash**: SHA-256 (``hashlib``), matching the paper's ``H()``.
* **Randomness**: ``secrets`` (CSPRNG) for all scalars and nonces.

We expose ECC points and scalars as Python objects but ALWAYS serialize them
to canonical bytes before hashing, so both endpoints agree on the byte image
of an EC point (the classic source of "my verifier doesn't match" bugs).
"""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass
from typing import Tuple

from ecdsa import NIST256p, SigningKey, VerifyingKey
from ecdsa.ellipticcurve import Point
from ecdsa.ecdsa import generator_256

import utils

logger = utils.get_logger(__name__)

# ---------------------------------------------------------------------------
# Curve domain parameters (public, shared by everyone)
# ---------------------------------------------------------------------------
# CURVE   : the NIST P-256 curve object.
# Q       : the base point / generator (the paper's "Q").
# ORDER n : the prime order of Q; all scalars live in Z_n^*  (1 .. n-1).
# PRIME p : the field characteristic (the "prime field" of the init phase).
# ---------------------------------------------------------------------------
CURVE = NIST256p
Q: Point = generator_256                      # base point Q
ORDER_N: int = CURVE.order                    # prime order n of Q
PRIME_P: int = CURVE.curve.p()                # field prime p (E over F_p)


# ---------------------------------------------------------------------------
# SHA-256 hashing  (the paper's H())
# ---------------------------------------------------------------------------
def sha256(*chunks: utils.Chunk) -> bytes:
    """
    Compute ``SHA-256(chunk_0 || chunk_1 || ...)`` and return the 32-byte digest.

    Each chunk is canonically byte-encoded via :func:`utils.to_bytes`, so ints,
    strings, bytes and serialized EC points can be mixed freely — mirroring the
    paper's ``H(a || b || c)`` notation.
    """
    return hashlib.sha256(utils.concat(*chunks)).digest()


def hash_to_int(*chunks: utils.Chunk) -> int:
    """
    Hash the inputs and reduce the digest modulo the curve order ``n``.

    Useful whenever a hash output must be used as an EC scalar (e.g. building
    Schnorr-like verifiers in the batch-authentication phase).  Reducing mod n
    keeps the value inside Z_n.
    """
    return utils.bytes_to_int(sha256(*chunks)) % ORDER_N


# ---------------------------------------------------------------------------
# Random number generation (nonces, salts, challenges)
# ---------------------------------------------------------------------------
def gen_nonce(nbytes: int = 16) -> bytes:
    """Return a cryptographically secure random nonce of ``nbytes`` bytes."""
    return secrets.token_bytes(nbytes)


def gen_scalar() -> int:
    """
    Return a uniformly random non-zero scalar in Z_n^*  (1 <= x <= n-1).

    Used for ECC private keys and per-session ephemeral secrets.
    """
    # secrets.randbelow(n-1) -> 0 .. n-2 ; +1 -> 1 .. n-1  (never 0).
    return 1 + secrets.randbelow(ORDER_N - 1)


def gen_salt(nbytes: int = 16) -> bytes:
    """Return a random salt (used by the hash-based biometric replacement)."""
    return secrets.token_bytes(nbytes)


# ---------------------------------------------------------------------------
# ECC key generation and point arithmetic
# ---------------------------------------------------------------------------
@dataclass
class ECKeyPair:
    """
    A simple container for an ECC key pair.

    Attributes
    ----------
    private:
        The secret scalar (e.g. the server's ``s`` or a drone's ``kj``).
    public:
        The public EC point  ``public = private . Q``  (e.g. ``Ppub`` / ``Kj``).
    """

    private: int
    public: Point

    def public_bytes(self) -> bytes:
        """Canonical byte image of the public point (for hashing/transport)."""
        return point_to_bytes(self.public)


def gen_keypair() -> Tuple[int, Point]:
    """
    Generate an ECC key pair ``(x, X)`` with ``X = x . Q``.

    Returns
    -------
    (int, Point)
        ``x`` is the private scalar, ``X`` is the corresponding public point.

    Examples
    --------
    Server master key in the paper::

        s, Ppub = gen_keypair()        # Ppub = s . Q

    Drone key in the paper::

        kj, Kj = gen_keypair()         # Kj = kj . Q
    """
    x = gen_scalar()
    X = scalar_mult(x, Q)
    return x, X


def scalar_mult(scalar: int, point: Point = Q) -> Point:
    """
    Elliptic-curve scalar multiplication ``scalar . point``.

    Defaults ``point`` to the base point ``Q`` so that ``scalar_mult(s)`` reads
    exactly like the paper's ``s . Q``.
    """
    # Reduce the scalar mod n; multiples of n act as the identity element.
    return (scalar % ORDER_N) * point


def point_add(a: Point, b: Point) -> Point:
    """Elliptic-curve point addition ``a + b`` (group operation)."""
    return a + b


# ---------------------------------------------------------------------------
# Point (de)serialization
# ---------------------------------------------------------------------------
# We encode an EC point as its uncompressed SEC1 octet string via the ecdsa
# VerifyingKey helper.  This gives a stable, standard byte representation that
# both endpoints reproduce identically before hashing.
# ---------------------------------------------------------------------------
def point_to_bytes(point: Point) -> bytes:
    """Serialize an EC point to canonical uncompressed SEC1 bytes."""
    vk = VerifyingKey.from_public_point(point, curve=CURVE)
    return vk.to_string()  # 64 bytes for P-256 (X||Y, fixed width)


def bytes_to_point(data: bytes) -> Point:
    """Deserialize canonical SEC1 bytes back into an EC point."""
    vk = VerifyingKey.from_string(data, curve=CURVE)
    return vk.pubkey.point


# ---------------------------------------------------------------------------
# Session key derivation
# ---------------------------------------------------------------------------
def derive_session_key(*chunks: utils.Chunk) -> bytes:
    """
    Derive a symmetric session key ``SK = H(shared_material...)``.

    In the batch-authentication phase, the User and each Drone independently
    hash the SAME agreed material (identities, nonces, a shared EC point, and
    timestamps) to obtain an identical 256-bit session key ``SKij`` without ever
    transmitting it.  This function centralizes that derivation so both sides
    use byte-identical inputs.
    """
    return sha256(*chunks)


# ---------------------------------------------------------------------------
# Convenience: raw ECDSA sign/verify (available if a phase needs a signature)
# ---------------------------------------------------------------------------
def new_signing_key() -> SigningKey:
    """Create a fresh ECDSA signing key on the project curve (P-256)."""
    return SigningKey.generate(curve=CURVE)


def domain_parameters() -> dict:
    """
    Return the public domain parameters as a dict (for printing in the
    Initialization phase: prime field p, order n, and base point Q).
    """
    return {
        "curve": "NIST P-256 (secp256r1)",
        "prime_p": PRIME_P,
        "order_n": ORDER_N,
        "Q_bytes": point_to_bytes(Q),
    }
