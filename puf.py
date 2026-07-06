"""
puf.py
======

Simulated Physical Unclonable Function (PUF) layer for SeBAC-IoD.

Why a *simulated* PUF?
----------------------
A real PUF is a piece of silicon whose tiny manufacturing variations turn an
input **challenge** into an output **response** that is:

    * deterministic  -> same challenge always yields the same response on that
                        specific chip,
    * device-unique  -> a different chip yields a different response,
    * unclonable     -> you cannot predict/reproduce the response without the
                        physical chip.

We have no silicon here, so (per the project brief) we model those properties
in software with a keyed hash::

        PUF(challenge) = SHA-256( challenge || device_secret )

The per-device ``device_secret`` is the software stand-in for the chip's unique
physical randomness:

    * deterministic  -> hashing is deterministic,
    * device-unique  -> each Drone holds a different random ``device_secret``,
    * unclonable     -> without the secret you cannot compute the response,
                        and the secret never leaves the device.

This module exposes the four functions required by the brief:

    generate_challenge()      -> create a fresh random challenge C_j
    simulate_puf()            -> the raw PUF mapping  C |-> R
    response_generation()     -> a device produces (challenge, response) pairs
    response_verification()   -> the server re-derives and checks a response

Note on the trust model
-----------------------
In a real PUF deployment the *verifier* (Control Server) stores Challenge-
Response Pairs (CRPs) captured during a secure enrollment, and never needs the
secret.  In this simulation the Control Server is allowed to know each drone's
``device_secret`` (it is the registration authority) so it can re-derive
responses on demand; we ALSO keep an explicit CRP store to demonstrate the
real-world pattern.  Both checks are provided.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import crypto
import utils

logger = utils.get_logger(__name__)

# Width (in bytes) of a PUF challenge.  16 bytes = 128-bit challenge space.
CHALLENGE_BYTES = 16
# Width (in bytes) of the per-device secret modelling physical randomness.
DEVICE_SECRET_BYTES = 32


# ---------------------------------------------------------------------------
# Core PUF primitives
# ---------------------------------------------------------------------------
def generate_challenge(nbytes: int = CHALLENGE_BYTES) -> bytes:
    """
    Generate a fresh, random PUF challenge ``C_j``.

    Parameters
    ----------
    nbytes:
        Length of the challenge in bytes (default 16 -> 128-bit).

    Returns
    -------
    bytes
        A cryptographically random challenge produced via ``secrets``.
    """
    return crypto.gen_nonce(nbytes)


def generate_device_secret(nbytes: int = DEVICE_SECRET_BYTES) -> bytes:
    """
    Generate a per-device secret that models the chip's physical randomness.

    Each Drone calls this ONCE at manufacture/enrollment and keeps the value
    private for its entire lifetime.  Two drones will (overwhelmingly likely)
    get different secrets, giving device-unique responses.
    """
    return crypto.gen_nonce(nbytes)


def simulate_puf(challenge: bytes, device_secret: bytes) -> bytes:
    """
    The raw PUF mapping:  ``response = SHA-256(challenge || device_secret)``.

    This is the heart of the simulation.  It is deterministic for a fixed
    (challenge, device_secret) pair, unique per device, and unpredictable
    without the secret.

    Parameters
    ----------
    challenge:
        The input challenge ``C_j``.
    device_secret:
        The drone's private physical-randomness stand-in.

    Returns
    -------
    bytes
        The 32-byte PUF response ``R``.
    """
    return crypto.sha256(challenge, device_secret)


# ---------------------------------------------------------------------------
# Device-side helper object
# ---------------------------------------------------------------------------
@dataclass
class PUFDevice:
    """
    A drone's on-board PUF instance.

    The ``device_secret`` lives ONLY inside this object — exactly like a real
    PUF, the protocol code never reads it directly; it only ever asks the
    device to evaluate a challenge.

    Attributes
    ----------
    device_id:
        Human-readable identifier (mirrors the drone's ``IDd``); used for logs.
    device_secret:
        The private per-device randomness (auto-generated if not supplied).
    """

    device_id: str
    device_secret: bytes = field(default_factory=generate_device_secret, repr=False)

    def evaluate(self, challenge: bytes) -> bytes:
        """
        Evaluate the on-board PUF on a challenge and return its response.

        This is what a real drone does physically: feed in ``C_j``, read out
        ``R = PUF(C_j)``.
        """
        response = simulate_puf(challenge, self.device_secret)
        logger.debug(
            "PUFDevice[%s].evaluate: C=%s -> R=%s",
            self.device_id,
            utils.short_hx(challenge),
            utils.short_hx(response),
        )
        return response


# ---------------------------------------------------------------------------
# Response generation (device side, enrollment)
# ---------------------------------------------------------------------------
def response_generation(device: PUFDevice, challenge: bytes) -> bytes:
    """
    Produce a PUF response for a given challenge using the device's PUF.

    Corresponds to the drone-side computation ``Rest = PUF(C_j)`` in the
    enrollment phase.

    Parameters
    ----------
    device:
        The drone's :class:`PUFDevice`.
    challenge:
        The challenge ``C_j`` to evaluate.

    Returns
    -------
    bytes
        The PUF response ``Rest``.
    """
    return device.evaluate(challenge)


# ---------------------------------------------------------------------------
# Challenge-Response Pair (CRP) store + verification (verifier side)
# ---------------------------------------------------------------------------
@dataclass
class CRPStore:
    """
    A verifier-side store of Challenge-Response Pairs captured at enrollment.

    In a real PUF system the Control Server keeps these so it can later
    challenge a device and compare the fresh response against the stored one,
    WITHOUT needing the device secret.  We model that here.
    """

    # Maps device_id -> { challenge_hex : response }
    _crps: Dict[str, Dict[str, bytes]] = field(default_factory=dict)

    def enroll(self, device_id: str, challenge: bytes, response: bytes) -> None:
        """Record a (challenge, response) pair captured during enrollment."""
        self._crps.setdefault(device_id, {})[challenge.hex()] = response
        logger.debug(
            "CRPStore.enroll: device=%s C=%s R=%s",
            device_id,
            utils.short_hx(challenge),
            utils.short_hx(response),
        )

    def lookup(self, device_id: str, challenge: bytes) -> bytes | None:
        """Return the stored response for a (device, challenge), or None."""
        return self._crps.get(device_id, {}).get(challenge.hex())


def response_verification(
    presented_response: bytes,
    *,
    device_secret: bytes | None = None,
    challenge: bytes | None = None,
    expected_response: bytes | None = None,
) -> bool:
    """
    Verify that a presented PUF response is correct.

    Two verification modes are supported (use exactly one):

    1. **Secret-based re-derivation** (registration-authority mode):
       provide ``device_secret`` and ``challenge``; we recompute
       ``SHA-256(challenge || device_secret)`` and compare.

    2. **CRP-based comparison** (classic verifier mode):
       provide ``expected_response`` (previously stored at enrollment) and
       compare directly.

    Parameters
    ----------
    presented_response:
        The response value being checked.
    device_secret, challenge:
        Inputs for mode (1).
    expected_response:
        Stored response for mode (2).

    Returns
    -------
    bool
        ``True`` iff the presented response matches.

    Raises
    ------
    ValueError
        If the provided arguments don't select exactly one valid mode.
    """
    if expected_response is not None:
        # Mode 2: constant-time compare against a stored CRP.
        return _ct_equal(presented_response, expected_response)

    if device_secret is not None and challenge is not None:
        # Mode 1: re-derive from the secret and compare.
        recomputed = simulate_puf(challenge, device_secret)
        return _ct_equal(presented_response, recomputed)

    raise ValueError(
        "response_verification: provide either expected_response, "
        "or both device_secret and challenge"
    )


def _ct_equal(a: bytes, b: bytes) -> bool:
    """
    Constant-time byte comparison (avoids timing side channels on equality).

    Uses Python's stdlib ``hmac.compare_digest`` under the hood.
    """
    import hmac

    return hmac.compare_digest(a, b)
