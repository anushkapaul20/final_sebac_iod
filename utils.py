"""
utils.py
========

Foundation / utility layer for the SeBAC-IoD simulation.

This module deliberately depends on NOTHING else in the project so that it can
be safely imported by every other module without creating circular imports.

It provides:
    * A project-wide logger that writes simultaneously to the console and to
      ``logs/protocol_log.txt`` (used to satisfy the "LOGGING" requirement).
    * Timestamp helpers + a freshness/replay window check (``T1``, ``T2`` in the
      paper's message flow).
    * Byte <-> integer conversion helpers (needed to feed hashes/ECC scalars).
    * XOR over byte strings (the protocol masks secrets with XOR, e.g. ``Ej``).
    * Concatenation / serialization helpers so that heterogeneous values
      (ints, bytes, EC points encoded as bytes, strings) can be hashed uniformly.

All functions are fully type-hinted and documented.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Iterable, Union

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
# A "Hashable chunk" is anything we are willing to feed into a hash function
# after canonical byte-encoding.  Keeping this explicit makes the protocol code
# read very close to the paper's notation (concatenation of mixed values).
Chunk = Union[bytes, bytearray, int, str]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
# We expose ONE configured logger.  Every module calls ``get_logger(__name__)``
# and they all share the same file/console handlers (handlers are attached only
# once to the root "sebac" logger to avoid duplicate lines).
# ---------------------------------------------------------------------------
_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
_LOG_FILE = os.path.join(_LOG_DIR, "protocol_log.txt")
_ROOT_LOGGER_NAME = "sebac"
_CONFIGURED = False


def _configure_root_logger() -> None:
    """Attach console + file handlers to the root 'sebac' logger exactly once."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    os.makedirs(_LOG_DIR, exist_ok=True)

    root = logging.getLogger(_ROOT_LOGGER_NAME)
    root.setLevel(logging.DEBUG)
    root.propagate = False  # don't bubble up to Python's default root logger

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)-16s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler: human-friendly, INFO and above.
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)

    # File handler: full DEBUG trace, appended to logs/protocol_log.txt.
    file_handler = logging.FileHandler(_LOG_FILE, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)

    root.addHandler(console)
    root.addHandler(file_handler)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """
    Return a child logger under the shared 'sebac' root logger.

    Parameters
    ----------
    name:
        Usually ``__name__`` of the calling module.

    Returns
    -------
    logging.Logger
        A logger that writes to both the console and ``logs/protocol_log.txt``.
    """
    _configure_root_logger()
    # Use a short child name so log lines stay readable (e.g. "sebac.drone").
    short = name.split(".")[-1]
    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{short}")


def banner(logger: logging.Logger, title: str) -> None:
    """Log a visually distinct section banner (used between protocol phases)."""
    line = "=" * 70
    logger.info(line)
    logger.info(title.center(70))
    logger.info(line)


# ---------------------------------------------------------------------------
# Timestamps & freshness (replay protection)
# ---------------------------------------------------------------------------
# The paper attaches timestamps T1/T2 to messages and the receiver rejects a
# message whose timestamp is outside an acceptable transmission window
# ``delta_t``.  This defends against replay attacks.
# ---------------------------------------------------------------------------
def now_ts() -> int:
    """Return the current Unix timestamp as an integer (seconds)."""
    return int(time.time())


def is_fresh(ts: int, delta_t: int = 120) -> bool:
    """
    Validate a message timestamp against the local clock.

    Parameters
    ----------
    ts:
        The timestamp carried by the incoming message (e.g. ``T1``).
    delta_t:
        Maximum tolerated transmission delay in seconds.  In a real deployment
        this is small (a few seconds); we use a generous default so that a slow
        simulation step never spuriously fails.

    Returns
    -------
    bool
        ``True`` if ``|now - ts| <= delta_t`` (message considered fresh).
    """
    return abs(now_ts() - ts) <= delta_t


# ---------------------------------------------------------------------------
# Byte / integer conversion helpers
# ---------------------------------------------------------------------------
def int_to_bytes(value: int, length: int | None = None) -> bytes:
    """
    Convert a non-negative integer to big-endian bytes.

    Parameters
    ----------
    value:
        Non-negative integer (ECC scalars, hash digests as ints, nonces).
    length:
        Optional fixed width.  If omitted, the minimal number of bytes is used
        (at least 1 byte, so 0 -> b"\\x00").

    Raises
    ------
    ValueError
        If ``value`` is negative.
    """
    if value < 0:
        raise ValueError("int_to_bytes only supports non-negative integers")
    if length is None:
        length = max(1, (value.bit_length() + 7) // 8)
    return value.to_bytes(length, byteorder="big")


def bytes_to_int(data: bytes) -> int:
    """Convert big-endian bytes to a non-negative integer."""
    return int.from_bytes(data, byteorder="big")


def to_bytes(chunk: Chunk) -> bytes:
    """
    Canonically encode a single heterogeneous value to bytes.

    This is the single source of truth for how each value type is turned into
    bytes before hashing/XOR, guaranteeing both sides of the protocol agree on
    the byte representation.

    * bytes/bytearray -> as-is
    * int             -> minimal big-endian bytes
    * str             -> UTF-8 encoding
    """
    if isinstance(chunk, (bytes, bytearray)):
        return bytes(chunk)
    if isinstance(chunk, int):
        return int_to_bytes(chunk)
    if isinstance(chunk, str):
        return chunk.encode("utf-8")
    raise TypeError(f"Unsupported chunk type for encoding: {type(chunk)!r}")


def concat(*chunks: Chunk) -> bytes:
    """
    Concatenate heterogeneous values into a single byte string.

    Mirrors the paper's ``a || b || c`` notation.  Example::

        concat(IDu, r, T1)  ->  bytes(IDu) + bytes(r) + bytes(T1)
    """
    return b"".join(to_bytes(c) for c in chunks)


# ---------------------------------------------------------------------------
# XOR helpers (used for masking, e.g. Ej = H(...) XOR secret)
# ---------------------------------------------------------------------------
def xor_bytes(a: bytes, b: bytes) -> bytes:
    """
    XOR two byte strings of equal length.

    Raises
    ------
    ValueError
        If the two inputs differ in length (a programming error that would
        otherwise silently truncate and break verification).
    """
    if len(a) != len(b):
        raise ValueError(
            f"xor_bytes length mismatch: {len(a)} != {len(b)} "
            "(pad/truncate the operands to equal width before XOR)"
        )
    return bytes(x ^ y for x, y in zip(a, b))


def xor_many(*items: bytes) -> bytes:
    """XOR several equal-length byte strings together (left to right)."""
    if not items:
        raise ValueError("xor_many requires at least one operand")
    result = items[0]
    for nxt in items[1:]:
        result = xor_bytes(result, nxt)
    return result


def fit(data: bytes, length: int = 32) -> bytes:
    """
    Force ``data`` to exactly ``length`` bytes so it can be XOR-ed safely.

    Shorter inputs are left-padded with zero bytes; longer inputs are reduced
    by hashing-then-truncation is avoided here intentionally — we simply take
    the rightmost ``length`` bytes, which is sufficient for the simulation's
    fixed 32-byte (SHA-256) masking convention.
    """
    if len(data) == length:
        return data
    if len(data) < length:
        return data.rjust(length, b"\x00")
    return data[-length:]


# ---------------------------------------------------------------------------
# Pretty printing helpers (used by main.py to "print all generated values")
# ---------------------------------------------------------------------------
def hx(value: Union[bytes, int]) -> str:
    """Return a short hex string for display/logging of a secret or digest."""
    if isinstance(value, int):
        value = int_to_bytes(value)
    return value.hex()


def short_hx(value: Union[bytes, int], head: int = 12) -> str:
    """Return a truncated hex preview like ``a1b2c3...`` for compact logs."""
    full = hx(value)
    return full if len(full) <= head else f"{full[:head]}..."


def kv(name: str, value: Chunk) -> str:
    """Format a ``name = value`` line with hex-encoded bytes/ints for logging."""
    if isinstance(value, (bytes, bytearray, int)):
        return f"{name} = {hx(bytes(value) if isinstance(value, bytearray) else value)}"
    return f"{name} = {value}"
