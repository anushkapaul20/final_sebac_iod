"""
protocol_messages.py
====================

Serialization / deserialization helpers for SeBAC-IoD socket communication.

All messages between entities (User <-> CS, CS <-> Drone, Drone <-> User) are
sent over TCP as JSON-encoded byte strings.  This module defines:

    * Message type constants
    * encode(msg_type, payload) -> bytes   (sender calls this)
    * decode(raw_bytes)         -> (type, payload)   (receiver calls this)

The payload is always a plain dict whose values are either plain Python
scalars, or bytes encoded as hex strings (since JSON can't carry raw bytes).

Helper functions ``b2h`` and ``h2b`` convert between bytes <-> hex strings so
the protocol code can work in bytes everywhere and this layer handles the
JSON boundary transparently.
"""

from __future__ import annotations

import json
import struct
from typing import Any, Dict, Tuple

# ---------------------------------------------------------------------------
# Message type constants
# ---------------------------------------------------------------------------
# Enrollment (secure channel, simulated)
MSG_DRONE_ENROLL_REQ   = "DRONE_ENROLL_REQ"    # Drone  -> CS
MSG_DRONE_ENROLL_ACK   = "DRONE_ENROLL_ACK"    # CS     -> Drone
MSG_USER_ENROLL_REQ    = "USER_ENROLL_REQ"     # User   -> CS
MSG_USER_ENROLL_ACK    = "USER_ENROLL_ACK"     # CS     -> User

# Authentication (insecure channel)
MSG_MU1                = "MU1"                 # User   -> CS
MSG_MU2                = "MU2"                 # CS     -> Drone (one per drone)
MSG_MU3                = "MU3"                 # Drone  -> User
MSG_MU2_BATCH          = "MU2_BATCH"           # CS     -> User (forwarded batch)

# Control / error
MSG_ERROR              = "ERROR"
MSG_OK                 = "OK"
MSG_SESSION_KEY_RESULT = "SESSION_KEY_RESULT"  # User   -> (self, printed)
MSG_SECURE_DATA        = "SECURE_DATA"         # Drone  -> User (AES encrypted)


# ---------------------------------------------------------------------------
# bytes <-> hex-string helpers
# ---------------------------------------------------------------------------
def b2h(data: bytes) -> str:
    """Encode bytes as a hex string for JSON transport."""
    return data.hex()


def h2b(hex_str: str) -> bytes:
    """Decode a hex string back to bytes."""
    return bytes.fromhex(hex_str)


def _encode_value(v: Any) -> Any:
    """Recursively encode bytes objects to hex strings for JSON."""
    if isinstance(v, bytes):
        return {"__bytes__": v.hex()}
    if isinstance(v, dict):
        return {k: _encode_value(val) for k, val in v.items()}
    if isinstance(v, list):
        return [_encode_value(item) for item in v]
    if isinstance(v, tuple):
        return {"__tuple__": [_encode_value(item) for item in v]}
    return v


def _decode_value(v: Any) -> Any:
    """Recursively decode hex strings back to bytes from JSON."""
    if isinstance(v, dict):
        if "__bytes__" in v:
            return bytes.fromhex(v["__bytes__"])
        if "__tuple__" in v:
            return tuple(_decode_value(item) for item in v["__tuple__"])
        return {k: _decode_value(val) for k, val in v.items()}
    if isinstance(v, list):
        return [_decode_value(item) for item in v]
    return v


# ---------------------------------------------------------------------------
# Frame format
# ---------------------------------------------------------------------------
# Each frame on the wire is:
#   [4 bytes big-endian length][JSON-encoded message bytes]
# This allows the receiver to know exactly how many bytes to read.
# ---------------------------------------------------------------------------
HEADER_SIZE = 4   # bytes for the length prefix


def encode(msg_type: str, payload: Dict[str, Any]) -> bytes:
    """
    Serialize a message to wire bytes.

    Parameters
    ----------
    msg_type:
        One of the ``MSG_*`` constants above.
    payload:
        A dict with the message fields.  bytes values are auto-converted to hex.

    Returns
    -------
    bytes
        Length-prefixed JSON frame ready to send over a socket.
    """
    envelope = {"type": msg_type, "payload": _encode_value(payload)}
    body = json.dumps(envelope).encode("utf-8")
    header = struct.pack(">I", len(body))
    return header + body


def decode(raw: bytes) -> Tuple[str, Dict[str, Any]]:
    """
    Deserialize a raw JSON body (without the 4-byte length header).

    Parameters
    ----------
    raw:
        The body bytes (after stripping the 4-byte length prefix).

    Returns
    -------
    (msg_type, payload)
    """
    envelope = json.loads(raw.decode("utf-8"))
    return envelope["type"], _decode_value(envelope["payload"])


# ---------------------------------------------------------------------------
# Socket I/O helpers
# ---------------------------------------------------------------------------
def send_msg(sock, msg_type: str, payload: Dict[str, Any],
             sender: str = "?", receiver: str = "?", phase: str = "unknown") -> int:
    """Send one framed message over ``sock``. Returns wire size in bytes."""
    data = encode(msg_type, payload)
    sock.sendall(data)
    # Log communication cost
    try:
        from comm_logger import get_comm_logger
        get_comm_logger().log_instant(msg_type, sender, receiver, phase, len(data))
    except Exception:
        pass
    return len(data)


def recv_msg(sock) -> Tuple[str, Dict[str, Any]]:
    """
    Receive one framed message from ``sock``.

    Reads the 4-byte length header first, then the exact body length.
    Raises ``ConnectionError`` if the socket closes mid-read.
    """
    # Read the 4-byte length header.
    header = _recv_exactly(sock, HEADER_SIZE)
    if not header:
        raise ConnectionError("Socket closed while reading message header")
    body_len = struct.unpack(">I", header)[0]

    # Read the exact body.
    body = _recv_exactly(sock, body_len)
    if not body:
        raise ConnectionError("Socket closed while reading message body")
    return decode(body)


def _recv_exactly(sock, n: int) -> bytes:
    """Read exactly ``n`` bytes from ``sock``, blocking until done."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return b""
        buf.extend(chunk)
    return bytes(buf)
