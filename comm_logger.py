"""
comm_logger.py
==============

Communication cost and timing recorder for SeBAC-IoD.

Sir asked to record communicational cost/time for every message.
This module logs every message with:
    - Phase          (enrollment / authentication / data_exchange)
    - Message type   (DRONE_ENROLL_REQ, MU1, MU2, MU3, SECURE_DATA ...)
    - Sender         (who sent it)
    - Receiver       (who received it)
    - Size in bytes  (actual wire size)
    - Size in bits   (for comparison with paper's Table V/VI)
    - Timestamp sent / received
    - Latency in ms  (recv_time - send_time)

All records saved to: logs/comm_log.json

How to read the log
-------------------
After running the simulation, open logs/comm_log.json to see every
message exchanged, its size, and how long it took.

Usage
-----
Import and use in server_app.py / drone_app.py / user_app.py:

    from comm_logger import CommLogger
    clog = CommLogger()

    # When sending:
    clog.log_send("MU1", sender="User", receiver="CS",
                  phase="authentication", size_bytes=len(raw_data))

    # When receiving:
    clog.log_recv("MU1", sender="User", receiver="CS",
                  phase="authentication", size_bytes=len(raw_data))

    # Save at end:
    clog.save()
    clog.print_summary()
"""

from __future__ import annotations

import json
import os
import time
from typing import List, Dict

import utils

logger = utils.get_logger("comm_logger")

LOG_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
LOG_FILE = os.path.join(LOG_DIR, "comm_log.json")


class CommLogger:
    """Records all message sizes and timings."""

    def __init__(self) -> None:
        self._records: List[Dict] = []
        self._pending: Dict[str, float] = {}  # msg_key -> send_time

    def log_send(self, msg_type: str, sender: str, receiver: str,
                 phase: str, size_bytes: int) -> str:
        """
        Record that a message was sent. Returns a key to match with log_recv.

        Parameters
        ----------
        msg_type    : e.g. "MU1", "DRONE_ENROLL_REQ"
        sender      : e.g. "User-01", "Drone-00", "CS"
        receiver    : e.g. "CS", "User-01"
        phase       : "enrollment" / "authentication" / "data_exchange"
        size_bytes  : number of bytes in the wire message
        """
        ts   = time.time()
        key  = f"{msg_type}_{sender}_{receiver}_{ts}"
        self._pending[key] = ts

        record = {
            "phase":       phase,
            "message":     msg_type,
            "sender":      sender,
            "receiver":    receiver,
            "size_bytes":  size_bytes,
            "size_bits":   size_bytes * 8,
            "timestamp_sent": ts,
            "timestamp_recv": None,
            "latency_ms":  None,
            "_key":        key,
        }
        self._records.append(record)
        logger.debug("[SEND] %s | %s->%s | %d bytes (%d bits)",
                     msg_type, sender, receiver, size_bytes, size_bytes * 8)
        return key

    def log_recv(self, key: str, size_bytes: int = None) -> None:
        """
        Record receipt of a message previously logged with log_send.

        Parameters
        ----------
        key         : the string returned by log_send
        size_bytes  : optional override of size if different at receiver
        """
        recv_time = time.time()
        for r in reversed(self._records):
            if r.get("_key") == key:
                send_time      = r["timestamp_sent"]
                latency_ms     = (recv_time - send_time) * 1000
                r["timestamp_recv"] = recv_time
                r["latency_ms"]     = round(latency_ms, 3)
                if size_bytes is not None:
                    r["size_bytes"] = size_bytes
                    r["size_bits"]  = size_bytes * 8
                logger.debug("[RECV] %s | latency=%.3f ms", r["message"], latency_ms)
                break

    def log_instant(self, msg_type: str, sender: str, receiver: str,
                    phase: str, size_bytes: int) -> None:
        """
        Log a send+recv in one call (when latency is not separately measured).
        """
        ts = time.time()
        record = {
            "phase":          phase,
            "message":        msg_type,
            "sender":         sender,
            "receiver":       receiver,
            "size_bytes":     size_bytes,
            "size_bits":      size_bytes * 8,
            "timestamp_sent": ts,
            "timestamp_recv": ts,
            "latency_ms":     0.0,
        }
        self._records.append(record)

    def save(self) -> None:
        """Save all records to logs/comm_log.json."""
        os.makedirs(LOG_DIR, exist_ok=True)
        # Remove internal _key field before saving
        clean = [{k: v for k, v in r.items() if k != "_key"}
                 for r in self._records]
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(clean, f, indent=2)
        logger.info("Communication log saved: %s (%d records)", LOG_FILE, len(clean))

    def print_summary(self) -> None:
        """Print a summary table of all messages."""
        if not self._records:
            logger.info("No communication records.")
            return

        print("\n" + "="*80)
        print("  SeBAC-IoD COMMUNICATION COST SUMMARY")
        print("="*80)
        print(f"  {'Phase':<16} {'Message':<22} {'From':<12} {'To':<12} "
              f"{'Bytes':>7} {'Bits':>7} {'Latency(ms)':>12}")
        print("-"*80)

        total_bytes = 0
        total_bits  = 0
        by_phase: Dict[str, int] = {}

        for r in self._records:
            if r.get("_key"):
                continue   # skip un-received sends
            lat = f"{r['latency_ms']:.3f}" if r['latency_ms'] is not None else "N/A"
            print(f"  {r['phase']:<16} {r['message']:<22} {r['sender']:<12} "
                  f"{r['receiver']:<12} {r['size_bytes']:>7} "
                  f"{r['size_bits']:>7} {lat:>12}")
            total_bytes += r["size_bytes"]
            total_bits  += r["size_bits"]
            by_phase[r["phase"]] = by_phase.get(r["phase"], 0) + r["size_bytes"]

        print("-"*80)
        print(f"  {'TOTAL':<52} {total_bytes:>7} {total_bits:>7}")
        print("="*80)
        print("\n  Cost by phase:")
        for phase, b in by_phase.items():
            print(f"    {phase:<20} {b} bytes  ({b*8} bits)")
        print()


# ---------------------------------------------------------------------------
# Global singleton — import and use anywhere
# ---------------------------------------------------------------------------
_global_logger: CommLogger = None


def get_comm_logger() -> CommLogger:
    """Return the global CommLogger instance (creates one if not exists)."""
    global _global_logger
    if _global_logger is None:
        _global_logger = CommLogger()
    return _global_logger
