"""
drone_app.py
============

Drone (Dj) — TCP socket client/server for SeBAC-IoD.

Run AFTER server_app.py, one process per drone (or multiple in threads).

Usage
-----
    python drone_app.py --id Drone-00 --port 9100
    python drone_app.py --id Drone-01 --port 9101
    ...

Each drone:
1.  Enrolls with the CS over ENROLL_PORT (sends DRONE_ENROLL_REQ, receives
    DRONE_ENROLL_ACK, stores on-board credentials).
2.  Starts a small TCP listener on its own AUTH_PORT.  The CS connects to this
    port to deliver mu2.
3.  On receiving mu2, validates it (authenticates CS), produces a Schnorr
    signature (Ij), and connects to the USER_AUTH_PORT to deliver mu3.

The user's address (host, port) is passed as a command-line argument so the
drone knows where to send mu3.
"""

from __future__ import annotations

import argparse
import socket
import threading
import time

import drone as dmod
import puf
import utils
import aes_utils
from protocol_messages import (
    MSG_DRONE_ENROLL_REQ, MSG_DRONE_ENROLL_ACK,
    MSG_MU2, MSG_MU3, MSG_SECURE_DATA,
    MSG_ERROR,
    recv_msg, send_msg,
)

logger = utils.get_logger("drone_app")

# Defaults — override via CLI args for real network deployment
CS_HOST         = "127.0.0.1"   # --cs-host  : IP of the laptop running server_app.py
CS_ENROLL_PORT  = 9000
USER_AUTH_HOST  = "127.0.0.1"   # --user-host : IP of the laptop running user_app.py
USER_AUTH_PORT  = 9200


class DroneApp:
    """Networked drone entity."""

    def __init__(self, id_d: str, auth_port: int,
                 cs_host: str = "127.0.0.1",
                 user_host: str = "127.0.0.1") -> None:
        self.drone     = dmod.Drone(id_d)
        self.auth_port = auth_port
        self.cs_host   = cs_host
        self.user_host = user_host
        self._enrolled = threading.Event()

    # ------------------------------------------------------------------ #
    # Step 1 — Enrollment                                                 #
    # ------------------------------------------------------------------ #
    def enroll(self) -> None:
        """
        Connect to the CS enrollment server, send DRONE_ENROLL_REQ,
        receive DRONE_ENROLL_ACK and store the returned credentials.
        """
        req = self.drone.begin_enrollment()

        logger.info("Drone %s connecting to CS enrollment server at %s:%d ...",
                    self.drone.id_d, self.cs_host, CS_ENROLL_PORT)
        with socket.create_connection((self.cs_host, CS_ENROLL_PORT), timeout=15) as sock:
            send_msg(sock, MSG_DRONE_ENROLL_REQ, {
                "id_d":          req.id_d,
                "v":             req.v,
                "challenge":     req.challenge,
                "puf_response":  req.puf_response,
                "device_secret": self.drone.device_secret,
                "auth_host":     self.user_host,
                "auth_port":     self.auth_port,
            }, sender=self.drone.id_d, receiver="CS", phase="enrollment")

            msg_type, payload = recv_msg(sock)

        if msg_type == MSG_ERROR:
            raise RuntimeError(f"Enrollment rejected: {payload['reason']}")
        if msg_type != MSG_DRONE_ENROLL_ACK:
            raise RuntimeError(f"Unexpected enrollment response: {msg_type}")

        logger.info("Drone %s received DRONE_ENROLL_ACK from Control Server.", self.drone.id_d)
        logger.info("  DIDj = %s", utils.short_hx(payload["DIDj"]))
        logger.info("  Ksd  = %s", utils.short_hx(payload["Ksd"]))
        logger.info("  Ej   = %s", utils.short_hx(payload["Ej"]))
        logger.info("  Nj   = %s", utils.short_hx(payload["Nj"]))

        self.drone.store_credentials({
            "DIDj": payload["DIDj"],
            "Ksd":  payload["Ksd"],
            "Kj":   payload["Kj"],
            "kj":   payload["kj"],
            "Ej":   payload["Ej"],
            "Wj":   payload["Wj"],
            "Nj":   payload["Nj"],
        })
        logger.info("Drone %s stored enrollment credentials successfully.", self.drone.id_d)
        self._enrolled.set()

    # ------------------------------------------------------------------ #
    # Step 2 — Listen for mu2 from CS                                     #
    # ------------------------------------------------------------------ #
    def run_auth_listener(self) -> None:
        """
        Small TCP server that waits for the CS to push a mu2 message.
        On receipt, processes it and sends mu3 to the user.
        """
        # Wait until enrollment is done before opening the auth listener.
        self._enrolled.wait()

        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("0.0.0.0", self.auth_port))   # listen on all interfaces
        srv.listen(5)
        logger.info("Drone %s listening for mu2 on port %d", self.drone.id_d, self.auth_port)

        while True:
            conn, addr = srv.accept()
            t = threading.Thread(
                target=self._handle_mu2, args=(conn, addr), daemon=True
            )
            t.start()

    def _handle_mu2(self, conn: socket.socket, addr) -> None:
        """Receive mu2, validate, build mu3, send to user."""
        try:
            msg_type, payload = recv_msg(conn)
            if msg_type != MSG_MU2:
                logger.warning("Drone expected MU2, got %s", msg_type)
                return

            # Build a mu2-like object from the deserialized payload.
            class _Mu2:
                pass

            mu2 = _Mu2()
            mu2.yj = payload["yj"]
            mu2.m5 = payload["m5"]
            mu2.g2 = payload["g2"]
            mu2.g3 = payload["g3"]
            mu2.t2 = payload["t2"]

            logger.info("Drone %s received MU2 from Control Server.", self.drone.id_d)

            mu3 = self.drone.handle_mu2(mu2)

            logger.info("MU2 verified successfully by %s.", self.drone.id_d)

            # Send mu3 to the user.
            self._send_mu3_to_user(mu3)

        except ConnectionError:
            pass  # port-probe, ignore
        except Exception as exc:
            logger.error("Drone %s mu2 handling error: %s", self.drone.id_d, exc)
        finally:
            conn.close()

    def _send_mu3_to_user(self, mu3: dmod.Mu3) -> None:
        """Connect to the user's mu3 listener and deliver mu3, then send encrypted message."""
        Rj_bytes, sj = mu3.Ij
        logger.info("Drone %s sending mu3 to user at %s:%d ...",
                    self.drone.id_d, self.user_host, USER_AUTH_PORT)
        try:
            with socket.create_connection((self.user_host, USER_AUTH_PORT), timeout=15) as sock:
                send_msg(sock, MSG_MU3, {
                    "m5":     mu3.m5,
                    "did_j":  mu3.did_j,
                    "Rj":     Rj_bytes,
                    "sj":     sj,
                    "kj_pub": mu3.kj_pub,
                }, sender=self.drone.id_d, receiver="User", phase="authentication")
            logger.info("%s -> User : MU3 sent", self.drone.id_d)

            # Small delay — give user time to verify batch and store SKij
            import time as _t; _t.sleep(2)

            # ── Secure data exchange ─────────────────────────────────────
            sk = self.drone.derive_session_key()
            message = f"hello from {self.drone.id_d}"
            encrypted = aes_utils.encrypt(message, sk)
            mode = "AES-256-CBC" if aes_utils.is_using_real_aes() else "XOR-fallback"
            logger.info("Drone %s encrypting '%s' with %s ...", self.drone.id_d, message, mode)
            logger.info("  Ciphertext : %s...", encrypted.hex()[:24])
            with socket.create_connection((self.user_host, USER_AUTH_PORT + 1), timeout=15) as sock:
                send_msg(sock, MSG_SECURE_DATA, {
                    "did_j":     mu3.did_j,
                    "encrypted": encrypted,
                    "mode":      mode,
                }, sender=self.drone.id_d, receiver="User", phase="data_exchange")
            logger.info("Drone %s sent encrypted message to user.", self.drone.id_d)

        except Exception as exc:
            logger.error("Drone %s failed to send mu3/data: %s", self.drone.id_d, exc)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="SeBAC-IoD Drone node")
    parser.add_argument("--id",        default="Drone-00",   help="Drone identity IDd")
    parser.add_argument("--port",      type=int, default=9100, help="Drone auth listener port")
    parser.add_argument("--cs-host",   default="127.0.0.1",  help="IP of the Control Server laptop")
    parser.add_argument("--user-host", default="127.0.0.1",  help="IP of the User laptop (for mu3)")
    args = parser.parse_args()

    app = DroneApp(args.id, args.port,
                   cs_host=args.cs_host,
                   user_host=args.user_host)

    # Enroll first, then start the auth listener.
    app.enroll()

    # Run the auth listener in a background thread.
    t = threading.Thread(target=app.run_auth_listener, daemon=True)
    t.start()

    logger.info("Drone %s ready. Press Ctrl+C to stop.", args.id)
    try:
        t.join()
    except KeyboardInterrupt:
        logger.info("Drone %s shutting down.", args.id)


if __name__ == "__main__":
    main()
