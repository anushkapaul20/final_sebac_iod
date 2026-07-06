"""
user_app.py
===========

User (Ui) — TCP socket client for SeBAC-IoD.

Run LAST, after server_app.py and all drone_app.py instances.

    python user_app.py --id User-01 --password S3cure-Pass! --drones 3

What this does
--------------
1.  Enrolls with the CS (sends USER_ENROLL_REQ, receives USER_ENROLL_ACK).
2.  Opens a TCP listener on USER_AUTH_PORT to receive mu3 messages from drones.
3.  Builds mu1 and sends it to the CS authentication server.
4.  The CS verifies mu1 and forwards mu2 to each registered drone.
    The CS also replies to the user with the list of drones in the batch.
5.  Each drone sends its mu3 directly to the user's listener.
6.  The user collects all mu3 messages, runs batch verification, and derives
    one session key SKij per valid drone.

The number of drones to wait for is passed via --drones (must match the number
of drone_app.py processes that enrolled).
"""

from __future__ import annotations

import argparse
import socket
import threading
import time
from typing import List

import drone as dmod   # for Mu3 dataclass
import user as umod
import utils
import aes_utils
from protocol_messages import (
    MSG_USER_ENROLL_REQ, MSG_USER_ENROLL_ACK,
    MSG_MU1, MSG_MU2,
    MSG_MU3, MSG_SECURE_DATA,
    MSG_ERROR,
    recv_msg, send_msg,
)

logger = utils.get_logger("user_app")

# Defaults — override via CLI args for real network deployment
CS_HOST        = "127.0.0.1"   # --cs-host : IP of the laptop running server_app.py
CS_ENROLL_PORT = 9000
CS_AUTH_PORT   = 9001
USER_AUTH_HOST = "0.0.0.0"     # listen on all interfaces so RPi can reach us
USER_AUTH_PORT = 9200


class UserApp:
    """Networked user entity."""

    def __init__(self, id_u: str, password: str, num_drones: int,
                 cs_host: str = "127.0.0.1") -> None:
        self.user_obj   = umod.User(id_u, password)
        self.num_drones = num_drones
        self.cs_host    = cs_host
        self._mu3_batch: List[dmod.Mu3] = []
        self._mu3_lock  = threading.Lock()
        self._mu3_ready = threading.Event()
        self._sessions: dict = {}   # DIDj_hex -> SKij, filled after verify

    # ------------------------------------------------------------------ #
    # Step 1 — Enrollment                                                 #
    # ------------------------------------------------------------------ #
    def enroll(self) -> None:
        """Enroll with the CS over the enrollment channel."""
        id_u, uid = self.user_obj.begin_enrollment()

        logger.info("User %s connecting to CS at %s:%d for enrollment ...",
                    id_u, self.cs_host, CS_ENROLL_PORT)
        with socket.create_connection((self.cs_host, CS_ENROLL_PORT), timeout=15) as sock:
            send_msg(sock, MSG_USER_ENROLL_REQ, {
                "id_u": id_u,
                "uid":  uid,
            }, sender=id_u, receiver="CS", phase="enrollment")
            msg_type, payload = recv_msg(sock)

        if msg_type == MSG_ERROR:
            raise RuntimeError(f"Enrollment rejected: {payload['reason']}")
        if msg_type != MSG_USER_ENROLL_ACK:
            raise RuntimeError(f"Unexpected enrollment response: {msg_type}")

        logger.info("User received USER_ENROLL_ACK from Control Server.")
        logger.info("  PIDu = %s", utils.short_hx(payload["PIDu"]))
        logger.info("  FIDs = %s", utils.short_hx(payload["FIDs"]))
        logger.info("  Bu   = %s", utils.short_hx(payload["Bu"]))

        self.user_obj.complete_enrollment({
            "PIDu": payload["PIDu"],
            "FIDs": payload["FIDs"],
            "Bu":   payload["Bu"],
        })
        logger.info("User enrollment credentials stored successfully.")

    # ------------------------------------------------------------------ #
    # Step 2 — Listen for mu3 from drones                                 #
    # ------------------------------------------------------------------ #
    def start_mu3_listener(self) -> None:
        """Start background threads for mu3 and secure data messages."""
        t1 = threading.Thread(target=self._run_mu3_server, daemon=True)
        t2 = threading.Thread(target=self._run_secure_data_server, daemon=True)
        t1.start()
        t2.start()

    def _run_secure_data_server(self) -> None:
        """Listen on USER_AUTH_PORT+1 for AES-encrypted messages from drones."""
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((USER_AUTH_HOST, USER_AUTH_PORT + 1))
        srv.listen(20)
        logger.info("User secure-data listener on %s:%d", USER_AUTH_HOST, USER_AUTH_PORT + 1)
        while True:
            conn, addr = srv.accept()
            t = threading.Thread(target=self._handle_secure_data, args=(conn,), daemon=True)
            t.start()

    def _handle_secure_data(self, conn: socket.socket) -> None:
        """Receive an encrypted message from a drone and decrypt it."""
        try:
            msg_type, payload = recv_msg(conn)
            if msg_type != MSG_SECURE_DATA:
                return

            did_j     = payload["did_j"]        # bytes
            encrypted = payload["encrypted"]    # bytes
            mode      = payload.get("mode", "unknown")

            did_hex = did_j.hex()
            sk = self._sessions.get(did_hex)

            if sk is None:
                logger.warning("User: no session key for DIDj=%s — cannot decrypt", did_hex[:12])
                return

            plaintext = aes_utils.decrypt(encrypted, sk)

            utils.banner(logger, "SECURE DATA EXCHANGE")
            logger.info("Received encrypted message from drone DIDj=%s", did_hex[:12])
            logger.info("  Encryption : %s", mode)
            logger.info("  Ciphertext : %s...", encrypted.hex()[:24])
            logger.info("  Session key: %s...", sk.hex()[:12])
            logger.info("  Decrypted  : %s", plaintext)
            logger.info("Secure data exchange successful!")

        except Exception as exc:
            logger.error("Secure data receive error: %s", exc)
        finally:
            conn.close()

    def _run_mu3_server(self) -> None:
        """TCP server that accepts mu3 connections from drones."""
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((USER_AUTH_HOST, USER_AUTH_PORT))
        srv.listen(20)
        logger.info("User mu3 listener on %s:%d", USER_AUTH_HOST, USER_AUTH_PORT)

        while True:
            conn, addr = srv.accept()
            t = threading.Thread(target=self._handle_mu3, args=(conn,), daemon=True)
            t.start()

    def _handle_mu3(self, conn: socket.socket) -> None:
        """Receive one mu3 from a drone and store it."""
        try:
            msg_type, payload = recv_msg(conn)
            if msg_type != MSG_MU3:
                logger.warning("User expected MU3, got %s", msg_type)
                return

            # Reconstruct a Mu3 dataclass from the network payload.
            mu3 = dmod.Mu3(
                m5     = payload["m5"],
                did_j  = payload["did_j"],
                Ij     = (payload["Rj"], payload["sj"]),
                kj_pub = payload["kj_pub"],
            )
            with self._mu3_lock:
                self._mu3_batch.append(mu3)
                received = len(self._mu3_batch)

            logger.info("User received MU3 from Drone (DIDj=%s)  (%d/%d)",
                        utils.short_hx(mu3.did_j), received, self.num_drones)

            if received >= self.num_drones:
                self._mu3_ready.set()

        except ConnectionError:
            pass  # port-probe, ignore
        except Exception as exc:
            logger.error("mu3 receive error: %s", exc)
        finally:
            conn.close()

    # ------------------------------------------------------------------ #
    # Step 3 — Build and send mu1, then wait for all mu3                  #
    # ------------------------------------------------------------------ #
    def authenticate(self) -> None:
        """
        Full authentication flow:
          build mu1 -> send to CS -> wait for all mu3 -> batch verify -> SKij.
        """
        mu1 = self.user_obj.build_mu1()

        logger.info("User sending mu1 to CS auth server at %s:%d ...",
                    self.cs_host, CS_AUTH_PORT)
        with socket.create_connection((self.cs_host, CS_AUTH_PORT), timeout=15) as sock:
            send_msg(sock, MSG_MU1, {
                "pid_u": mu1.pid_u,
                "m1":    mu1.m1,
                "m2":    mu1.m2,
                "m3":    mu1.m3,
                "t1":    mu1.t1,
            }, sender=self.user_obj.id_u, receiver="CS", phase="authentication")
            # CS replies with the drone list (so user knows how many mu3 to expect).
            msg_type, payload = recv_msg(sock)

        if msg_type == MSG_ERROR:
            raise RuntimeError(f"Authentication rejected by CS: {payload['reason']}")

        drone_list = payload.get("drones", [])
        logger.info("CS confirmed batch of %d drone(s). Waiting for mu3 ...",
                    len(drone_list))

        # Update how many mu3 we need in case the CS batch differs from --drones.
        if drone_list:
            self.num_drones = len(drone_list)

        # Wait up to 30 s for all mu3 messages to arrive.
        arrived = self._mu3_ready.wait(timeout=30)
        if not arrived:
            with self._mu3_lock:
                count = len(self._mu3_batch)
            logger.warning("Timeout: received %d/%d mu3 messages. Proceeding with what arrived.",
                           count, self.num_drones)

        with self._mu3_lock:
            batch = list(self._mu3_batch)

        # ---- Batch verification -------------------------------------------
        logger.info("Starting Batch Verification...")
        result = self.user_obj.batch_verify(batch)

        # ---- Session key derivation per valid drone -----------------------
        sessions = self.user_obj.establish_sessions(batch, result)
        self._sessions = sessions

        if result.all_valid:
            logger.info("Batch Verification Successful.")

        utils.banner(logger, "AUTHENTICATION RESULTS")
        logger.info("All valid : %s", result.all_valid)
        logger.info("Valid     : %d drone(s)", len(result.valid_drones))
        logger.info("Invalid   : %d drone(s)", len(result.invalid_drones))
        logger.info("EC checks : %d", result.checks)

        utils.banner(logger, "SESSION KEYS (SKij)")
        for did_hex, sk in sessions.items():
            logger.info("  DIDj=%s  ->  SKij=%s", did_hex[:12], utils.short_hx(sk))
            logger.info("Session Key established with DIDj=%s", did_hex[:12])

        if result.invalid_drones:
            logger.warning("Invalid drones detected:")
            for did in result.invalid_drones:
                logger.warning("  DIDj = %s", utils.short_hx(did))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="SeBAC-IoD User node")
    parser.add_argument("--id",       default="User-01",      help="User identity IDu")
    parser.add_argument("--password", default="S3cure-Pass!",  help="User password")
    parser.add_argument("--drones",   type=int, default=3,     help="Expected number of drones")
    parser.add_argument("--cs-host",  default="127.0.0.1",     help="IP of the Control Server")
    args = parser.parse_args()

    app = UserApp(args.id, args.password, args.drones, cs_host=args.cs_host)

    app.start_mu3_listener()
    time.sleep(0.5)

    app.enroll()
    app.authenticate()

    # Wait a few seconds for encrypted messages from drones to arrive
    logger.info("Waiting for encrypted messages from drones ...")
    time.sleep(5)
    logger.info("User app complete.")


if __name__ == "__main__":
    main()
