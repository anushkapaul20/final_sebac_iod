"""
server_app.py
=============

Control Server (CS) — TCP socket server for SeBAC-IoD.

Run this FIRST, before drone_app.py and user_app.py.

    python server_app.py

The server listens on two ports:
    * ENROLL_PORT (9000) — secure enrollment channel (drones + users register here)
    * AUTH_PORT   (9001) — insecure authentication channel (receives mu1, sends mu2 batch)

Protocol flow handled here
--------------------------
1. Drone sends DRONE_ENROLL_REQ  -> CS enrolls it, replies DRONE_ENROLL_ACK
2. User  sends USER_ENROLL_REQ   -> CS enrolls it, replies USER_ENROLL_ACK
3. User  sends MU1               -> CS verifies, builds mu2 for each drone,
                                    forwards MU2 to each drone (CS connects to
                                    each drone's AUTH port), collects MU3 from
                                    each drone, and sends MU2_BATCH (containing
                                    the drone list + mu2 params) back to the user.
   The user then receives MU3 directly from each drone.

Drone AUTH address book
-----------------------
Each drone registers its (host, auth_port) when it enrolls, so the CS knows
where to forward mu2 messages.
"""

from __future__ import annotations

import socket
import threading
from typing import Dict, List

import control_server as csmod
import utils
from protocol_messages import (
    MSG_DRONE_ENROLL_REQ, MSG_DRONE_ENROLL_ACK,
    MSG_USER_ENROLL_REQ,  MSG_USER_ENROLL_ACK,
    MSG_MU1,              MSG_MU2,
    MSG_ERROR,
    recv_msg, send_msg,
)

logger = utils.get_logger("server_app")

# Defaults — override via CLI arguments when running on real hardware
ENROLL_HOST = "0.0.0.0"    # listen on all interfaces (reachable from RPi)
ENROLL_PORT = 9000
AUTH_HOST   = "0.0.0.0"
AUTH_PORT   = 9001


class CSServer:
    """Wraps ControlServer logic behind TCP sockets."""

    def __init__(self, host: str = "0.0.0.0") -> None:
        self.host = host
        self.cs = csmod.ControlServer()
        self.cs.initialize()

        # Maps IDd -> (host, port) so CS can push mu2 to each drone.
        self._drone_auth_addrs: Dict[str, tuple] = {}
        self._lock = threading.Lock()

    def run_enroll_server(self) -> None:
        """Accept enrollment connections on ENROLL_PORT."""
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.host, ENROLL_PORT))
        srv.listen(20)
        logger.info("CS enrollment server listening on %s:%d", self.host, ENROLL_PORT)
        while True:
            conn, addr = srv.accept()
            t = threading.Thread(target=self._handle_enroll, args=(conn, addr), daemon=True)
            t.start()

    def _handle_enroll(self, conn: socket.socket, addr) -> None:
        """Handle one enrollment connection."""
        try:
            msg_type, payload = recv_msg(conn)
            if not msg_type:
                return  # port-probe connection, ignore silently

            if msg_type == MSG_DRONE_ENROLL_REQ:
                self._enroll_drone(conn, payload)
            elif msg_type == MSG_USER_ENROLL_REQ:
                self._enroll_user(conn, payload)
            else:
                send_msg(conn, MSG_ERROR, {"reason": f"Unknown enroll msg: {msg_type}"})
        except ConnectionError:
            pass  # port-probe or clean close, not a real error
        except Exception as exc:
            logger.error("Enroll error from %s: %s", addr, exc)
            try:
                send_msg(conn, MSG_ERROR, {"reason": str(exc)})
            except Exception:
                pass
        finally:
            conn.close()

    def _enroll_drone(self, conn: socket.socket, payload: dict) -> None:
        """Process a DRONE_ENROLL_REQ and respond with DRONE_ENROLL_ACK."""
        id_d         = payload["id_d"]
        v            = payload["v"]
        challenge    = payload["challenge"]
        puf_response = payload["puf_response"]
        device_secret= payload["device_secret"]
        auth_host    = payload.get("auth_host", "127.0.0.1")
        auth_port    = int(payload.get("auth_port", 0))

        logger.info("CS received mu0 (Drone Enrollment Request) from %s", id_d)

        record = self.cs.enroll_drone(id_d, v, challenge, puf_response, device_secret)
        creds  = self.cs.issue_drone_credentials(record)

        logger.info("CS generated credentials for %s :", id_d)
        logger.info("  DIDj = %s", utils.short_hx(creds["DIDj"]))
        logger.info("  Ksd  = %s", utils.short_hx(creds["Ksd"]))
        logger.info("  Ej   = %s", utils.short_hx(creds["Ej"]))
        logger.info("  Nj   = %s", utils.short_hx(creds["Nj"]))

        # Store the drone's auth address for mu2 forwarding.
        with self._lock:
            self._drone_auth_addrs[id_d] = (auth_host, auth_port)

        send_msg(conn, MSG_DRONE_ENROLL_ACK, {
            "DIDj": creds["DIDj"],
            "Ksd":  creds["Ksd"],
            "Kj":   creds["Kj"],
            "kj":   creds["kj"],
            "Ej":   creds["Ej"],
            "Wj":   creds["Wj"],
            "Nj":   creds["Nj"],
        }, sender="CS", receiver=id_d, phase="enrollment")
        logger.info("CS -> %s : DRONE_ENROLL_ACK sent", id_d)

    def _enroll_user(self, conn: socket.socket, payload: dict) -> None:
        """Process a USER_ENROLL_REQ and respond with USER_ENROLL_ACK."""
        id_u = payload["id_u"]
        uid  = payload["uid"]

        logger.info("CS received User Enrollment Request from %s", id_u)

        result = self.cs.enroll_user(id_u, uid)

        logger.info("CS generated credentials for %s :", id_u)
        logger.info("  PIDu = %s", utils.short_hx(result["PIDu"]))
        logger.info("  FIDs = %s", utils.short_hx(result["FIDs"]))
        logger.info("  Bu   = %s", utils.short_hx(result["Bu"]))

        send_msg(conn, MSG_USER_ENROLL_ACK, {
            "PIDu": result["PIDu"],
            "FIDs": result["FIDs"],
            "Bu":   result["Bu"],
        }, sender="CS", receiver=id_u, phase="enrollment")
        logger.info("CS -> %s : USER_ENROLL_ACK sent", id_u)

    def run_auth_server(self) -> None:
        """Accept authentication connections on AUTH_PORT."""
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.host, AUTH_PORT))
        srv.listen(20)
        logger.info("CS auth server listening on %s:%d", self.host, AUTH_PORT)
        while True:
            conn, addr = srv.accept()
            t = threading.Thread(target=self._handle_auth, args=(conn, addr), daemon=True)
            t.start()

    def _handle_auth(self, conn: socket.socket, addr) -> None:
        """Handle one authentication (mu1) connection from a user."""
        try:
            msg_type, payload = recv_msg(conn)
            if msg_type != MSG_MU1:
                send_msg(conn, MSG_ERROR, {"reason": f"Expected MU1, got {msg_type}"})
                return
            self._process_mu1(conn, payload)
        except ConnectionError:
            pass  # port-probe or clean close
        except Exception as exc:
            logger.error("Auth error from %s: %s", addr, exc)
            try:
                send_msg(conn, MSG_ERROR, {"reason": str(exc)})
            except Exception:
                pass
        finally:
            conn.close()

    def _process_mu1(self, conn: socket.socket, payload: dict) -> None:
        """
        Verify mu1, build mu2 for each drone, forward each mu2 to its drone,
        then tell the user which drones are in the batch (so the user can
        collect mu3 directly from them).
        """
        # Reconstruct a mu1-like object from the payload.
        class _Mu1:
            pass

        mu1 = _Mu1()
        mu1.pid_u = payload["pid_u"]
        mu1.m1    = payload["m1"]
        mu1.m2    = payload["m2"]
        mu1.m3    = payload["m3"]
        mu1.t1    = payload["t1"]

        logger.info("CS received MU1 from User (PIDu=%s)", utils.short_hx(mu1.pid_u))

        # CS verifies mu1 and builds per-drone mu2 messages.
        mu2_list = self.cs.build_mu2_batch(mu1)

        logger.info("CS generated MU2 for a batch of %d drone(s).", len(mu2_list))

        # Forward each mu2 to the corresponding drone.
        drones = self.cs.db.all_drones()
        drone_info: List[dict] = []

        for drone_record, mu2 in zip(drones, mu2_list):
            id_d = drone_record.id_d
            with self._lock:
                addr = self._drone_auth_addrs.get(id_d)

            if addr is None:
                logger.warning("No auth address for drone %s — skipping", id_d)
                continue

            try:
                self._send_mu2_to_drone(addr, mu2, id_d)
            except Exception as exc:
                logger.error("Failed to send mu2 to drone %s: %s", id_d, exc)
                continue

            drone_info.append({
                "id_d":      id_d,
                "did_j":     drone_record.did_j,
                "auth_host": addr[0],
                "auth_port": addr[1],
            })

        send_msg(conn, MSG_MU2, {"drones": drone_info},
                 sender="CS", receiver="User", phase="authentication")
        logger.info("CS sent MU2_BATCH info to user (%d drones).", len(drone_info))

    def _send_mu2_to_drone(self, addr: tuple, mu2: csmod.Mu2, id_d: str = "Drone") -> None:
        """Open a short-lived TCP connection to a drone and deliver mu2."""
        host, port = addr
        with socket.create_connection((host, port), timeout=10) as sock:
            send_msg(sock, MSG_MU2, {
                "yj": mu2.yj,
                "m5": mu2.m5,
                "g2": mu2.g2,
                "g3": mu2.g3,
                "t2": mu2.t2,
            }, sender="CS", receiver=id_d, phase="authentication")
            logger.info("CS -> %s : MU2 sent", id_d)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="SeBAC-IoD Control Server")
    parser.add_argument("--host", default="0.0.0.0",
                        help="Host/IP to bind (default: 0.0.0.0 = all interfaces)")
    args = parser.parse_args()

    cs_server = CSServer(host=args.host)

    t_enroll = threading.Thread(target=cs_server.run_enroll_server, daemon=True)
    t_auth   = threading.Thread(target=cs_server.run_auth_server,   daemon=True)

    t_enroll.start()
    t_auth.start()

    logger.info("SeBAC-IoD Control Server running on %s (ports %d/%d).  Press Ctrl+C to stop.",
                args.host, ENROLL_PORT, AUTH_PORT)
    try:
        t_enroll.join()
    except KeyboardInterrupt:
        logger.info("CS shutting down.")


if __name__ == "__main__":
    main()
