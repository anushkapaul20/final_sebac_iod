"""
socket_main.py
==============

All-in-one launcher for the SeBAC-IoD socket simulation.

Run this single script to start the CS server, all drone nodes, and the user
node inside the SAME process (each in its own thread), so you can watch the
full networked protocol flow without opening multiple terminals.

    python socket_main.py [--drones N]

The output shows the exact messages sent and received over sockets, making
the protocol phases (enrollment, mu1, mu2, mu3, batch verify, SK derivation)
clearly visible in the log.

How threads map to entities
---------------------------
    Thread 1 : CS enrollment server   (port 9000)
    Thread 2 : CS auth server          (port 9001)
    Thread 3 : Drone-00 listener       (port 9100)
    Thread 4 : Drone-01 listener       (port 9101)
    ...
    Thread N : User mu3 listener       (port 9200)
    Main     : orchestration (enroll all entities, then trigger authentication)
"""

from __future__ import annotations

import argparse
import socket
import threading
import time
from typing import List

# Import the app classes directly (not subprocess) so they share the same
# in-process logger and we get all output in one console window.
import drone as dmod
import user as umod
import control_server as csmod
import utils
from drone_app  import DroneApp
from user_app   import UserApp
from server_app import CSServer
from protocol_messages import (
    MSG_DRONE_ENROLL_REQ, MSG_USER_ENROLL_REQ,
    recv_msg, send_msg,
)

logger = utils.get_logger("socket_main")

BASE_DRONE_PORT = 9100   # first drone listens here; next on 9101, 9102, ...
USER_MU3_PORT   = 9200


def wait_for_port(host: str, port: int, timeout: float = 10.0) -> None:
    """Block until a TCP port is accepting connections (server is ready)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.1)
    raise TimeoutError(f"Port {port} did not open within {timeout}s")


def run_socket_simulation(num_drones: int = 3) -> None:
    utils.banner(logger, "SeBAC-IoD SOCKET SIMULATION START")
    logger.info("Simulating %d drones over TCP sockets.", num_drones)

    # ------------------------------------------------------------------ #
    # 1. Start the Control Server (both ports)                            #
    # ------------------------------------------------------------------ #
    cs_server = CSServer()

    t_cs_enroll = threading.Thread(target=cs_server.run_enroll_server, daemon=True,
                                    name="CS-Enroll")
    t_cs_auth   = threading.Thread(target=cs_server.run_auth_server,   daemon=True,
                                    name="CS-Auth")
    t_cs_enroll.start()
    t_cs_auth.start()

    wait_for_port("127.0.0.1", 9000)
    wait_for_port("127.0.0.1", 9001)
    logger.info("CS servers are ready.")

    # ------------------------------------------------------------------ #
    # 2. Create and enroll drones                                         #
    # ------------------------------------------------------------------ #
    drone_apps: List[DroneApp] = []

    for i in range(num_drones):
        port = BASE_DRONE_PORT + i
        app  = DroneApp(f"Drone-{i:02d}", port)
        drone_apps.append(app)

        # Enroll synchronously (each drone connects to CS and waits for ACK).
        app.enroll()

        # Start the drone's mu2 listener.
        t = threading.Thread(target=app.run_auth_listener, daemon=True,
                              name=f"Drone-{i:02d}-Listener")
        t.start()
        wait_for_port("127.0.0.1", port)

    logger.info("All %d drones enrolled and listening.", num_drones)

    # ------------------------------------------------------------------ #
    # 3. Create and enroll the user                                       #
    # ------------------------------------------------------------------ #
    user_app = UserApp("User-01", "S3cure-Pass!", num_drones)

    # Start the user's mu3 listener BEFORE enrollment/authentication.
    user_app.start_mu3_listener()
    wait_for_port("127.0.0.1", USER_MU3_PORT)

    user_app.enroll()
    logger.info("User enrolled.")

    # ------------------------------------------------------------------ #
    # 4. Run batch authentication                                         #
    # ------------------------------------------------------------------ #
    utils.banner(logger, "STARTING BATCH AUTHENTICATION OVER SOCKETS")
    user_app.authenticate()

    # Wait for encrypted messages from drones (sent 2s after mu3)
    logger.info("Waiting for encrypted data exchange ...")
    import time as _t; _t.sleep(5)

    utils.banner(logger, "SeBAC-IoD SOCKET SIMULATION COMPLETE")

    # Save and print communication cost summary
    try:
        from comm_logger import get_comm_logger
        clog = get_comm_logger()
        clog.save()
        clog.print_summary()
    except Exception as exc:
        logger.warning("Could not save comm log: %s", exc)


def main() -> None:
    parser = argparse.ArgumentParser(description="SeBAC-IoD socket simulation launcher")
    parser.add_argument("--drones", type=int, default=3,
                        help="Number of drones to simulate (default: 3)")
    args = parser.parse_args()

    run_socket_simulation(num_drones=args.drones)


if __name__ == "__main__":
    main()
