"""
ra.py
=====

Registration Authority (RA) for SeBAC-IoD.

Who is the RA?
--------------
In SeBAC-IoD, the Control Server (CS) is the trusted Registration Authority.
The CS is assumed to be a resourceful trusted server that enrolls users and
drones into the system through registration on a confidential channel.

This file separates the REGISTRATION part from the CS into a dedicated module.
It handles:
    1. Enrolling drones  -> issues DIDj, Ksd, Kj, Ej, Wj, Nj
    2. Enrolling users   -> issues PIDu, FIDs, Bu
    3. Saving credentials to JSON files (persistent storage)
    4. Loading credentials from JSON files on restart

Why JSON files?
---------------
The current database.py keeps everything in memory — lost on program exit.
JSON files make the registration persistent:
    registrations/
        system_params.json      <- CS public parameters
        drone_Drone-00.json     <- Drone-00 credentials
        drone_Drone-01.json     <- Drone-01 credentials
        user_User-01.json       <- User-01 credentials

Run this BEFORE server_app.py to pre-register drones and users.
server_app.py can then load from JSON instead of re-enrolling.

Usage
-----
    python ra.py --drones 2 --user User-01
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import control_server as csmod
import drone as dmod
import puf
import utils

logger = utils.get_logger("ra")

REGISTRATIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "registrations")


# ---------------------------------------------------------------------------
# JSON save/load helpers
# ---------------------------------------------------------------------------

def _to_json_safe(obj) -> object:
    """Recursively convert bytes to hex strings for JSON serialization."""
    if isinstance(obj, bytes):
        return obj.hex()
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(i) for i in obj]
    return obj


def _from_json_safe(obj, bytes_keys: set = None) -> object:
    """Convert hex strings back to bytes for specified keys."""
    if bytes_keys is None:
        bytes_keys = set()
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            if k in bytes_keys and isinstance(v, str):
                result[k] = bytes.fromhex(v)
            else:
                result[k] = _from_json_safe(v, bytes_keys)
        return result
    return obj


def save_json(filename: str, data: dict) -> None:
    """Save a dict to a JSON file in the registrations directory."""
    os.makedirs(REGISTRATIONS_DIR, exist_ok=True)
    path = os.path.join(REGISTRATIONS_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_json_safe(data), f, indent=2)
    logger.info("Saved: %s", path)


def load_json(filename: str) -> dict:
    """Load a JSON file from the registrations directory."""
    path = os.path.join(REGISTRATIONS_DIR, filename)
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Registration Authority
# ---------------------------------------------------------------------------

class RegistrationAuthority:
    """
    The Registration Authority — runs the CS enrollment procedures
    and saves all credentials to JSON files.
    """

    def __init__(self) -> None:
        self.cs = csmod.ControlServer(name="RA/CS")
        self.cs.initialize()
        self._save_system_params()

    def _save_system_params(self) -> None:
        """Save CS public parameters to system_params.json."""
        import crypto
        params = {
            "curve":   "NIST P-256 (secp256r1)",
            "prime_p": str(self.cs.prime_p),
            "order_n": str(self.cs.order_n),
            "Q":       crypto.point_to_bytes(self.cs.Q).hex(),
            "Ppub":    crypto.point_to_bytes(self.cs.Ppub).hex(),
        }
        save_json("system_params.json", params)
        logger.info("System parameters saved.")

    def register_drone(self, id_d: str) -> dict:
        """
        Enroll a drone and save its credentials to JSON.

        Steps (paper Section IV-B):
            1. Drone selects IDd, random v, generates Cj, computes Rest=PUF(Cj)
            2. CS computes DIDj, Ksd, kj, Kj, Ej, Wj, Nj
            3. Credentials saved to registrations/drone_<id_d>.json

        Parameters
        ----------
        id_d : str
            The drone's real identity (e.g. "Drone-00")

        Returns
        -------
        dict
            The credentials issued to the drone.
        """
        logger.info("RA enrolling drone: %s", id_d)

        # Drone side — generate enrollment request
        drone = dmod.Drone(id_d)
        req   = drone.begin_enrollment()

        # CS side — process enrollment
        record = self.cs.enroll_drone(
            req.id_d, req.v, req.challenge, req.puf_response, drone.device_secret
        )
        creds = self.cs.issue_drone_credentials(record)

        # Save to JSON
        drone_data = {
            "id_d":          id_d,
            "DIDj":          creds["DIDj"],
            "Ksd":           creds["Ksd"],
            "Kj":            creds["Kj"],
            "kj":            creds["kj"],
            "Ej":            creds["Ej"],
            "Wj":            creds["Wj"],
            "Nj":            creds["Nj"],
            "challenge":     req.challenge,
            "puf_response":  req.puf_response,
            "device_secret": drone.device_secret,
        }
        save_json(f"drone_{id_d}.json", drone_data)
        logger.info("Drone %s registered and saved to JSON.", id_d)
        return creds

    def register_user(self, id_u: str, password: str = "S3cure-Pass!") -> dict:
        """
        Enroll a user and save credentials to JSON.

        Steps (paper Section IV-C):
            1. User selects IDu, random r, computes UID = H(IDu || r)
            2. CS computes FIDs, PIDu, Bu
            3. Credentials saved to registrations/user_<id_u>.json

        Parameters
        ----------
        id_u : str
            The user's real identity (e.g. "User-01")
        password : str
            The user's password (biometric replacement)

        Returns
        -------
        dict
            The credentials issued to the user.
        """
        import user as umod
        import crypto

        logger.info("RA enrolling user: %s", id_u)

        # User side — begin enrollment
        usr     = umod.User(id_u, password)
        _, uid  = usr.begin_enrollment()

        # CS side — process enrollment
        result  = self.cs.enroll_user(id_u, uid)

        # Save to JSON
        user_data = {
            "id_u":     id_u,
            "uid":      uid,
            "PIDu":     result["PIDu"],
            "FIDs":     result["FIDs"],
            "Bu":       result["Bu"],
        }
        save_json(f"user_{id_u}.json", user_data)
        logger.info("User %s registered and saved to JSON.", id_u)
        return result

    def list_registered(self) -> None:
        """Print all registered drones and users."""
        utils.banner(logger, "REGISTERED ENTITIES")
        drones = self.cs.db.all_drones()
        users  = list(self.cs.db.users.values())
        logger.info("Drones : %d", len(drones))
        for d in drones:
            logger.info("  %s  DIDj=%s", d.id_d, utils.short_hx(d.did_j))
        logger.info("Users  : %d", len(users))
        for u in users:
            logger.info("  %s  PIDu=%s", u.id_u, utils.short_hx(u.pid_u))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="SeBAC-IoD Registration Authority")
    parser.add_argument("--drones",   type=int, default=1,
                        help="Number of drones to register (default: 1)")
    parser.add_argument("--user",     default="User-01",
                        help="User identity to register (default: User-01)")
    parser.add_argument("--password", default="S3cure-Pass!",
                        help="User password (default: S3cure-Pass!)")
    args = parser.parse_args()

    utils.banner(logger, "SeBAC-IoD REGISTRATION AUTHORITY")

    ra = RegistrationAuthority()

    # Register drones
    for i in range(args.drones):
        ra.register_drone(f"Drone-{i:02d}")

    # Register user
    ra.register_user(args.user, args.password)

    # Summary
    ra.list_registered()

    utils.banner(logger, "REGISTRATION COMPLETE")
    logger.info("JSON files saved in: %s", REGISTRATIONS_DIR)
    logger.info("Files created:")
    logger.info("  registrations/system_params.json")
    for i in range(args.drones):
        logger.info("  registrations/drone_Drone-%02d.json", i)
    logger.info("  registrations/user_%s.json", args.user)


if __name__ == "__main__":
    main()
