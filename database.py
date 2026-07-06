"""
database.py
===========

In-memory database simulation for SeBAC-IoD.

The paper's Control Server persists registration material for every drone and
user, and tracks live sessions.  We model that persistence layer with plain
Python dictionaries wrapped in a typed ``Database`` class so the protocol code
gets clean, intention-revealing methods instead of poking at raw dicts.

Stored collections
-------------------
* ``users``    : UID/pseudo-identity -> :class:`UserRecord`
* ``drones``   : IDd               -> :class:`DroneRecord`
* ``pids``     : PIDu (hex)         -> UID            (reverse index for users)
* ``dids``     : DIDj (hex)         -> IDd            (reverse index for drones)
* ``sessions`` : session_id         -> :class:`SessionRecord`

Everything is type-hinted; lookups return ``None`` on a miss so callers can
make explicit decisions instead of catching ``KeyError`` everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import utils

logger = utils.get_logger(__name__)


# ---------------------------------------------------------------------------
# Record types
# ---------------------------------------------------------------------------
@dataclass
class DroneRecord:
    """
    Registration material the Control Server stores for one drone.

    Mirrors the values computed in the Drone Enrollment phase.  Bytes-valued
    fields hold raw 32-byte (SHA-256) images unless noted.
    """

    id_d: str                 # real drone identity IDd
    did_j: bytes              # DIDj : drone pseudo-identity
    ksd: bytes                # Ksd  : shared secret between drone and server
    kj_pub: bytes             # Kj   : drone public EC point (serialized bytes)
    kj_priv: int              # kj   : drone private scalar  (Kj = kj . Q)
    ej: bytes                 # Ej   : masked credential
    wj: bytes                 # Wj   : verifier term
    nj: bytes                 # Nj   : stored verifier / nonce-bound value
    challenge: bytes          # Cj   : enrollment challenge
    puf_response: bytes       # Rest : PUF response captured at enrollment
    device_secret: bytes = field(repr=False)  # secret (authority knows it here)


@dataclass
class UserRecord:
    """
    Registration material the Control Server stores for one user.

    Mirrors the values computed in the User Enrollment phase.
    """

    id_u: str                 # real user identity IDu
    uid: bytes                # UID  : user-derived identity commitment
    pid_u: bytes              # PIDu : user pseudo-identity issued by CS
    fid_s: bytes              # FIDs : fake/anonymous identity issued by CS
    bu: bytes = b""           # Bu   : server credential delivered to the user
    # The following four terms are stored on the USER's device (derived from
    # the password), NOT by the Control Server.  They default to empty here so
    # the CS can create a record without them; the User class fills its own
    # local copy.  They are kept on the record for completeness/inspection.
    gamma_u: bytes = b""      # gamma_u : stored auth term
    delta_u: bytes = b""      # delta_u : stored auth term
    au: bytes = b""           # Au      : stored auth term
    du: bytes = b""           # Du      : stored auth term


@dataclass
class SessionRecord:
    """A live authenticated session between the user and one drone."""

    session_id: str
    user_id: str
    drone_id: str
    session_key: bytes
    timestamp: int
    valid: bool = True


# ---------------------------------------------------------------------------
# The database
# ---------------------------------------------------------------------------
class Database:
    """
    A tiny in-memory database backed by dictionaries.

    Not thread-safe and not persistent — by design, since this is a single
    process simulation.  Every mutating call is logged so the protocol log
    shows exactly what the server stored and when.
    """

    def __init__(self) -> None:
        self.users: Dict[str, UserRecord] = {}        # key: UID hex
        self.drones: Dict[str, DroneRecord] = {}      # key: IDd
        self.pids: Dict[str, str] = {}                # PIDu hex -> UID hex
        self.dids: Dict[str, str] = {}                # DIDj hex -> IDd
        self.sessions: Dict[str, SessionRecord] = {}  # key: session_id
        logger.debug("Database initialized (in-memory)")

    # ------------------------------------------------------------------ drones
    def add_drone(self, record: DroneRecord) -> None:
        """Insert/replace a drone registration record and index its DIDj."""
        self.drones[record.id_d] = record
        self.dids[record.did_j.hex()] = record.id_d
        logger.info(
            "DB store DRONE: IDd=%s DIDj=%s", record.id_d, utils.short_hx(record.did_j)
        )

    def get_drone(self, id_d: str) -> Optional[DroneRecord]:
        """Fetch a drone record by its real identity IDd (or None)."""
        return self.drones.get(id_d)

    def get_drone_by_did(self, did_j: bytes) -> Optional[DroneRecord]:
        """Fetch a drone record by its pseudo-identity DIDj (or None)."""
        id_d = self.dids.get(did_j.hex())
        return self.drones.get(id_d) if id_d is not None else None

    def all_drones(self) -> List[DroneRecord]:
        """Return every stored drone record (used to assemble a batch)."""
        return list(self.drones.values())

    # ------------------------------------------------------------------- users
    def add_user(self, record: UserRecord) -> None:
        """Insert/replace a user registration record and index its PIDu."""
        self.users[record.uid.hex()] = record
        self.pids[record.pid_u.hex()] = record.uid.hex()
        logger.info(
            "DB store USER: IDu=%s PIDu=%s", record.id_u, utils.short_hx(record.pid_u)
        )

    def get_user_by_uid(self, uid: bytes) -> Optional[UserRecord]:
        """Fetch a user record by its UID commitment (or None)."""
        return self.users.get(uid.hex())

    def get_user_by_pid(self, pid_u: bytes) -> Optional[UserRecord]:
        """Fetch a user record by its pseudo-identity PIDu (or None)."""
        uid_hex = self.pids.get(pid_u.hex())
        return self.users.get(uid_hex) if uid_hex is not None else None

    # ---------------------------------------------------------------- sessions
    def add_session(self, record: SessionRecord) -> None:
        """Record a freshly established user<->drone session."""
        self.sessions[record.session_id] = record
        logger.info(
            "DB store SESSION: %s  user=%s drone=%s SK=%s",
            record.session_id,
            record.user_id,
            record.drone_id,
            utils.short_hx(record.session_key),
        )

    def get_session(self, session_id: str) -> Optional[SessionRecord]:
        """Fetch a session by id (or None)."""
        return self.sessions.get(session_id)

    def all_sessions(self) -> List[SessionRecord]:
        """Return every stored session record."""
        return list(self.sessions.values())

    # ----------------------------------------------------------------- summary
    def summary(self) -> str:
        """Return a one-line census of the database (for logs/printing)."""
        return (
            f"Database[users={len(self.users)}, drones={len(self.drones)}, "
            f"sessions={len(self.sessions)}]"
        )
