"""
drone.py
========

The Drone (Dj) entity for the SeBAC-IoD simulation.

A drone plays two roles across the protocol:

    * Enrollment  : it creates its identity IDd, a random v, a PUF challenge Cj
                    and the response Rest, then hands { IDd, v, Cj, Rest } to the
                    Control Server (which stores the credentials).  Afterwards
                    the CS provisions the on-board material
                    { DIDj, Ksd, Kj, kj, Ej, Wj, Nj } back to the drone.

    * Authentication : during the batch phase the drone receives mu2 from the
                       Control Server, validates it, and replies with mu3 — a
                       Schnorr-style signature Ij = (Rj, sj) over the session
                       transcript that the User can batch-verify against Kj.

The session key SKij is derived from an ECDH value shared with the user:

        Zj  = kj . X            (drone uses its private kj and the user point X)
        SKij = H( PIDu || DIDj || Zj || Kj )

which the user reproduces as  Zj = x . Kj.  The key is therefore never sent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import crypto
import puf
import utils

logger = utils.get_logger(__name__)


# ---------------------------------------------------------------------------
# Message containers
# ---------------------------------------------------------------------------
@dataclass
class EnrollmentRequest:
    """The drone -> CS enrollment request  { IDd, v, Cj, Rest }."""

    id_d: str
    v: bytes
    challenge: bytes      # Cj
    puf_response: bytes   # Rest


@dataclass
class Mu3:
    """
    The drone -> user authentication response  mu3.

    Fields
    ------
    m5:
        The transcript binder  M5 = H(sigma_u || PIDu || DIDj || T1)  echoed
        back so the user can confirm the drone saw the right session context.
    did_j:
        The drone's pseudo-identity  DIDj.
    Ij:
        The Schnorr signature pair  (Rj_bytes, sj) where Rj = rj . Q and
        sj = rj + ej . kj  (mod n).  ``Rj_bytes`` is the serialized point.
    kj_pub:
        The drone's public key  Kj  (serialized), used by the user in batch
        verification and in the session-key/ECDH computation.
    """

    m5: bytes
    did_j: bytes
    Ij: tuple          # (Rj_bytes: bytes, sj: int)
    kj_pub: bytes      # Kj


class Drone:
    """
    A single drone Dj in the Internet of Drones.

    Attributes
    ----------
    id_d:
        Real identity IDd.
    puf_device:
        The on-board :class:`puf.PUFDevice` (holds the private device secret).
    credentials:
        The on-board material provisioned by the CS after enrollment
        (``DIDj, Ksd, Kj, kj, Ej, Wj, Nj``).  Empty until enrolled.
    """

    def __init__(self, id_d: str) -> None:
        """
        Construct a drone with a fresh PUF device.

        Parameters
        ----------
        id_d:
            The drone's real identity string (IDd).
        """
        self.id_d: str = id_d
        self.puf_device: puf.PUFDevice = puf.PUFDevice(device_id=id_d)

        # On-board credentials (filled by store_credentials after enrollment).
        self.credentials: Dict[str, object] = {}

        # Per-session ephemeral state kept between mu2 receipt and SKij
        # derivation (the random rj and the user's point X).
        self._rj: Optional[int] = None
        self._user_point = None            # X received inside mu2 (as EC point)
        self._pid_u: Optional[bytes] = None

        logger.debug("Drone constructed: IDd=%s", id_d)

    # ------------------------------------------------------------------ #
    # ENROLLMENT (drone side)                                             #
    # ------------------------------------------------------------------ #
    def begin_enrollment(self) -> EnrollmentRequest:
        """
        Perform the drone-side steps of the Drone Enrollment Procedure.

            1. select identity IDd (already set) and a random value v
            2. generate a PUF challenge Cj
            3. compute the response Rest = PUF(Cj)
            4. return the request { IDd, v, Cj, Rest } for the Control Server

        Returns
        -------
        EnrollmentRequest
            The bundle to hand to ``ControlServer.enroll_drone`` (the drone's
            ``device_secret`` is provisioned to the CS separately over the
            secure enrollment channel — see :meth:`device_secret`).
        """
        v = crypto.gen_nonce()
        challenge = puf.generate_challenge()                  # Cj
        response = puf.response_generation(self.puf_device, challenge)  # Rest

        logger.info("Drone %s begins enrollment:", self.id_d)
        logger.info("  v    = %s", utils.short_hx(v))
        logger.info("  Cj   = %s", utils.short_hx(challenge))
        logger.info("  Rest = %s", utils.short_hx(response))

        return EnrollmentRequest(
            id_d=self.id_d, v=v, challenge=challenge, puf_response=response
        )

    @property
    def device_secret(self) -> bytes:
        """Expose the PUF device secret for provisioning to the CS at enrollment."""
        return self.puf_device.device_secret

    def store_credentials(self, credentials: Dict[str, object]) -> None:
        """
        Store the on-board credentials returned by the Control Server.

        Parameters
        ----------
        credentials:
            The dict from ``ControlServer.issue_drone_credentials``:
            ``{ DIDj, Ksd, Kj, kj, Ej, Wj, Nj }``.
        """
        self.credentials = dict(credentials)
        logger.info(
            "Drone %s stored credentials: DIDj=%s",
            self.id_d,
            utils.short_hx(self.credentials["DIDj"]),  # type: ignore[arg-type]
        )

    # ------------------------------------------------------------------ #
    # AUTHENTICATION (drone side) — receive mu2, emit mu3                 #
    # ------------------------------------------------------------------ #
    def handle_mu2(self, mu2: "Mu2Like") -> Mu3:
        """
        Validate mu2 from the Control Server and build the response mu3.

        Steps
        -----
        1. Freshness: reject if the timestamp T2 is outside the allowed window.
        2. Authenticate the CS: recompute Yj from the drone's own Ksd, DIDj and
           the relayed (G2=X, G3=PIDu, T2) and compare with the received Yj.
        3. Pick a per-session random rj, set Rj = rj . Q.
        4. Compute the challenge ej = H(Rj || Kj || PIDu || DIDj || X) and the
           Schnorr response sj = rj + ej . kj (mod n).  Ij = (Rj, sj).
        5. Stash (rj, X, PIDu) so the session key can be derived after the user
           confirms the batch, and return mu3 = { M5, DIDj, Ij, Kj }.

        Parameters
        ----------
        mu2:
            The message object carrying ``Yj, M5, G2 (=X bytes), G3 (=PIDu),
            T2``.

        Returns
        -------
        Mu3
            The drone's authentication response.

        Raises
        ------
        ValueError
            If mu2 is stale or the CS authentication check on Yj fails.
        """
        if not self.credentials:
            raise ValueError(f"Drone {self.id_d} has no credentials; enroll first.")

        # --- 1. freshness check on T2 --------------------------------------
        if not utils.is_fresh(mu2.t2):
            raise ValueError(f"Drone {self.id_d}: mu2 timestamp T2 not fresh (replay?)")

        ksd: bytes = self.credentials["Ksd"]      # type: ignore[assignment]
        did_j: bytes = self.credentials["DIDj"]   # type: ignore[assignment]
        kj_priv: int = self.credentials["kj"]     # type: ignore[assignment]
        kj_pub: bytes = self.credentials["Kj"]    # type: ignore[assignment]

        # --- 2. authenticate the Control Server via Yj ---------------------
        expected_yj = crypto.sha256(ksd, did_j, mu2.g3, mu2.g2, mu2.t2)
        if not puf._ct_equal(expected_yj, mu2.yj):
            raise ValueError(f"Drone {self.id_d}: Yj mismatch - CS authentication failed")

        # G3 carries PIDu, G2 carries the user's ephemeral point X (bytes).
        pid_u: bytes = mu2.g3
        user_point = crypto.bytes_to_point(mu2.g2)            # X

        # --- 3. ephemeral commitment Rj = rj . Q ---------------------------
        rj = crypto.gen_scalar()
        Rj = crypto.scalar_mult(rj)
        Rj_bytes = crypto.point_to_bytes(Rj)

        # --- 4. Schnorr challenge ej and response sj -----------------------
        ej = crypto.hash_to_int(Rj_bytes, kj_pub, pid_u, did_j, mu2.g2)
        sj = (rj + ej * kj_priv) % crypto.ORDER_N
        Ij = (Rj_bytes, sj)

        # --- 5. transcript binder M5 + stash session state -----------------
        m5 = mu2.m5  # echo the CS-provided binder so the user can correlate
        self._rj = rj
        self._user_point = user_point
        self._pid_u = pid_u

        logger.info("Drone %s validated mu2 and built mu3:", self.id_d)
        logger.info("  Rj = %s", utils.short_hx(Rj_bytes))
        logger.info("  sj = %s", utils.short_hx(sj))

        return Mu3(m5=m5, did_j=did_j, Ij=Ij, kj_pub=kj_pub)

    # ------------------------------------------------------------------ #
    # SESSION KEY (drone side)                                            #
    # ------------------------------------------------------------------ #
    def derive_session_key(self, pid_u: Optional[bytes] = None) -> bytes:
        """
        Derive the shared session key SKij with the user.

        Uses the ECDH value  Zj = kj . X  (the user reproduces  Zj = x . Kj):

            SKij = H( PIDu || DIDj || Zj || Kj )

        Parameters
        ----------
        pid_u:
            Optional override of the user pseudo-identity; defaults to the one
            captured from mu2.

        Returns
        -------
        bytes
            The 32-byte session key SKij.

        Raises
        ------
        RuntimeError
            If called before a successful :meth:`handle_mu2`.
        """
        if self._user_point is None:
            raise RuntimeError(f"Drone {self.id_d}: no session state; handle mu2 first.")

        pidu = pid_u if pid_u is not None else self._pid_u
        kj_priv: int = self.credentials["kj"]     # type: ignore[assignment]
        kj_pub: bytes = self.credentials["Kj"]    # type: ignore[assignment]
        did_j: bytes = self.credentials["DIDj"]   # type: ignore[assignment]

        Zj = crypto.scalar_mult(kj_priv, self._user_point)   # Zj = kj . X
        Zj_bytes = crypto.point_to_bytes(Zj)
        sk = crypto.derive_session_key(pidu, did_j, Zj_bytes, kj_pub)

        logger.info("Drone %s derived session key SKij = %s", self.id_d, utils.short_hx(sk))
        return sk


# Structural type hint alias for mu2 (defined fully in control_server.py).
# A drone only needs these attributes: yj, m5, g2, g3, t2.
Mu2Like = object
