"""
user.py
=======

The User (U) entity for the SeBAC-IoD simulation.

The user authenticates a WHOLE BATCH of drones at once.  Its responsibilities:

    * Enrollment    : pick IDu, a random r, a password (the biometric stand-in),
                      compute UID = H(IDu || r), hand { IDu, UID } to the CS, and
                      then store the local verifiers (gamma_u, delta_u, Au, Du).
    * mu1           : open the authentication session by sending
                      { PIDu, M1, M2, M3, T1 } to the Control Server.
    * Batch verify  : collect every drone's mu3 and verify them all with a SINGLE
                      aggregated elliptic-curve equation using random weights Vecj.
    * Fault isolate : if the aggregate check fails, use divide-and-conquer to find
                      the offending drone(s) in O(log N) sub-checks.
    * Session keys  : derive SKij = H(PIDu || DIDj || Zj || Kj) per valid drone
                      with Zj = x . Kj (ECDH; the drone reproduces kj . X).

Biometric replacement (professor's instruction)
-----------------------------------------------
The paper feeds a biometric BIOu through a fuzzy extractor (Gen/Rep).  Here the
biometric is replaced by a password hash::

        sigma_bio = H( password || salt )

The ``salt`` plays the role of the fuzzy-extractor helper data (stored locally),
and ``sigma_bio`` plays the role of the extracted biometric key.  The
authentication FLOW is unchanged — only the source of the secret differs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import crypto
import utils

logger = utils.get_logger(__name__)


# ---------------------------------------------------------------------------
# Message containers
# ---------------------------------------------------------------------------
@dataclass
class Mu1:
    """The user -> CS session-opening message  { PIDu, M1, M2, M3, T1 }."""

    pid_u: bytes
    m1: bytes      # M1 = X = x . Q  (serialized ephemeral point)
    m2: bytes      # M2 = H(sigma_u || PIDu || T1) XOR UID   (masked UID)
    m3: bytes      # M3 = H(PIDu || UID || M1 || T1 || sigma_u)  (authenticator)
    t1: int        # T1 timestamp


@dataclass
class VerificationResult:
    """Outcome of the batch verification step."""

    all_valid: bool
    valid_drones: List[bytes]      # DIDj of drones that passed
    invalid_drones: List[bytes]    # DIDj of drones that failed
    checks: int                    # number of EC aggregate checks performed


class User:
    """
    A user U that batch-authenticates a set of drones.

    Attributes
    ----------
    id_u:
        Real identity IDu.
    password:
        The secret used in place of a biometric.
    """

    def __init__(self, id_u: str, password: str) -> None:
        """
        Construct a user holding a real identity and a password.

        Parameters
        ----------
        id_u:
            The user's real identity string (IDu).
        password:
            The password whose hash replaces the biometric BIOu.
        """
        self.id_u: str = id_u
        self._password: str = password

        # Enrollment-time secrets / state (filled by begin/complete enrollment).
        self._r: Optional[bytes] = None          # random r
        self._salt: Optional[bytes] = None       # password salt (helper data)
        self.uid: Optional[bytes] = None         # UID = H(IDu || r)
        self.pid_u: Optional[bytes] = None       # PIDu (recovered from Bu)
        self.fid_s: Optional[bytes] = None       # FIDs
        self._sigma_u: Optional[bytes] = None    # sigma_u = H(UID || s) shared w/ CS

        # Locally stored verifiers (paper: gamma_u, delta_u, Au, Du).
        self.gamma_u: Optional[bytes] = None
        self.delta_u: Optional[bytes] = None
        self.au: Optional[bytes] = None
        self.du: Optional[bytes] = None

        # Per-session ephemeral state (set when building mu1).
        self._x: Optional[int] = None            # ephemeral scalar x
        self._X: Optional[bytes] = None          # X = x . Q (serialized)
        self._t1: Optional[int] = None

        logger.debug("User constructed: IDu=%s", id_u)

    # ------------------------------------------------------------------ #
    # ENROLLMENT (user side)                                              #
    # ------------------------------------------------------------------ #
    def begin_enrollment(self) -> Tuple[str, bytes]:
        """
        Perform the first user-side enrollment steps and return { IDu, UID }.

            1. choose a random r
            2. choose a password salt (fuzzy-extractor helper-data stand-in)
            3. compute UID = H( IDu || r )

        Returns
        -------
        (str, bytes)
            ``(IDu, UID)`` to send to ``ControlServer.enroll_user``.
        """
        self._r = crypto.gen_nonce()
        self._salt = crypto.gen_salt()
        self.uid = crypto.sha256(self.id_u, self._r)

        logger.info("User %s begins enrollment:", self.id_u)
        logger.info("  r   = %s", utils.short_hx(self._r))
        logger.info("  UID = %s", utils.short_hx(self.uid))
        return self.id_u, self.uid

    def _bio_hash(self) -> bytes:
        """
        The biometric replacement:  sigma_bio = H( password || salt ).

        This is the single place the password is turned into a secret key; the
        rest of the flow treats it exactly as the paper treats the biometric
        key from the fuzzy extractor.
        """
        assert self._salt is not None, "salt missing; call begin_enrollment first"
        return crypto.sha256(self._password, self._salt)

    def complete_enrollment(self, cs_response: Dict[str, bytes]) -> None:
        """
        Finish enrollment using the Control Server's response { PIDu, FIDs, Bu }.

        The user recovers PIDu from Bu and computes its locally stored
        verifiers, binding them to the password hash (biometric replacement):

            PIDu     = Bu XOR H( UID || ??? )         # see note below
            sigma_u  = H( UID )-bound shared secret    (recovered via Bu)
            gamma_u  = H( PIDu || sigma_bio )          # binds password
            delta_u  = H( UID || sigma_bio || r )      # binds password + r
            Au       = sigma_u XOR sigma_bio           # masks sigma_u w/ password
            Du       = H( gamma_u || delta_u )         # integrity check value

        Note on Bu / sigma_u
        --------------------
        The CS issued  Bu = H(UID || s) XOR PIDu.  Only the CS knows s, but the
        user does NOT need s: it stores ``sigma_u = H(UID || s)`` *implicitly*
        by keeping ``Au = sigma_u XOR sigma_bio`` — and we obtain sigma_u here
        directly from Bu and PIDu because  sigma_u = Bu XOR PIDu.

        Parameters
        ----------
        cs_response:
            ``{ "PIDu", "FIDs", "Bu" }`` from ``ControlServer.enroll_user``.
        """
        assert self.uid is not None and self._r is not None

        self.pid_u = cs_response["PIDu"]
        self.fid_s = cs_response["FIDs"]
        bu = cs_response["Bu"]

        # sigma_u = H(UID || s) = Bu XOR PIDu  (user reconstructs the shared
        # secret without ever learning the master key s).
        self._sigma_u = utils.xor_bytes(bu, utils.fit(self.pid_u, 32))

        sigma_bio = self._bio_hash()                      # biometric stand-in

        # Locally stored verifiers, all bound to the password hash.
        self.gamma_u = crypto.sha256(self.pid_u, sigma_bio)
        self.delta_u = crypto.sha256(self.uid, sigma_bio, self._r)
        self.au = utils.xor_bytes(self._sigma_u, utils.fit(sigma_bio, 32))
        self.du = crypto.sha256(self.gamma_u, self.delta_u)

        logger.info("User %s completed enrollment (hash-based, no biometric):", self.id_u)
        logger.info("  PIDu    = %s", utils.short_hx(self.pid_u))
        logger.info("  gamma_u = %s", utils.short_hx(self.gamma_u))
        logger.info("  delta_u = %s", utils.short_hx(self.delta_u))
        logger.info("  Au      = %s", utils.short_hx(self.au))
        logger.info("  Du      = %s", utils.short_hx(self.du))

    # ------------------------------------------------------------------ #
    # LOGIN — local password check (biometric Rep() replacement)          #
    # ------------------------------------------------------------------ #
    def _local_login(self, password: str) -> bytes:
        """
        Reproduce sigma_u from a supplied password (the Rep() replacement).

        Recompute sigma_bio = H(password || salt), recover
        sigma_u = Au XOR sigma_bio, and verify it against Du via gamma_u.

        Returns
        -------
        bytes
            The recovered ``sigma_u`` if the password is correct.

        Raises
        ------
        PermissionError
            If the password fails the local Du integrity check.
        """
        assert self._salt is not None and self.au is not None
        sigma_bio = crypto.sha256(password, self._salt)
        sigma_u = utils.xor_bytes(self.au, utils.fit(sigma_bio, 32))

        # Recompute gamma_u/delta_u/Du and compare to detect a wrong password.
        gamma_chk = crypto.sha256(self.pid_u, sigma_bio)
        delta_chk = crypto.sha256(self.uid, sigma_bio, self._r)
        du_chk = crypto.sha256(gamma_chk, delta_chk)
        if du_chk != self.du:
            raise PermissionError(f"User {self.id_u}: local password check failed")
        return sigma_u

    # ------------------------------------------------------------------ #
    # mu1 — open the authentication session                               #
    # ------------------------------------------------------------------ #
    def build_mu1(self, password: Optional[str] = None) -> Mu1:
        """
        Build the session-opening message mu1 = { PIDu, M1, M2, M3, T1 }.

            1. local login -> recover sigma_u (verifies the password)
            2. ephemeral key  x  in Z_n^*,  X = x . Q
            3. T1 = current timestamp
            4. M1 = X
               M2 = H(sigma_u || PIDu || T1) XOR UID        (masks UID)
               M3 = H(PIDu || UID || M1 || T1 || sigma_u)   (authenticator)

        Parameters
        ----------
        password:
            Password to authenticate with (defaults to the constructor value).

        Returns
        -------
        Mu1
            The message to send to ``ControlServer``.
        """
        pwd = password if password is not None else self._password
        sigma_u = self._local_login(pwd)

        # ephemeral ECC key
        self._x = crypto.gen_scalar()
        X = crypto.scalar_mult(self._x)
        self._X = crypto.point_to_bytes(X)
        self._t1 = utils.now_ts()

        m1 = self._X
        mask = crypto.sha256(sigma_u, self.pid_u, self._t1)
        m2 = utils.xor_bytes(mask, utils.fit(self.uid, 32))
        m3 = crypto.sha256(self.pid_u, self.uid, m1, self._t1, sigma_u)

        logger.info("User %s built mu1:", self.id_u)
        logger.info("  PIDu = %s", utils.short_hx(self.pid_u))
        logger.info("  M1=X = %s", utils.short_hx(m1))
        logger.info("  M2   = %s", utils.short_hx(m2))
        logger.info("  M3   = %s", utils.short_hx(m3))
        logger.info("  T1   = %d", self._t1)

        return Mu1(pid_u=self.pid_u, m1=m1, m2=m2, m3=m3, t1=self._t1)

    # ------------------------------------------------------------------ #
    # BATCH VERIFICATION                                                  #
    # ------------------------------------------------------------------ #
    def _challenge_for(self, mu3) -> int:
        """
        Recompute the Schnorr challenge ej for one drone's mu3.

            ej = H( Rj || Kj || PIDu || DIDj || X )

        (Identical to the drone's computation in ``Drone.handle_mu2``.)
        """
        Rj_bytes, _sj = mu3.Ij
        return crypto.hash_to_int(Rj_bytes, mu3.kj_pub, self.pid_u, mu3.did_j, self._X)

    def _verify_single(self, mu3) -> bool:
        """
        Verify ONE drone's signature individually:  sj . Q == Rj + ej . Kj.

        Used as the leaf check inside divide-and-conquer fault isolation.
        """
        Rj_bytes, sj = mu3.Ij
        Rj = crypto.bytes_to_point(Rj_bytes)
        Kj = crypto.bytes_to_point(mu3.kj_pub)
        ej = self._challenge_for(mu3)
        lhs = crypto.scalar_mult(sj)                          # sj . Q
        rhs = crypto.point_add(Rj, crypto.scalar_mult(ej, Kj))  # Rj + ej . Kj
        return lhs == rhs

    def _aggregate_holds(self, batch: List) -> bool:
        """
        The core batch check over a list of mu3 messages.

        With per-drone random weights  d_j (Vecj)  the N individual equations
            sj . Q = Rj + ej . Kj
        are combined into ONE aggregated equation:

            ( sum_j  d_j * s_j ) . Q  ==  sum_j d_j . Rj  +  sum_j (d_j * e_j) . Kj

        The random weights stop a forger from making per-drone errors cancel.
        A single pair of EC multi-scalar sides verifies the whole batch.

        Returns
        -------
        bool
            ``True`` iff the aggregate equation holds for ``batch``.
        """
        if not batch:
            return True

        lhs_scalar = 0                       # sum_j d_j * s_j   (mod n)
        rhs_point = None                     # running EC sum on the right

        for mu3 in batch:
            Rj_bytes, sj = mu3.Ij
            Rj = crypto.bytes_to_point(Rj_bytes)
            Kj = crypto.bytes_to_point(mu3.kj_pub)
            ej = self._challenge_for(mu3)

            # Random weight d_j  (Vecj).  64-bit weights are plenty to make the
            # forgery-cancellation probability ~2^-64 while staying cheap.
            d_j = 1 + (utils.bytes_to_int(crypto.gen_nonce(8)))   # nonzero weight

            lhs_scalar = (lhs_scalar + d_j * sj) % crypto.ORDER_N

            term = crypto.point_add(
                crypto.scalar_mult(d_j, Rj),
                crypto.scalar_mult((d_j * ej) % crypto.ORDER_N, Kj),
            )
            rhs_point = term if rhs_point is None else crypto.point_add(rhs_point, term)

        lhs_point = crypto.scalar_mult(lhs_scalar)            # (sum d_j s_j) . Q
        return lhs_point == rhs_point

    def batch_verify(self, batch: List) -> VerificationResult:
        """
        Verify an entire batch of drone responses with the aggregated equation,
        falling back to divide-and-conquer to pinpoint any invalid drones.

        Parameters
        ----------
        batch:
            List of ``Mu3`` messages, one per responding drone.

        Returns
        -------
        VerificationResult
            ``all_valid``, the lists of valid/invalid DIDj, and how many
            aggregate checks were performed (illustrates the O(log N) cost on
            failure vs. a single check on success).
        """
        utils.banner(logger, "PHASE 4 - BATCH VERIFICATION")
        counter = {"checks": 0}

        if self._aggregate_holds_counted(batch, counter):
            dids = [m.did_j for m in batch]
            logger.info(
                "Batch of %d drones verified in ONE aggregate check.", len(batch)
            )
            return VerificationResult(True, dids, [], counter["checks"])

        # Aggregate failed -> at least one bad drone; isolate by divide & conquer.
        logger.info("Aggregate check FAILED - starting divide-and-conquer isolation.")
        valid: List[bytes] = []
        invalid: List[bytes] = []
        self._divide_and_conquer(batch, valid, invalid, counter)

        logger.info(
            "Isolation done: %d valid, %d invalid (%d aggregate checks).",
            len(valid), len(invalid), counter["checks"],
        )
        return VerificationResult(False, valid, invalid, counter["checks"])

    def _aggregate_holds_counted(self, batch: List, counter: Dict[str, int]) -> bool:
        """``_aggregate_holds`` wrapper that tallies how many checks were run."""
        counter["checks"] += 1
        return self._aggregate_holds(batch)

    def _divide_and_conquer(
        self,
        batch: List,
        valid: List[bytes],
        invalid: List[bytes],
        counter: Dict[str, int],
    ) -> None:
        """
        Recursively isolate invalid drones.

        Strategy
        --------
        * If the sub-batch's aggregate check passes, ALL its drones are valid.
        * If it has a single drone and fails, that drone is the culprit.
        * Otherwise split in half and recurse on each side.

        This finds every bad drone in O(k log N) aggregate checks (k = number of
        bad drones), far cheaper than N individual verifications when k is small.
        """
        if not batch:
            return

        if self._aggregate_holds_counted(batch, counter):
            valid.extend(m.did_j for m in batch)
            return

        if len(batch) == 1:
            invalid.append(batch[0].did_j)
            logger.info("  -> invalid drone isolated: DIDj=%s",
                        utils.short_hx(batch[0].did_j))
            return

        mid = len(batch) // 2
        self._divide_and_conquer(batch[:mid], valid, invalid, counter)
        self._divide_and_conquer(batch[mid:], valid, invalid, counter)

    # ------------------------------------------------------------------ #
    # SESSION KEYS (user side)                                            #
    # ------------------------------------------------------------------ #
    def derive_session_key(self, mu3) -> bytes:
        """
        Derive SKij with one drone from its mu3 using ECDH  Zj = x . Kj:

            SKij = H( PIDu || DIDj || Zj || Kj )

        (The drone reproduces the same key as  Zj = kj . X.)

        Parameters
        ----------
        mu3:
            The drone's verified response.

        Returns
        -------
        bytes
            The 32-byte session key SKij.
        """
        Kj = crypto.bytes_to_point(mu3.kj_pub)
        Zj = crypto.scalar_mult(self._x, Kj)                 # Zj = x . Kj
        sk = crypto.derive_session_key(
            self.pid_u, mu3.did_j, crypto.point_to_bytes(Zj), mu3.kj_pub
        )
        logger.info(
            "User %s <-> drone %s session key SKij = %s",
            self.id_u, utils.short_hx(mu3.did_j), utils.short_hx(sk),
        )
        return sk

    def establish_sessions(self, batch: List, result: VerificationResult) -> Dict[str, bytes]:
        """
        After verification, derive a session key for every VALID drone.

        Parameters
        ----------
        batch:
            The list of all mu3 messages.
        result:
            The :class:`VerificationResult` from :meth:`batch_verify`.

        Returns
        -------
        Dict[str, bytes]
            Map of ``DIDj_hex -> SKij`` for each valid drone.
        """
        valid_set = {d.hex() for d in result.valid_drones}
        sessions: Dict[str, bytes] = {}
        for mu3 in batch:
            if mu3.did_j.hex() in valid_set:
                sessions[mu3.did_j.hex()] = self.derive_session_key(mu3)
        return sessions
