"""
control_server.py
==================

The Control Server (CS) entity for the SeBAC-IoD simulation.

The Control Server is the fully trusted registration authority of the scheme.
Across the protocol it is responsible for:

    * Initialization     -> set up the public system parameters and the master
                            secret key  (THIS FILE / THIS PART)
    * Drone Enrollment    -> register drones        (added in a later part)
    * User Enrollment     -> register users          (added in a later part)
    * Batch Authentication-> broker the mu1/mu2/mu3 flow (added in a later part)

This first part contains only:
    1. imports
    2. class ControlServer
    3. __init__()
    4. the Initialization Procedure

The remaining procedures will be appended to this same class in subsequent
parts, so the class body intentionally ends right after initialization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import crypto
import database
import puf
import utils

logger = utils.get_logger(__name__)


@dataclass
class Mu2:
    """
    The Control Server -> Drone message  mu2.

    One mu2 is built per drone in the batch.  Field meanings:

    yj:
        CS authenticator for this drone:  Yj = H(Ksd || DIDj || PIDu || X || T2).
        Only a party knowing the drone's Ksd (i.e. the CS) can produce it, so
        the drone authenticates the CS by recomputing and comparing it.
    m5:
        Session-transcript binder  M5 = H(sigma_u || PIDu || DIDj || T2).
    g2:
        Relayed user ephemeral point  X = x . Q  (serialized) — used by the
        drone for its signature challenge and for ECDH session-key derivation.
    g3:
        The user pseudo-identity  PIDu.
    t2:
        Timestamp T2 for freshness/replay protection.
    """

    yj: bytes
    m5: bytes
    g2: bytes      # X (serialized)
    g3: bytes      # PIDu
    t2: int


class ControlServer:
    """
    The trusted Control Server (CS) of SeBAC-IoD.

    Attributes
    ----------
    name:
        Human-readable label for logs (default "ControlServer").
    db:
        The in-memory :class:`database.Database` holding all registration and
        session records.
    crp_store:
        A :class:`puf.CRPStore` keeping each drone's Challenge-Response Pairs
        captured during enrollment (verifier-side PUF state).

    Public system parameters (set during initialization)
    ----------------------------------------------------
    These are published to every entity and are NOT secret:
        prime_p : the field prime p of the elliptic curve E(F_p)
        order_n : the prime order n of the base point Q
        Q       : the base point / generator of the curve group
        Ppub    : the server public key, where  Ppub = s . Q

    Master secret (set during initialization)
    -----------------------------------------
        _s : the Control Server's private master key s  (kept private; the
             leading underscore marks it as internal — it must never leave CS).
    """

    def __init__(self, name: str = "ControlServer") -> None:
        """
        Construct an *uninitialized* Control Server.

        The cryptographic system parameters are deliberately NOT created here;
        they are produced by :meth:`initialize` so that the Initialization
        Procedure is an explicit, logged protocol step (matching the paper,
        which treats system setup as its own phase).

        Parameters
        ----------
        name:
            Label used in log lines.
        """
        self.name: str = name

        # Persistence + PUF verifier state.
        self.db: database.Database = database.Database()
        self.crp_store: puf.CRPStore = puf.CRPStore()

        # ---- Public system parameters (filled in by initialize()) ----------
        self.prime_p: Optional[int] = None          # field prime p
        self.order_n: Optional[int] = None           # group order n
        self.Q = None                                # base point Q (EC point)
        self.Ppub = None                             # public key Ppub = s . Q

        # ---- Master secret (filled in by initialize(); never published) ----
        self._s: Optional[int] = None                # private master key s

        # Tracks whether initialize() has run, so other phases can guard.
        self._initialized: bool = False

        logger.debug("%s constructed (awaiting initialization)", self.name)

    # ------------------------------------------------------------------ #
    # 1. INITIALIZATION PROCEDURE                                         #
    # ------------------------------------------------------------------ #
    def initialize(self) -> None:
        """
        Initialization Procedure of SeBAC-IoD.

        The Control Server bootstraps the whole system:

            1. Fix the elliptic curve domain parameters over the prime field
               F_p:   the curve E, its base point Q (generator) and the prime
               order n of Q.  (These come from the standard NIST P-256 curve
               configured in ``crypto.py``.)
            2. Adopt SHA-256 as the one-way hash function H() used everywhere
               (provided by ``crypto.sha256`` / ``crypto.hash_to_int``).
            3. Choose a random master secret key  s  in  Z_n^*.
            4. Compute the corresponding public key  Ppub = s . Q.
            5. Publish the public parameters { E(F_p), Q, n, p, Ppub, H() } and
               keep s private.

        After this call the public parameters and ``Ppub`` are available as
        attributes, while ``_s`` is retained privately by the server.

        Returns
        -------
        None
            Results are stored on ``self`` and printed/logged.

        Raises
        ------
        RuntimeError
            If initialization is attempted more than once.
        """
        if self._initialized:
            raise RuntimeError(
                f"{self.name}.initialize() called twice — system parameters "
                "are already established."
            )

        utils.banner(logger, "PHASE 1 - INITIALIZATION PROCEDURE")

        # --- Step 1: adopt the standard curve domain parameters -------------
        # crypto.py exposes the NIST P-256 base point Q, its order n, and the
        # field prime p.  We simply publish references to them.
        self.Q = crypto.Q
        self.order_n = crypto.ORDER_N
        self.prime_p = crypto.PRIME_P

        # --- Step 2: hash function -----------------------------------------
        # H() is SHA-256 throughout; nothing to instantiate, but we log the
        # choice so the Initialization phase is self-documenting.
        hash_name = "SHA-256"

        # --- Step 3: choose the master secret key s in Z_n^* ----------------
        # --- Step 4: compute the public key Ppub = s . Q --------------------
        # gen_keypair() returns (s, s.Q) in one shot using a CSPRNG scalar.
        self._s, self.Ppub = crypto.gen_keypair()

        # --- Step 5: publish + print every generated value ------------------
        self._initialized = True

        logger.info("%s adopted curve  : %s", self.name, "NIST P-256 (secp256r1)")
        logger.info("%s hash function H : %s", self.name, hash_name)
        logger.info("%s prime field p   : %d-bit", self.name, self.prime_p.bit_length())
        logger.info("%s group order n   : %d-bit", self.name, self.order_n.bit_length())
        logger.info(
            "%s base point Q    : %s", self.name, utils.short_hx(crypto.point_to_bytes(self.Q))
        )
        # The master key s is secret: log only a short preview to the file.
        logger.debug("%s master key s    : %s", self.name, utils.short_hx(self._s))
        logger.info(
            "%s public key Ppub : %s", self.name, utils.short_hx(crypto.point_to_bytes(self.Ppub))
        )
        logger.info("%s initialization complete.", self.name)

    @property
    def is_initialized(self) -> bool:
        """Return True once :meth:`initialize` has successfully run."""
        return self._initialized

    def _require_initialized(self) -> None:
        """
        Internal guard used by later phases (enrollment, authentication) to
        ensure the system parameters exist before they run.
        """
        if not self._initialized:
            raise RuntimeError(
                f"{self.name} is not initialized — call initialize() first."
            )

    # ------------------------------------------------------------------ #
    # 2. DRONE ENROLLMENT PROCEDURE                                       #
    # ------------------------------------------------------------------ #
    def enroll_drone(
        self,
        id_d: str,
        v: bytes,
        challenge: bytes,
        puf_response: bytes,
        device_secret: bytes,
    ) -> database.DroneRecord:
        """
        Drone Enrollment Procedure of SeBAC-IoD (Control Server side).

        This runs over a secure channel during deployment.  The drone has
        already performed its own side:

            1. selects its identity  IDd  and a random value  v
            2. generates a PUF challenge  Cj
            3. computes  Rest = PUF(Cj)
            4. sends  { IDd, v, Cj, Rest }  to the Control Server

        (``device_secret`` is provisioned to the CS at manufacture so the CS
        can later re-derive PUF responses; it is NOT part of the public
        request and never leaves the secure enrollment channel.)

        The Control Server then computes and stores the per-drone material.
        The equations below are the canonical SeBAC-IoD enrollment relations
        that the Drone and User classes must mirror exactly:

            DIDj = H( IDd || v || s )                 # drone pseudo-identity
            Ksd  = H( IDd || s  || Rest )             # drone<->server secret
            kj   = H( DIDj || s ) mod n               # drone private scalar
            Kj   = kj . Q                             # drone public EC point
            Ej   = H( Ksd || Cj ) XOR Rest            # masked PUF response
            Wj   = H( DIDj || Ksd || Rest )           # verifier term
            Nj   = H( IDd || Rest || s )              # stored verifier

        Parameters
        ----------
        id_d:
            The drone's real identity  IDd.
        v:
            The random value  v  chosen by the drone.
        challenge:
            The PUF challenge  Cj  generated by the drone.
        puf_response:
            The PUF response  Rest = PUF(Cj)  produced by the drone.
        device_secret:
            The drone's PUF secret, provisioned to the CS for this simulation
            so it can verify/re-derive responses during authentication.

        Returns
        -------
        database.DroneRecord
            The stored registration record (also persisted in ``self.db``).

        Raises
        ------
        RuntimeError
            If the server has not been initialized.
        """
        self._require_initialized()
        utils.banner(logger, f"PHASE 2 - DRONE ENROLLMENT  (IDd={id_d})")

        # --- Received request fields (logged for traceability) -------------
        logger.info("CS received enrollment request mu0 = { IDd, v, Cj, Rest }")
        logger.info("  IDd  = %s", id_d)
        logger.info("  v    = %s", utils.short_hx(v))
        logger.info("  Cj   = %s", utils.short_hx(challenge))
        logger.info("  Rest = %s", utils.short_hx(puf_response))

        # --- DIDj : drone pseudo-identity ----------------------------------
        # Binds the real identity, the drone nonce v, and the master secret s,
        # so only the CS could have produced it and IDd never travels in clear.
        did_j = crypto.sha256(id_d, v, self._s)

        # --- Ksd : long-term shared secret between this drone and the CS ----
        # Derived from IDd, the master key s, and the PUF response Rest, so it
        # is bound to the drone's physical fingerprint.
        ksd = crypto.sha256(id_d, self._s, puf_response)

        # --- (kj, Kj) : the drone's ECC key pair  with  Kj = kj . Q --------
        # The private scalar kj is derived deterministically from DIDj and s so
        # the CS can recompute it; Kj is the corresponding public point.
        kj_priv = crypto.hash_to_int(did_j, self._s)        # kj in Z_n
        kj_point = crypto.scalar_mult(kj_priv)              # Kj = kj . Q
        kj_pub = crypto.point_to_bytes(kj_point)            # serialized Kj

        # --- Ej : masked PUF response --------------------------------------
        # H(Ksd || Cj) is a 32-byte mask; XOR-ing it with Rest hides the PUF
        # response while letting a party who knows Ksd and Cj recover Rest.
        ej_mask = crypto.sha256(ksd, challenge)             # 32-byte mask
        ej = utils.xor_bytes(ej_mask, utils.fit(puf_response, 32))

        # --- Wj : verifier term --------------------------------------------
        # Lets the CS later confirm a drone holds the correct (Ksd, Rest).
        wj = crypto.sha256(did_j, ksd, puf_response)

        # --- Nj : stored verifier ------------------------------------------
        # An additional binding of IDd, Rest and the master key s.
        nj = crypto.sha256(id_d, puf_response, self._s)

        # --- Persist: DB record + verifier-side CRP ------------------------
        record = database.DroneRecord(
            id_d=id_d,
            did_j=did_j,
            ksd=ksd,
            kj_pub=kj_pub,
            kj_priv=kj_priv,
            ej=ej,
            wj=wj,
            nj=nj,
            challenge=challenge,
            puf_response=puf_response,
            device_secret=device_secret,
        )
        self.db.add_drone(record)
        # Keep the Challenge-Response Pair so the CS can verify the PUF later.
        self.crp_store.enroll(id_d, challenge, puf_response)

        # --- Print every computed value ------------------------------------
        logger.info("CS computed and stored drone credentials:")
        logger.info("  DIDj = %s", utils.short_hx(did_j))
        logger.info("  Ksd  = %s", utils.short_hx(ksd))
        logger.info("  kj   = %s (private scalar)", utils.short_hx(kj_priv))
        logger.info("  Kj   = %s (= kj . Q)", utils.short_hx(kj_pub))
        logger.info("  Ej   = %s", utils.short_hx(ej))
        logger.info("  Wj   = %s", utils.short_hx(wj))
        logger.info("  Nj   = %s", utils.short_hx(nj))
        logger.info("Drone %s enrolled successfully.", id_d)

        return record

    def issue_drone_credentials(self, record: database.DroneRecord) -> dict:
        """
        Return the subset of credentials the Control Server provisions back to
        the drone to store on board after enrollment.

        In SeBAC-IoD the drone keeps the values it needs to participate in the
        authentication phase (its pseudo-identity, masked credential and key
        material).  The master secret ``s`` is never shared.

        Parameters
        ----------
        record:
            The :class:`database.DroneRecord` produced by :meth:`enroll_drone`.

        Returns
        -------
        dict
            ``{ "DIDj", "Ksd", "Kj", "kj", "Ej", "Wj", "Nj" }`` for the drone.
        """
        return {
            "DIDj": record.did_j,
            "Ksd": record.ksd,
            "Kj": record.kj_pub,
            "kj": record.kj_priv,
            "Ej": record.ej,
            "Wj": record.wj,
            "Nj": record.nj,
        }

    # ------------------------------------------------------------------ #
    # 3. USER ENROLLMENT PROCEDURE                                        #
    # ------------------------------------------------------------------ #
    def enroll_user(self, id_u: str, uid: bytes) -> dict:
        """
        User Enrollment Procedure of SeBAC-IoD (Control Server side).

        This runs over a secure channel.  The user has already performed its
        own side (see ``User.begin_enrollment``):

            1. selects a real identity  IDu
            2. generates a random value  r
            3. computes a commitment    UID = H( IDu || r )

        and sends  { IDu, UID }  to the Control Server.

        IMPORTANT — biometric replacement
        ---------------------------------
        The original paper binds a biometric ``BIOu`` (via a fuzzy extractor
        Gen()/Rep()) into the user's stored credentials.  Per the project
        brief, the biometric is replaced on the USER side by a password hash
        ``H(password || salt)``.  The Control Server's job here is unchanged in
        spirit: it issues the anonymous identities and a server-side credential
        ``Bu`` that the user later combines with its password hash to form the
        locally stored verifiers (gamma_u, delta_u, Au, Du).  No biometric (and
        no password) ever reaches the server.

        Control-Server computations
        ---------------------------
            FIDs = H( IDu || s || r_s )              # fake/anonymous identity
            PIDu = H( UID || FIDs || s )             # user pseudo-identity
            Bu   = H( UID || s ) XOR PIDu            # credential issued to user

        Parameters
        ----------
        id_u:
            The user's real identity  IDu.
        uid:
            The user's commitment  UID = H(IDu || r)  (the value r stays with
            the user; the server never sees it).

        Returns
        -------
        dict
            ``{ "PIDu", "FIDs", "Bu" }`` — the material the CS returns to the
            user so it can complete enrollment locally.

        Raises
        ------
        RuntimeError
            If the server has not been initialized.
        """
        self._require_initialized()
        utils.banner(logger, f"PHASE 3 - USER ENROLLMENT  (IDu={id_u})")

        logger.info("CS received enrollment request = { IDu, UID }")
        logger.info("  IDu = %s", id_u)
        logger.info("  UID = %s", utils.short_hx(uid))

        # --- FIDs : fake / anonymous identity ------------------------------
        # A fresh server nonce r_s makes FIDs unlinkable across re-enrollments.
        r_s = crypto.gen_nonce()
        fid_s = crypto.sha256(id_u, self._s, r_s)

        # --- PIDu : user pseudo-identity -----------------------------------
        # Binds the user commitment, the anonymous identity and the master key;
        # only the CS can produce it and IDu never appears in cleartext.
        pid_u = crypto.sha256(uid, fid_s, self._s)

        # --- Bu : credential delivered to the user -------------------------
        # H(UID || s) is known only to the CS (it depends on s); XOR-masking
        # PIDu with it lets the user recover PIDu while binding it to UID.
        bu_mask = crypto.sha256(uid, self._s)              # 32-byte mask
        bu = utils.xor_bytes(bu_mask, utils.fit(pid_u, 32))

        # --- Persist the server-side user record ---------------------------
        record = database.UserRecord(
            id_u=id_u,
            uid=uid,
            pid_u=pid_u,
            fid_s=fid_s,
            bu=bu,
        )
        self.db.add_user(record)

        # --- Print every intermediate computation --------------------------
        logger.info("CS computed and stored user credentials:")
        logger.info("  FIDs = %s", utils.short_hx(fid_s))
        logger.info("  PIDu = %s", utils.short_hx(pid_u))
        logger.info("  Bu   = %s (issued to user)", utils.short_hx(bu))
        logger.info("User %s enrolled successfully.", id_u)

        return {"PIDu": pid_u, "FIDs": fid_s, "Bu": bu}

    # ------------------------------------------------------------------ #
    # 4. BATCH AUTHENTICATION (Control Server side)                       #
    # ------------------------------------------------------------------ #
    def _recover_sigma_u(self, record: database.UserRecord) -> bytes:
        """
        Recompute the user<->CS shared secret  sigma_u = H(UID || s).

        The CS holds the master key s and the user's UID, so it derives the same
        sigma_u the user reconstructed from Bu during enrollment.
        """
        return crypto.sha256(record.uid, self._s)

    def verify_mu1(self, mu1: "Mu1Like") -> database.UserRecord:
        """
        Authenticate the user's session-opening message mu1.

        Checks
        ------
        1. Freshness of T1 (replay protection).
        2. Resolve PIDu -> UserRecord.
        3. Recover sigma_u = H(UID || s); unmask UID' = M2 XOR H(sigma_u||PIDu||T1)
           and confirm it equals the stored UID.
        4. Recompute the authenticator M3 = H(PIDu||UID||M1||T1||sigma_u) and
           compare with the received M3.

        Parameters
        ----------
        mu1:
            The message object with attributes ``pid_u, m1, m2, m3, t1``.

        Returns
        -------
        database.UserRecord
            The authenticated user's record.

        Raises
        ------
        ValueError
            If mu1 is stale, the user is unknown, or a verifier mismatches.
        """
        self._require_initialized()
        utils.banner(logger, "PHASE 4 - BATCH AUTHENTICATION (mu1 received)")

        if not utils.is_fresh(mu1.t1):
            raise ValueError("CS: mu1 timestamp T1 not fresh (replay?)")

        record = self.db.get_user_by_pid(mu1.pid_u)
        if record is None:
            raise ValueError("CS: unknown PIDu in mu1")

        sigma_u = self._recover_sigma_u(record)

        # Unmask and check UID.
        mask = crypto.sha256(sigma_u, mu1.pid_u, mu1.t1)
        uid_prime = utils.xor_bytes(mu1.m2, utils.fit(mask, 32))
        if uid_prime != utils.fit(record.uid, 32):
            raise ValueError("CS: UID check failed in mu1")

        # Recompute and check the authenticator M3.
        m3_expected = crypto.sha256(mu1.pid_u, record.uid, mu1.m1, mu1.t1, sigma_u)
        if not puf._ct_equal(m3_expected, mu1.m3):
            raise ValueError("CS: M3 authenticator mismatch in mu1")

        logger.info("CS authenticated user %s (PIDu=%s)",
                    record.id_u, utils.short_hx(mu1.pid_u))
        return record

    def build_mu2_batch(
        self, mu1: "Mu1Like", drone_ids: Optional[List[str]] = None
    ) -> List[Mu2]:
        """
        After authenticating mu1, build one mu2 per drone in the batch.

        For each drone the CS computes, using that drone's stored Ksd/DIDj:

            Yj = H( Ksd || DIDj || PIDu || X || T2 )      # CS authenticator
            M5 = H( sigma_u || PIDu || DIDj || T2 )       # transcript binder
            G2 = X   (relayed user point)                  G3 = PIDu

        A single fresh timestamp T2 is shared across the batch.

        Parameters
        ----------
        mu1:
            The authenticated session-opening message (provides PIDu, M1=X, ...).
        drone_ids:
            Optional explicit list of drone IDd to include; defaults to ALL
            enrolled drones (the full batch).

        Returns
        -------
        List[Mu2]
            The per-drone messages, in the same order as the batch.

        Raises
        ------
        ValueError
            If mu1 fails authentication or a requested drone is unknown.
        """
        record = self.verify_mu1(mu1)
        sigma_u = self._recover_sigma_u(record)

        # Assemble the batch of drone records.
        if drone_ids is None:
            drones = self.db.all_drones()
        else:
            drones = []
            for did in drone_ids:
                dr = self.db.get_drone(did)
                if dr is None:
                    raise ValueError(f"CS: unknown drone {did} requested for batch")
                drones.append(dr)

        t2 = utils.now_ts()
        x_point_bytes = mu1.m1           # G2 = X
        pid_u = mu1.pid_u                # G3 = PIDu

        messages: List[Mu2] = []
        for dr in drones:
            yj = crypto.sha256(dr.ksd, dr.did_j, pid_u, x_point_bytes, t2)
            m5 = crypto.sha256(sigma_u, pid_u, dr.did_j, t2)
            messages.append(Mu2(yj=yj, m5=m5, g2=x_point_bytes, g3=pid_u, t2=t2))

        logger.info("CS built mu2 for a batch of %d drone(s) (T2=%d).", len(messages), t2)
        return messages


# Structural alias: mu1 only needs attributes pid_u, m1, m2, m3, t1.
Mu1Like = object
