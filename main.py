"""
main.py
=======

End-to-end driver for the SeBAC-IoD simulation.

Running ``python main.py`` executes the complete protocol and prints every
generated value:

    Phase 1  Initialization        : curve, Q, n, p, master key s, Ppub
    Phase 2  Drone Enrollment       : DIDj, Ksd, Kj, Ej, Wj, Nj per drone
    Phase 3  User Enrollment         : UID, FIDs, PIDu, gamma_u/delta_u/Au/Du
    Phase 4  Batch Authentication    : mu1 -> mu2 -> mu3, batch verify, SKij

It also demonstrates the divide-and-conquer fault isolation by tampering with
one drone's response in a second verification pass, and writes the ASCII
architecture/sequence diagrams under ``diagrams/``.
"""

from __future__ import annotations

import os
from typing import List

import control_server as csmod
import crypto
import drone as dmod
import user as umod
import utils

logger = utils.get_logger(__name__)

NUM_DRONES = 5
USER_ID = "User-01"
USER_PASSWORD = "S3cure-Pass!"   # biometric replacement input


# ---------------------------------------------------------------------------
# Diagram writers
# ---------------------------------------------------------------------------
def write_diagrams() -> None:
    """Write the architecture and sequence diagrams to diagrams/*.txt."""
    here = os.path.dirname(os.path.abspath(__file__))
    ddir = os.path.join(here, "diagrams")
    os.makedirs(ddir, exist_ok=True)

    architecture = r"""
SeBAC-IoD — System Architecture
===============================

                         +------------------------+
                         |     Control Server     |
                         |  (Registration / KGC)  |
                         |  master key s, Ppub=sQ |
                         +-----------+------------+
                                     |
              secure enrollment +    | publishes { E(Fp), Q, n, p, Ppub, H() }
              batch authentication   |
                 +-------------------+-------------------+
                 |                   |                   |
            +----+----+         +----+----+         +----+----+
            | Drone D1 |        | Drone D2 |        | Drone Dn |
            |  PUF, Kj |        |  PUF, Kj |        |  PUF, Kj |
            +----+----+         +----+----+         +----+----+
                 \                   |                   /
                  \                  |                  /
                   \                 |                 /
                    +---------------------------------+
                    |               User              |
                    |  password-hash (no biometric)   |
                    |  batch-verifies all drones at    |
                    |  once; derives SKij per drone    |
                    +---------------------------------+
"""

    sequence = r"""
SeBAC-IoD — Batch Authentication Sequence
=========================================

  User                     Control Server                 Drones (batch)
   |                              |                              |
   |  Enroll {IDu, UID}           |                              |
   |----------------------------->|                              |
   |  {PIDu, FIDs, Bu}            |                              |
   |<-----------------------------|                              |
   |                              |   Enroll {IDd,v,Cj,Rest}     |
   |                              |<-----------------------------|
   |                              |  {DIDj,Ksd,Kj,Ej,Wj,Nj}     |
   |                              |----------------------------->|
   |                              |                              |
   |  mu1 {PIDu,M1,M2,M3,T1}      |                              |
   |============================>>|                              |
   |                              |  mu2 {Yj,M5,G2,G3,T2}        |
   |                              |============================>>|
   |                              |                              |
   |              mu3 {M5,DIDj,Ij,Kj}  (one per drone)           |
   |<<===========================================================|
   |                              |                              |
   |  Batch verify (random Vecj): (Sum dj sj).Q == Sum dj Rj +   |
   |                               Sum (dj ej) Kj                 |
   |  on success -> SKij per drone ; on failure -> divide&conquer |
   |                              |                              |
"""

    with open(os.path.join(ddir, "architecture.txt"), "w", encoding="utf-8") as fh:
        fh.write(architecture.lstrip("\n"))
    with open(os.path.join(ddir, "sequence_diagram.txt"), "w", encoding="utf-8") as fh:
        fh.write(sequence.lstrip("\n"))
    logger.info("Diagrams written to diagrams/architecture.txt and sequence_diagram.txt")


# ---------------------------------------------------------------------------
# Protocol driver
# ---------------------------------------------------------------------------
def run() -> None:
    """Execute the full SeBAC-IoD protocol once and report results."""
    utils.banner(logger, "SeBAC-IoD SIMULATION START")

    # ---- Phase 1: Initialization --------------------------------------
    cs = csmod.ControlServer()
    cs.initialize()

    # ---- Phase 3 (setup): enroll the user -----------------------------
    user = umod.User(USER_ID, USER_PASSWORD)
    id_u, uid = user.begin_enrollment()
    user.complete_enrollment(cs.enroll_user(id_u, uid))

    # ---- Phase 2 (setup): enroll the drones ---------------------------
    drones: List[dmod.Drone] = []
    for i in range(NUM_DRONES):
        d = dmod.Drone(f"Drone-{i:02d}")
        req = d.begin_enrollment()
        record = cs.enroll_drone(
            req.id_d, req.v, req.challenge, req.puf_response, d.device_secret
        )
        d.store_credentials(cs.issue_drone_credentials(record))
        drones.append(d)

    logger.info("Setup complete: %s", cs.db.summary())

    # ---- Phase 4: Batch Authentication --------------------------------
    mu1 = user.build_mu1()
    mu2_list = cs.build_mu2_batch(mu1)                  # CS verifies mu1 inside
    mu3_batch = [d.handle_mu2(m) for d, m in zip(drones, mu2_list)]

    result = user.batch_verify(mu3_batch)
    sessions = user.establish_sessions(mu3_batch, result)

    # ---- Report session keys ------------------------------------------
    utils.banner(logger, "SESSION KEYS (SKij)")
    for d in drones:
        did_hex = d.credentials["DIDj"].hex()  # type: ignore[union-attr]
        sk = sessions.get(did_hex)
        if sk is not None:
            logger.info("  %s  ->  SKij = %s", d.id_d, utils.short_hx(sk))
            # Cross-check against the drone's own derivation.
            assert sk == d.derive_session_key(), "session key disagreement!"

    # ---- Demonstrate divide-and-conquer fault isolation ---------------
    utils.banner(logger, "FAULT-ISOLATION DEMO (one tampered drone)")
    victim = len(mu3_batch) // 2
    Rj_b, sj = mu3_batch[victim].Ij
    mu3_batch[victim].Ij = (Rj_b, (sj + 1) % crypto.ORDER_N)   # corrupt sj
    logger.info("Tampered with response of drone index %d (DIDj=%s)",
                victim, utils.short_hx(mu3_batch[victim].did_j))
    result2 = user.batch_verify(mu3_batch)
    logger.info("Result: all_valid=%s, invalid=%d, aggregate-checks=%d",
                result2.all_valid, len(result2.invalid_drones), result2.checks)

    # ---- Diagrams + summary -------------------------------------------
    write_diagrams()
    utils.banner(logger, "SeBAC-IoD SIMULATION COMPLETE")
    logger.info("Honest-batch verification used %d aggregate check for %d drones.",
                result.checks, NUM_DRONES)
    logger.info("Tampered-batch isolation used %d aggregate checks to find %d culprit(s).",
                result2.checks, len(result2.invalid_drones))


if __name__ == "__main__":
    run()
