"""
tests/helpers.py
================
Shared setup helpers so every test module gets a fully-enrolled CS/User/Drone
without copy-pasting the 20-line setup dance.
"""
from __future__ import annotations
from typing import List, Tuple

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import control_server as csmod
import drone as dmod
import user as umod


def make_cs() -> csmod.ControlServer:
    cs = csmod.ControlServer()
    cs.initialize()
    return cs


def enroll_drone(cs: csmod.ControlServer, id_d: str) -> dmod.Drone:
    d = dmod.Drone(id_d)
    req = d.begin_enrollment()
    record = cs.enroll_drone(req.id_d, req.v, req.challenge, req.puf_response, d.device_secret)
    d.store_credentials(cs.issue_drone_credentials(record))
    return d


def enroll_user(cs: csmod.ControlServer, id_u: str = "User-01",
                password: str = "Test-Pass!") -> umod.User:
    u = umod.User(id_u, password)
    id_u_str, uid = u.begin_enrollment()
    u.complete_enrollment(cs.enroll_user(id_u_str, uid))
    return u


def full_setup(n_drones: int = 3) -> Tuple[csmod.ControlServer, umod.User, List[dmod.Drone]]:
    """Return a fully enrolled (CS, User, [Drone...]) tuple ready for auth."""
    cs = make_cs()
    user = enroll_user(cs)
    drones = [enroll_drone(cs, f"Drone-{i:02d}") for i in range(n_drones)]
    return cs, user, drones


def run_auth(cs: csmod.ControlServer, user: umod.User,
             drones: List[dmod.Drone]):
    """Run the full mu1->mu2->mu3 flow and return (mu3_batch, result, sessions)."""
    mu1 = user.build_mu1()
    mu2_list = cs.build_mu2_batch(mu1)
    mu3_batch = [d.handle_mu2(m) for d, m in zip(drones, mu2_list)]
    result = user.batch_verify(mu3_batch)
    sessions = user.establish_sessions(mu3_batch, result)
    return mu3_batch, result, sessions
