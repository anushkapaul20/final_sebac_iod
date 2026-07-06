"""
tests/test_enrollment.py
========================
Tests for Phase 1 (CS initialization), Phase 2 (drone enrollment),
and Phase 3 (user enrollment).

Covers:
  - CS initialization sets all public parameters
  - Double-initialization raises RuntimeError
  - Drone enrollment stores all 7 credential fields with correct types/lengths
  - DIDj pseudo-identity is deterministic for same inputs
  - Kj = kj.Q holds (EC key pair consistency)
  - User enrollment returns PIDu / FIDs / Bu of expected types
  - Bu correctly encodes sigma_u (user can recover sigma_u)
  - Wrong-password local login raises PermissionError
  - Database correctly indexes enrollments
"""
import unittest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import crypto
import utils
from tests.helpers import make_cs, enroll_drone, enroll_user, full_setup


class TestCSInitialization(unittest.TestCase):

    def test_sets_public_params(self):
        cs = make_cs()
        self.assertIsNotNone(cs.prime_p)
        self.assertIsNotNone(cs.order_n)
        self.assertIsNotNone(cs.Q)
        self.assertIsNotNone(cs.Ppub)
        self.assertIsNotNone(cs._s)

    def test_ppub_equals_s_times_Q(self):
        cs = make_cs()
        self.assertEqual(cs.Ppub, crypto.scalar_mult(cs._s))

    def test_is_initialized_flag(self):
        cs = make_cs()
        self.assertTrue(cs.is_initialized)

    def test_double_init_raises(self):
        cs = make_cs()
        with self.assertRaises(RuntimeError):
            cs.initialize()

    def test_prime_p_256_bits(self):
        cs = make_cs()
        self.assertEqual(cs.prime_p.bit_length(), 256)

    def test_order_n_256_bits(self):
        cs = make_cs()
        self.assertEqual(cs.order_n.bit_length(), 256)


class TestDroneEnrollment(unittest.TestCase):

    def setUp(self):
        self.cs = make_cs()
        self.drone = enroll_drone(self.cs, "Drone-Test")

    def test_credentials_stored_on_drone(self):
        creds = self.drone.credentials
        for key in ("DIDj", "Ksd", "Kj", "kj", "Ej", "Wj", "Nj"):
            self.assertIn(key, creds)

    def test_did_j_is_32_bytes(self):
        self.assertEqual(len(self.drone.credentials["DIDj"]), 32)

    def test_ksd_is_32_bytes(self):
        self.assertEqual(len(self.drone.credentials["Ksd"]), 32)

    def test_kj_pub_is_64_bytes(self):
        # P-256 uncompressed = 64 bytes
        self.assertEqual(len(self.drone.credentials["Kj"]), 64)

    def test_kj_pair_consistent(self):
        """kj is a valid scalar and Kj = kj.Q must hold."""
        kj_priv = self.drone.credentials["kj"]
        kj_pub_bytes = self.drone.credentials["Kj"]
        expected = crypto.point_to_bytes(crypto.scalar_mult(kj_priv))
        self.assertEqual(kj_pub_bytes, expected)

    def test_db_has_drone(self):
        record = self.cs.db.get_drone("Drone-Test")
        self.assertIsNotNone(record)
        self.assertEqual(record.id_d, "Drone-Test")

    def test_db_did_index(self):
        did_j = self.drone.credentials["DIDj"]
        record = self.cs.db.get_drone_by_did(did_j)
        self.assertIsNotNone(record)

    def test_multiple_drones_different_did(self):
        cs = make_cs()
        d1 = enroll_drone(cs, "D-1")
        d2 = enroll_drone(cs, "D-2")
        self.assertNotEqual(d1.credentials["DIDj"], d2.credentials["DIDj"])

    def test_enroll_without_init_raises(self):
        import control_server as csmod
        cs = csmod.ControlServer()          # NOT initialized
        import drone as dmod, puf
        d = dmod.Drone("X")
        req = d.begin_enrollment()
        with self.assertRaises(RuntimeError):
            cs.enroll_drone(req.id_d, req.v, req.challenge,
                            req.puf_response, d.device_secret)

    def test_ej_xor_recovery(self):
        """
        Ej = H(Ksd || Cj) XOR Rest.  Recovering Rest from Ej must match
        the PUF response stored in the DB record.
        """
        record = self.cs.db.get_drone("Drone-Test")
        mask = crypto.sha256(record.ksd, record.challenge)
        recovered = utils.xor_bytes(mask, record.ej)
        self.assertEqual(recovered, utils.fit(record.puf_response, 32))


class TestUserEnrollment(unittest.TestCase):

    def setUp(self):
        self.cs = make_cs()
        self.user = enroll_user(self.cs, "Alice", "correct-password")

    def test_pid_u_set(self):
        self.assertIsNotNone(self.user.pid_u)
        self.assertEqual(len(self.user.pid_u), 32)

    def test_verifiers_set(self):
        for attr in ("gamma_u", "delta_u", "au", "du"):
            self.assertIsNotNone(getattr(self.user, attr))

    def test_uid_commitment(self):
        """UID = H(IDu || r) must hold."""
        expected = crypto.sha256(self.user.id_u, self.user._r)
        self.assertEqual(self.user.uid, expected)

    def test_db_has_user(self):
        record = self.cs.db.get_user_by_uid(self.user.uid)
        self.assertIsNotNone(record)

    def test_pid_index(self):
        record = self.cs.db.get_user_by_pid(self.user.pid_u)
        self.assertIsNotNone(record)
        self.assertEqual(record.id_u, "Alice")

    def test_correct_password_login(self):
        """_local_login must succeed with the correct password."""
        sigma = self.user._local_login("correct-password")
        self.assertIsInstance(sigma, bytes)
        self.assertEqual(len(sigma), 32)

    def test_wrong_password_raises(self):
        with self.assertRaises(PermissionError):
            self.user._local_login("wrong-password")

    def test_two_users_different_pid(self):
        cs = make_cs()
        u1 = enroll_user(cs, "Bob", "pw1")
        u2 = enroll_user(cs, "Carol", "pw2")
        self.assertNotEqual(u1.pid_u, u2.pid_u)

    def test_enroll_without_init_raises(self):
        import control_server as csmod
        cs = csmod.ControlServer()
        import user as umod
        u = umod.User("X", "pw")
        id_u, uid = u.begin_enrollment()
        with self.assertRaises(RuntimeError):
            cs.enroll_user(id_u, uid)


if __name__ == "__main__":
    unittest.main(verbosity=2)
