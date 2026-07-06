"""
tests/test_authentication.py
=============================
Tests for Phase 4 — batch authentication, batch verification,
session-key agreement, and divide-and-conquer fault isolation.

Covers:
  - mu1 builds correct fields / password check
  - CS accepts a valid mu1 and rejects a tampered one
  - mu2 builds and drone validates Yj
  - mu3 Schnorr signature verifies individually
  - Batch of N honest drones passes in ONE aggregate check
  - User and drone derive identical session keys (ECDH consistency)
  - Tampered sj caught by aggregate check
  - Divide-and-conquer locates exactly the tampered drone(s)
  - Multiple tampered drones all isolated
  - Replay attack (stale T1) rejected by CS
  - Unknown PIDu rejected by CS
  - Drone with no credentials raises ValueError
  - Empty batch passes with all_valid=True
"""
import copy
import unittest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import crypto
import utils
from tests.helpers import full_setup, run_auth, make_cs, enroll_drone, enroll_user


class TestMu1Building(unittest.TestCase):

    def setUp(self):
        self.cs, self.user, self.drones = full_setup(3)

    def test_mu1_fields_present(self):
        mu1 = self.user.build_mu1()
        self.assertIsNotNone(mu1.pid_u)
        self.assertIsNotNone(mu1.m1)
        self.assertIsNotNone(mu1.m2)
        self.assertIsNotNone(mu1.m3)
        self.assertIsNotNone(mu1.t1)

    def test_m1_is_valid_ec_point(self):
        """M1 = X = x.Q; should deserialize to a valid EC point."""
        mu1 = self.user.build_mu1()
        P = crypto.bytes_to_point(mu1.m1)
        self.assertIsNotNone(P)

    def test_pid_u_matches_enrolled(self):
        mu1 = self.user.build_mu1()
        self.assertEqual(mu1.pid_u, self.user.pid_u)

    def test_t1_is_fresh(self):
        mu1 = self.user.build_mu1()
        self.assertTrue(utils.is_fresh(mu1.t1))

    def test_wrong_password_blocks_mu1(self):
        from user import User
        u = User("Bob", "pw")
        id_u, uid = u.begin_enrollment()
        u.complete_enrollment(self.cs.enroll_user(id_u, uid))
        with self.assertRaises(PermissionError):
            u.build_mu1(password="wrong")


class TestCSVerifyMu1(unittest.TestCase):

    def setUp(self):
        self.cs, self.user, self.drones = full_setup(2)
        self.mu1 = self.user.build_mu1()

    def test_valid_mu1_accepted(self):
        record = self.cs.verify_mu1(self.mu1)
        self.assertEqual(record.id_u, self.user.id_u)

    def test_tampered_m3_rejected(self):
        self.mu1.m3 = bytes(b ^ 0xFF for b in self.mu1.m3)
        with self.assertRaises(ValueError):
            self.cs.verify_mu1(self.mu1)

    def test_tampered_m2_rejected(self):
        self.mu1.m2 = bytes(b ^ 0xFF for b in self.mu1.m2)
        with self.assertRaises(ValueError):
            self.cs.verify_mu1(self.mu1)

    def test_unknown_pid_rejected(self):
        self.mu1.pid_u = crypto.gen_nonce(32)  # random, not in DB
        with self.assertRaises(ValueError):
            self.cs.verify_mu1(self.mu1)

    def test_stale_timestamp_rejected(self):
        self.mu1.t1 = utils.now_ts() - 200   # 200 s in the past -> stale
        with self.assertRaises(ValueError):
            self.cs.verify_mu1(self.mu1)

    def test_future_timestamp_rejected(self):
        self.mu1.t1 = utils.now_ts() + 200   # 200 s in the future
        with self.assertRaises(ValueError):
            self.cs.verify_mu1(self.mu1)


class TestMu2Building(unittest.TestCase):

    def setUp(self):
        self.cs, self.user, self.drones = full_setup(3)

    def test_mu2_count_matches_drones(self):
        mu1 = self.user.build_mu1()
        mu2_list = self.cs.build_mu2_batch(mu1)
        self.assertEqual(len(mu2_list), len(self.drones))

    def test_mu2_fields_present(self):
        mu1 = self.user.build_mu1()
        for mu2 in self.cs.build_mu2_batch(mu1):
            self.assertIsNotNone(mu2.yj)
            self.assertIsNotNone(mu2.m5)
            self.assertIsNotNone(mu2.g2)
            self.assertIsNotNone(mu2.g3)
            self.assertIsNotNone(mu2.t2)

    def test_g3_is_pid_u(self):
        mu1 = self.user.build_mu1()
        for mu2 in self.cs.build_mu2_batch(mu1):
            self.assertEqual(mu2.g3, self.user.pid_u)


class TestDroneMu3(unittest.TestCase):

    def setUp(self):
        self.cs, self.user, self.drones = full_setup(3)

    def test_all_drones_produce_mu3(self):
        mu1 = self.user.build_mu1()
        mu2_list = self.cs.build_mu2_batch(mu1)
        for d, mu2 in zip(self.drones, mu2_list):
            mu3 = d.handle_mu2(mu2)
            self.assertIsNotNone(mu3)

    def test_mu3_did_matches_credentials(self):
        mu1 = self.user.build_mu1()
        mu2_list = self.cs.build_mu2_batch(mu1)
        for d, mu2 in zip(self.drones, mu2_list):
            mu3 = d.handle_mu2(mu2)
            self.assertEqual(mu3.did_j, d.credentials["DIDj"])

    def test_tampered_yj_rejected_by_drone(self):
        mu1 = self.user.build_mu1()
        mu2_list = self.cs.build_mu2_batch(mu1)
        mu2_bad = copy.copy(mu2_list[0])
        mu2_bad.yj = bytes(b ^ 0xFF for b in mu2_bad.yj)
        with self.assertRaises(ValueError):
            self.drones[0].handle_mu2(mu2_bad)

    def test_stale_mu2_rejected(self):
        mu1 = self.user.build_mu1()
        mu2_list = self.cs.build_mu2_batch(mu1)
        mu2_bad = copy.copy(mu2_list[0])
        mu2_bad.t2 = utils.now_ts() - 200
        with self.assertRaises(ValueError):
            self.drones[0].handle_mu2(mu2_bad)

    def test_drone_without_credentials_raises(self):
        import drone as dmod
        d = dmod.Drone("no-creds")
        mu1 = self.user.build_mu1()
        mu2 = self.cs.build_mu2_batch(mu1)[0]
        with self.assertRaises(ValueError):
            d.handle_mu2(mu2)

    def test_individual_schnorr_eq(self):
        """sj.Q == Rj + ej.Kj must hold for each honest drone."""
        mu1 = self.user.build_mu1()
        mu2_list = self.cs.build_mu2_batch(mu1)
        for d, mu2 in zip(self.drones, mu2_list):
            mu3 = d.handle_mu2(mu2)
            self.assertTrue(self.user._verify_single(mu3))


class TestBatchVerification(unittest.TestCase):

    def setUp(self):
        self.cs, self.user, self.drones = full_setup(5)
        self.mu3_batch, self.result, self.sessions = run_auth(
            self.cs, self.user, self.drones
        )

    def test_all_valid(self):
        self.assertTrue(self.result.all_valid)

    def test_valid_count(self):
        self.assertEqual(len(self.result.valid_drones), 5)

    def test_invalid_count_zero(self):
        self.assertEqual(len(self.result.invalid_drones), 0)

    def test_single_aggregate_check(self):
        """Honest batch should pass in exactly 1 EC check."""
        self.assertEqual(self.result.checks, 1)

    def test_empty_batch(self):
        result = self.user.batch_verify([])
        self.assertTrue(result.all_valid)
        self.assertEqual(result.valid_drones, [])

    def test_single_drone_batch(self):
        cs, user, drones = full_setup(1)
        _, result, _ = run_auth(cs, user, drones)
        self.assertTrue(result.all_valid)
        self.assertEqual(len(result.valid_drones), 1)


class TestSessionKeyAgreement(unittest.TestCase):

    def setUp(self):
        self.cs, self.user, self.drones = full_setup(4)
        self.mu3_batch, self.result, self.sessions = run_auth(
            self.cs, self.user, self.drones
        )

    def test_session_keys_exist_for_all_drones(self):
        self.assertEqual(len(self.sessions), 4)

    def test_session_key_length(self):
        for sk in self.sessions.values():
            self.assertEqual(len(sk), 32)

    def test_user_drone_key_agreement(self):
        """User's SKij must equal drone's SKij (ECDH correctness)."""
        for d in self.drones:
            did_hex = d.credentials["DIDj"].hex()
            user_sk = self.sessions[did_hex]
            drone_sk = d.derive_session_key()
            self.assertEqual(user_sk, drone_sk,
                             f"Key mismatch for drone {d.id_d}")

    def test_session_keys_distinct_per_drone(self):
        """Each drone gets a different session key."""
        keys = list(self.sessions.values())
        self.assertEqual(len(keys), len(set(keys)))


class TestFaultIsolation(unittest.TestCase):

    def _setup_and_tamper(self, n_drones: int, victim_indices):
        cs, user, drones = full_setup(n_drones)
        mu1 = user.build_mu1()
        mu2_list = cs.build_mu2_batch(mu1)
        mu3_batch = [d.handle_mu2(m) for d, m in zip(drones, mu2_list)]

        tampered_dids = set()
        for idx in victim_indices:
            Rj_b, sj = mu3_batch[idx].Ij
            mu3_batch[idx].Ij = (Rj_b, (sj + 1) % crypto.ORDER_N)
            tampered_dids.add(mu3_batch[idx].did_j)

        result = user.batch_verify(mu3_batch)
        return result, tampered_dids

    def test_one_tampered_drone_detected(self):
        result, tampered = self._setup_and_tamper(5, [2])
        self.assertFalse(result.all_valid)
        self.assertEqual(set(result.invalid_drones), tampered)
        self.assertEqual(len(result.valid_drones), 4)

    def test_two_tampered_drones_detected(self):
        result, tampered = self._setup_and_tamper(6, [1, 4])
        self.assertFalse(result.all_valid)
        self.assertEqual(set(result.invalid_drones), tampered)
        self.assertEqual(len(result.valid_drones), 4)

    def test_all_tampered(self):
        result, tampered = self._setup_and_tamper(3, [0, 1, 2])
        self.assertFalse(result.all_valid)
        self.assertEqual(set(result.invalid_drones), tampered)
        self.assertEqual(len(result.valid_drones), 0)

    def test_first_drone_tampered(self):
        result, tampered = self._setup_and_tamper(4, [0])
        self.assertFalse(result.all_valid)
        self.assertEqual(set(result.invalid_drones), tampered)

    def test_last_drone_tampered(self):
        result, tampered = self._setup_and_tamper(4, [3])
        self.assertFalse(result.all_valid)
        self.assertEqual(set(result.invalid_drones), tampered)

    def test_divide_and_conquer_uses_few_checks(self):
        """
        For k=1 tampered drone out of n=8, D&C uses at most 2*log2(n)+1
        aggregate checks — far fewer than n individual checks would require.
        The worst case for n=8 is ~8 checks (initial + recursive splits);
        the key property is that it scales as O(log n), not O(n).
        """
        result, _ = self._setup_and_tamper(16, [5])
        # For n=16, individual verification would need 16 checks.
        # D&C needs at most 2*log2(16)+1 = 9 checks.
        self.assertLessEqual(result.checks, 10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
