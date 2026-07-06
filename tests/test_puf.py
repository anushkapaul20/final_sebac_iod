"""
tests/test_puf.py
=================
Unit tests for puf.py — the Physical Unclonable Function simulation layer.

Covers:
  - PUF is deterministic (same challenge + secret -> same response)
  - PUF is device-unique (different secrets -> different responses)
  - generate_challenge produces correct-length random bytes
  - CRPStore enroll/lookup round-trip
  - response_verification (secret mode and CRP mode)
  - PUFDevice.evaluate wraps simulate_puf correctly
"""
import unittest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import puf
import crypto


class TestSimulatePUF(unittest.TestCase):

    def _make_pair(self):
        challenge = puf.generate_challenge()
        secret = puf.generate_device_secret()
        return challenge, secret

    def test_deterministic(self):
        c, s = self._make_pair()
        r1 = puf.simulate_puf(c, s)
        r2 = puf.simulate_puf(c, s)
        self.assertEqual(r1, r2)

    def test_device_unique(self):
        """Different device secrets produce different responses for same challenge."""
        c = puf.generate_challenge()
        s1 = puf.generate_device_secret()
        s2 = puf.generate_device_secret()
        self.assertNotEqual(puf.simulate_puf(c, s1), puf.simulate_puf(c, s2))

    def test_challenge_unique(self):
        """Different challenges produce different responses for same device secret."""
        s = puf.generate_device_secret()
        c1 = puf.generate_challenge()
        c2 = puf.generate_challenge()
        self.assertNotEqual(puf.simulate_puf(c1, s), puf.simulate_puf(c2, s))

    def test_output_length(self):
        c, s = self._make_pair()
        self.assertEqual(len(puf.simulate_puf(c, s)), 32)


class TestGenerateChallenge(unittest.TestCase):

    def test_default_length(self):
        c = puf.generate_challenge()
        self.assertEqual(len(c), puf.CHALLENGE_BYTES)

    def test_custom_length(self):
        c = puf.generate_challenge(32)
        self.assertEqual(len(c), 32)

    def test_randomness(self):
        cs = {puf.generate_challenge() for _ in range(10)}
        self.assertGreater(len(cs), 1)


class TestPUFDevice(unittest.TestCase):

    def test_evaluate_deterministic(self):
        device = puf.PUFDevice("test-drone")
        c = puf.generate_challenge()
        self.assertEqual(device.evaluate(c), device.evaluate(c))

    def test_evaluate_matches_simulate(self):
        device = puf.PUFDevice("test-drone")
        c = puf.generate_challenge()
        r_device = device.evaluate(c)
        r_raw = puf.simulate_puf(c, device.device_secret)
        self.assertEqual(r_device, r_raw)

    def test_two_devices_differ(self):
        d1 = puf.PUFDevice("drone-A")
        d2 = puf.PUFDevice("drone-B")
        c = puf.generate_challenge()
        self.assertNotEqual(d1.evaluate(c), d2.evaluate(c))


class TestResponseGeneration(unittest.TestCase):

    def test_matches_device_evaluate(self):
        device = puf.PUFDevice("regen-drone")
        c = puf.generate_challenge()
        self.assertEqual(puf.response_generation(device, c), device.evaluate(c))


class TestCRPStore(unittest.TestCase):

    def setUp(self):
        self.store = puf.CRPStore()
        self.device_id = "drone-crp"
        self.challenge = puf.generate_challenge()
        self.response = puf.simulate_puf(self.challenge, puf.generate_device_secret())

    def test_lookup_after_enroll(self):
        self.store.enroll(self.device_id, self.challenge, self.response)
        r = self.store.lookup(self.device_id, self.challenge)
        self.assertEqual(r, self.response)

    def test_lookup_unknown_device_returns_none(self):
        self.assertIsNone(self.store.lookup("no-such-drone", self.challenge))

    def test_lookup_unknown_challenge_returns_none(self):
        self.store.enroll(self.device_id, self.challenge, self.response)
        other_c = puf.generate_challenge()
        self.assertIsNone(self.store.lookup(self.device_id, other_c))

    def test_multiple_devices(self):
        c2 = puf.generate_challenge()
        r2 = puf.simulate_puf(c2, puf.generate_device_secret())
        self.store.enroll("drone-1", self.challenge, self.response)
        self.store.enroll("drone-2", c2, r2)
        self.assertEqual(self.store.lookup("drone-1", self.challenge), self.response)
        self.assertEqual(self.store.lookup("drone-2", c2), r2)
        self.assertIsNone(self.store.lookup("drone-1", c2))


class TestResponseVerification(unittest.TestCase):

    def setUp(self):
        self.secret = puf.generate_device_secret()
        self.challenge = puf.generate_challenge()
        self.response = puf.simulate_puf(self.challenge, self.secret)

    def test_crp_mode_correct(self):
        self.assertTrue(
            puf.response_verification(self.response, expected_response=self.response)
        )

    def test_crp_mode_wrong(self):
        wrong = bytes(b ^ 0xFF for b in self.response)
        self.assertFalse(
            puf.response_verification(wrong, expected_response=self.response)
        )

    def test_secret_mode_correct(self):
        self.assertTrue(
            puf.response_verification(
                self.response, device_secret=self.secret, challenge=self.challenge
            )
        )

    def test_secret_mode_wrong_challenge(self):
        other_c = puf.generate_challenge()
        self.assertFalse(
            puf.response_verification(
                self.response, device_secret=self.secret, challenge=other_c
            )
        )

    def test_raises_without_args(self):
        with self.assertRaises(ValueError):
            puf.response_verification(self.response)


if __name__ == "__main__":
    unittest.main(verbosity=2)
