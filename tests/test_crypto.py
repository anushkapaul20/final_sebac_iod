"""
tests/test_crypto.py
====================
Unit tests for crypto.py — the cryptographic primitive layer.

Covers:
  - SHA-256 correctness and determinism
  - hash_to_int stays in Z_n
  - gen_scalar / gen_keypair generate valid, non-zero values
  - scalar_mult / point_add basic EC arithmetic
  - point_to_bytes / bytes_to_point round-trip
  - derive_session_key determinism and sensitivity
"""
import unittest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import hashlib
import crypto
from ecdsa.ellipticcurve import INFINITY


class TestSHA256(unittest.TestCase):

    def test_known_vector(self):
        """SHA-256 of 'abc' must equal the NIST test vector."""
        expected = hashlib.sha256(b"abc").hexdigest()
        self.assertEqual(crypto.sha256("abc").hex(), expected)

    def test_deterministic(self):
        """Same inputs always produce the same digest."""
        d1 = crypto.sha256("hello", b"\x01\x02", 42)
        d2 = crypto.sha256("hello", b"\x01\x02", 42)
        self.assertEqual(d1, d2)

    def test_length(self):
        """Output is always 32 bytes."""
        self.assertEqual(len(crypto.sha256(b"anything")), 32)

    def test_different_inputs_differ(self):
        self.assertNotEqual(crypto.sha256("a"), crypto.sha256("b"))

    def test_concatenation_order_matters(self):
        """H(a||b) != H(b||a) in general."""
        self.assertNotEqual(crypto.sha256("foo", "bar"),
                            crypto.sha256("bar", "foo"))


class TestHashToInt(unittest.TestCase):

    def test_in_range(self):
        """Result is always in [0, n-1]."""
        for _ in range(20):
            v = crypto.hash_to_int(crypto.gen_nonce())
            self.assertGreaterEqual(v, 0)
            self.assertLess(v, crypto.ORDER_N)

    def test_deterministic(self):
        a = crypto.hash_to_int(b"fixed-input")
        b = crypto.hash_to_int(b"fixed-input")
        self.assertEqual(a, b)

    def test_nonzero_for_typical_inputs(self):
        """Probability of zero is negligible; assert it for a fixed input."""
        self.assertNotEqual(crypto.hash_to_int(b"nonzero-test"), 0)


class TestGenScalar(unittest.TestCase):

    def test_nonzero(self):
        for _ in range(50):
            s = crypto.gen_scalar()
            self.assertGreater(s, 0)
            self.assertLess(s, crypto.ORDER_N)

    def test_randomness(self):
        """Two successive scalars should (almost certainly) differ."""
        vals = {crypto.gen_scalar() for _ in range(10)}
        self.assertGreater(len(vals), 1)


class TestGenKeypair(unittest.TestCase):

    def test_returns_two_values(self):
        priv, pub = crypto.gen_keypair()
        self.assertIsInstance(priv, int)
        self.assertIsNotNone(pub)

    def test_public_key_matches(self):
        """Ppub = s.Q must hold."""
        s, Ppub = crypto.gen_keypair()
        self.assertEqual(crypto.scalar_mult(s), Ppub)

    def test_keypairs_are_distinct(self):
        k1, _ = crypto.gen_keypair()
        k2, _ = crypto.gen_keypair()
        self.assertNotEqual(k1, k2)


class TestScalarMult(unittest.TestCase):

    def test_identity(self):
        """n.Q = infinity (point at infinity)."""
        inf = crypto.scalar_mult(crypto.ORDER_N)
        self.assertEqual(inf, INFINITY)

    def test_one_times_Q(self):
        self.assertEqual(crypto.scalar_mult(1), crypto.Q)

    def test_commutativity_of_ecdh(self):
        """a.(b.Q) == b.(a.Q) — core ECDH property used for session keys."""
        a = crypto.gen_scalar()
        b = crypto.gen_scalar()
        ab_Q = crypto.scalar_mult(a, crypto.scalar_mult(b))
        ba_Q = crypto.scalar_mult(b, crypto.scalar_mult(a))
        self.assertEqual(ab_Q, ba_Q)

    def test_additivity(self):
        """(a+b).Q == a.Q + b.Q."""
        a, b = crypto.gen_scalar(), crypto.gen_scalar()
        lhs = crypto.scalar_mult((a + b) % crypto.ORDER_N)
        rhs = crypto.point_add(crypto.scalar_mult(a), crypto.scalar_mult(b))
        self.assertEqual(lhs, rhs)


class TestPointSerialization(unittest.TestCase):

    def test_round_trip(self):
        """serialize -> deserialize must recover the original point."""
        _, P = crypto.gen_keypair()
        b = crypto.point_to_bytes(P)
        P2 = crypto.bytes_to_point(b)
        self.assertEqual(P, P2)

    def test_length(self):
        """P-256 uncompressed point is 64 bytes (32 X + 32 Y)."""
        _, P = crypto.gen_keypair()
        self.assertEqual(len(crypto.point_to_bytes(P)), 64)

    def test_different_points_differ(self):
        _, P1 = crypto.gen_keypair()
        _, P2 = crypto.gen_keypair()
        self.assertNotEqual(crypto.point_to_bytes(P1),
                            crypto.point_to_bytes(P2))

    def test_base_point_serialization(self):
        """The base point Q serializes and deserializes correctly."""
        b = crypto.point_to_bytes(crypto.Q)
        Q2 = crypto.bytes_to_point(b)
        self.assertEqual(crypto.Q, Q2)


class TestDeriveSessionKey(unittest.TestCase):

    def test_deterministic(self):
        sk1 = crypto.derive_session_key(b"pid", b"did", b"Zj", b"Kj")
        sk2 = crypto.derive_session_key(b"pid", b"did", b"Zj", b"Kj")
        self.assertEqual(sk1, sk2)

    def test_length(self):
        sk = crypto.derive_session_key(b"a", b"b")
        self.assertEqual(len(sk), 32)

    def test_sensitive_to_each_input(self):
        base = crypto.derive_session_key(b"pid", b"did", b"Zj", b"Kj")
        self.assertNotEqual(base, crypto.derive_session_key(b"PID", b"did", b"Zj", b"Kj"))
        self.assertNotEqual(base, crypto.derive_session_key(b"pid", b"DID", b"Zj", b"Kj"))
        self.assertNotEqual(base, crypto.derive_session_key(b"pid", b"did", b"ZJ", b"Kj"))
        self.assertNotEqual(base, crypto.derive_session_key(b"pid", b"did", b"Zj", b"KJ"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
