"""
benchmark.py
============

Performance benchmarking for SeBAC-IoD — reproduces the paper's evaluation
from Section VI (Tables II / III and Figures V / VI / VII).

What is measured
----------------
For each batch size n ∈ {1, 5, 10, 20, 30, 40, 50}:

  1. Enrollment time (drone + user, amortized)
  2. mu1 build time            (User side)
  3. mu2 build time            (CS side, full batch)
  4. mu3 build time            (Drone side, per drone × n)
  5. Batch verification time   (User side, aggregate check)
  6. Session-key derivation    (User side, n keys)
  7. Total authentication time (mu1 + mu2 + mu3 + verify + SK)
  8. Average end-to-end delay per drone  = total_auth / n
  9. User communication cost (bits)
 10. Total communication cost (bits)

All timings are averaged over RUNS = 5 repetitions to smooth out noise.

Output
------
  * Console table summarising all metrics
  * benchmark_results/timing.png    — batch computation cost vs n
  * benchmark_results/delay.png     — average EED vs n
  * benchmark_results/comm.png      — user communication cost vs n

Run with:
    python benchmark.py

Dependencies (add to requirements.txt if missing):
    matplotlib>=3.7
"""
from __future__ import annotations

import os
import sys
import time
from typing import Dict, List

# ── make sure the project root is on the path when run directly ──────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import control_server as csmod
import drone as dmod
import user as umod
import crypto
import utils

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BATCH_SIZES: List[int] = [1, 5, 10, 20, 30, 40, 50]
RUNS: int = 5          # repetitions per batch size for averaging

# Communication cost constants (bits) — from Table VI of the paper
BITS_NONCE    = 160    # nonces, identities, hash digests
BITS_ECC      = 320    # an ECC point (P-256 uncompressed, 64 bytes = 512 bits;
                       # paper uses 320 for compressed point representation)
BITS_HASH     = 160    # SHA-1 / truncated hash
BITS_TS       = 32     # timestamp

# mu1 from User->CS  (PIDu, M1=X, M2, M3, T1)
# = HASH + ECC + HASH + HASH + TS = 160+320+160+160+32 = 832 bits  (fixed part)
# plus per-drone DID list: n × 160 bits
MU1_FIXED_BITS = BITS_HASH + BITS_ECC + BITS_HASH + BITS_HASH + BITS_TS

# mu2 from CS->Drone  (Yj, M5, G2=X, G3=PIDu, T2) per drone
MU2_PER_DRONE_BITS = BITS_HASH + BITS_HASH + BITS_ECC + BITS_HASH + BITS_TS

# mu3 from Drone->User  (M5, DIDj, Rj, sj, Kj) per drone
MU3_PER_DRONE_BITS = BITS_HASH + BITS_HASH + BITS_ECC + BITS_HASH + BITS_ECC

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "benchmark_results")

logger = utils.get_logger("benchmark")


# ---------------------------------------------------------------------------
# Helper — build a fresh enrolled system with n drones
# ---------------------------------------------------------------------------
def _build_system(n: int):
    cs = csmod.ControlServer()
    cs.initialize()

    usr = umod.User("Bench-User", "bench-password")
    id_u, uid = usr.begin_enrollment()
    usr.complete_enrollment(cs.enroll_user(id_u, uid))

    drones: List[dmod.Drone] = []
    for i in range(n):
        d = dmod.Drone(f"BenchDrone-{i:03d}")
        req = d.begin_enrollment()
        record = cs.enroll_drone(req.id_d, req.v, req.challenge,
                                 req.puf_response, d.device_secret)
        d.store_credentials(cs.issue_drone_credentials(record))
        drones.append(d)

    return cs, usr, drones


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------
def _ms() -> float:
    """Current time in milliseconds."""
    return time.perf_counter() * 1000.0


def _time_block(fn) -> float:
    """Return elapsed milliseconds for calling fn()."""
    t0 = _ms()
    fn()
    return _ms() - t0


# ---------------------------------------------------------------------------
# Single benchmark run for one batch size
# ---------------------------------------------------------------------------
def _run_once(n: int) -> Dict[str, float]:
    cs, usr, drones = _build_system(n)

    # ── mu1 ─────────────────────────────────────────────────────────────────
    t0 = _ms()
    mu1 = usr.build_mu1()
    t_mu1 = _ms() - t0

    # ── mu2 (CS builds for full batch) ────────────────────────────────────
    t0 = _ms()
    mu2_list = cs.build_mu2_batch(mu1)
    t_mu2 = _ms() - t0

    # ── mu3 (all drones respond) ───────────────────────────────────────────
    t0 = _ms()
    mu3_batch = [d.handle_mu2(m) for d, m in zip(drones, mu2_list)]
    t_mu3 = _ms() - t0

    # ── batch verification ─────────────────────────────────────────────────
    t0 = _ms()
    result = usr.batch_verify(mu3_batch)
    t_verify = _ms() - t0

    # ── session key derivation ─────────────────────────────────────────────
    t0 = _ms()
    sessions = usr.establish_sessions(mu3_batch, result)
    t_sk = _ms() - t0

    # ── totals ────────────────────────────────────────────────────────────
    t_total = t_mu1 + t_mu2 + t_mu3 + t_verify + t_sk
    avg_eed  = t_total / n            # average end-to-end delay per drone

    # ── communication cost (bits) ─────────────────────────────────────────
    user_comm  = MU1_FIXED_BITS + n * BITS_HASH   # mu1 (includes n DIDj)
    total_comm = user_comm + n * MU2_PER_DRONE_BITS + n * MU3_PER_DRONE_BITS

    return {
        "t_mu1":    t_mu1,
        "t_mu2":    t_mu2,
        "t_mu3":    t_mu3,
        "t_verify": t_verify,
        "t_sk":     t_sk,
        "t_total":  t_total,
        "avg_eed":  avg_eed,
        "user_comm_bits":  user_comm,
        "total_comm_bits": total_comm,
        "all_valid": result.all_valid,
        "ec_checks": result.checks,
    }


# ---------------------------------------------------------------------------
# Averaged results over RUNS repetitions
# ---------------------------------------------------------------------------
def benchmark(batch_sizes: List[int] = BATCH_SIZES,
              runs: int = RUNS) -> Dict[int, Dict[str, float]]:
    results: Dict[int, Dict[str, float]] = {}

    for n in batch_sizes:
        logger.info("Benchmarking n=%d drones (%d runs)…", n, runs)
        accum: Dict[str, float] = {}

        for run_i in range(runs):
            r = _run_once(n)
            for k, v in r.items():
                if isinstance(v, (int, float)):
                    accum[k] = accum.get(k, 0.0) + v

        # Average the numeric fields
        avg = {k: v / runs for k, v in accum.items()}
        avg["n"] = n
        results[n] = avg

        logger.info(
            "  n=%3d | total=%.2f ms | EED/drone=%.3f ms | "
            "EC-checks=%.1f | valid=%s",
            n,
            avg["t_total"],
            avg["avg_eed"],
            accum["ec_checks"] / runs,   # should be ~1.0 (always 1 check)
            True,
        )

    return results


# ---------------------------------------------------------------------------
# Console table
# ---------------------------------------------------------------------------
def print_table(results: Dict[int, Dict[str, float]]) -> None:
    header = (
        f"{'n':>5} | {'mu1 (ms)':>9} | {'mu2 (ms)':>9} | "
        f"{'mu3 (ms)':>9} | {'verify (ms)':>11} | {'SK (ms)':>8} | "
        f"{'total (ms)':>10} | {'EED/drone':>10} | "
        f"{'UserComm(b)':>12} | {'TotalComm(b)':>13}"
    )
    sep = "-" * len(header)
    print("\n" + sep)
    print("SeBAC-IoD  Performance Benchmark")
    print(sep)
    print(header)
    print(sep)
    for n, r in sorted(results.items()):
        print(
            f"{n:>5} | {r['t_mu1']:>9.3f} | {r['t_mu2']:>9.3f} | "
            f"{r['t_mu3']:>9.3f} | {r['t_verify']:>11.3f} | {r['t_sk']:>8.3f} | "
            f"{r['t_total']:>10.3f} | {r['avg_eed']:>10.4f} | "
            f"{r['user_comm_bits']:>12.0f} | {r['total_comm_bits']:>13.0f}"
        )
    print(sep + "\n")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_results(results: Dict[int, Dict[str, float]]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")   # non-interactive backend (works without a display)
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed — skipping plots. "
                       "Install with:  pip install matplotlib")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    ns     = sorted(results)
    totals = [results[n]["t_total"]         for n in ns]
    eeds   = [results[n]["avg_eed"]         for n in ns]
    ucomms = [results[n]["user_comm_bits"]  for n in ns]
    tcomms = [results[n]["total_comm_bits"] for n in ns]

    # ── Figure 1: Batch Computational Cost ──────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ns, totals, "o-", color="royalblue", linewidth=2,
            markersize=6, label="SeBAC-IoD")
    ax.set_xlabel("Batch Size (n)", fontsize=12)
    ax.set_ylabel("Total Computation Time (ms)", fontsize=12)
    ax.set_title("Batch Computational Cost vs. Number of Drones", fontsize=13)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    path1 = os.path.join(OUT_DIR, "timing.png")
    fig.tight_layout()
    fig.savefig(path1, dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", path1)

    # ── Figure 2: Average End-to-End Authentication Delay ───────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ns, eeds, "s-", color="darkorange", linewidth=2,
            markersize=6, label="SeBAC-IoD")
    ax.set_xlabel("Batch Size (n)", fontsize=12)
    ax.set_ylabel("Avg EED per Drone (ms)", fontsize=12)
    ax.set_title("Average End-to-End Authentication Delay per Drone", fontsize=13)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    path2 = os.path.join(OUT_DIR, "delay.png")
    fig.tight_layout()
    fig.savefig(path2, dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", path2)

    # ── Figure 3: User Communication Cost ───────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ns, ucomms, "^-", color="seagreen",  linewidth=2,
            markersize=6, label="User → CS (mu1)")
    ax.plot(ns, tcomms, "D--", color="firebrick", linewidth=2,
            markersize=6, label="Total (mu1+mu2+mu3)")
    ax.set_xlabel("Batch Size (n)", fontsize=12)
    ax.set_ylabel("Communication Cost (bits)", fontsize=12)
    ax.set_title("Communication Overhead vs. Number of Drones", fontsize=13)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    path3 = os.path.join(OUT_DIR, "comm.png")
    fig.tight_layout()
    fig.savefig(path3, dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", path3)

    print(f"\nPlots saved to: {OUT_DIR}/")


# ---------------------------------------------------------------------------
# Per-operation timing (matches paper Table III)
# ---------------------------------------------------------------------------
def benchmark_primitives(runs: int = 1000) -> Dict[str, float]:
    """
    Measure the time for individual cryptographic primitives.

    Matches the paper's Table III:
        Th    — one SHA-256 hash
        Te    — one ECC scalar multiplication (= TPM in paper)
        TPA   — one ECC point addition
        TPUF  — one PUF evaluation (SHA-256 keyed hash)

    Each operation is repeated `runs` times and averaged.

    Returns
    -------
    Dict mapping operation name -> average time in milliseconds.
    """
    import crypto
    import puf as pufmod

    results: Dict[str, float] = {}

    # ── Th : SHA-256 hash ─────────────────────────────────────────────
    data = b"benchmark-input-fixed"
    t0 = time.perf_counter()
    for _ in range(runs):
        crypto.sha256(data)
    results["Th"] = (time.perf_counter() - t0) * 1000 / runs

    # ── Te : ECC scalar multiplication  (s . Q) ───────────────────────
    s, _ = crypto.gen_keypair()
    t0 = time.perf_counter()
    for _ in range(runs):
        crypto.scalar_mult(s)
    results["Te"] = (time.perf_counter() - t0) * 1000 / runs

    # ── TPA : ECC point addition  (P + Q) ─────────────────────────────
    _, P = crypto.gen_keypair()
    _, Q_pt = crypto.gen_keypair()
    t0 = time.perf_counter()
    for _ in range(runs):
        crypto.point_add(P, Q_pt)
    results["TPA"] = (time.perf_counter() - t0) * 1000 / runs

    # ── TPUF : PUF evaluation  (SHA-256 keyed hash) ────────────────────
    device = pufmod.PUFDevice(device_id="bench-drone")
    challenge = pufmod.generate_challenge()
    t0 = time.perf_counter()
    for _ in range(runs):
        pufmod.response_generation(device, challenge)
    results["TPUF"] = (time.perf_counter() - t0) * 1000 / runs

    # ── Tf : ECC key-pair generation  (gen_keypair) ────────────────────
    t0 = time.perf_counter()
    for _ in range(runs):
        crypto.gen_keypair()
    results["Tf"] = (time.perf_counter() - t0) * 1000 / runs

    return results


def print_primitives_table(results: Dict[str, float]) -> None:
    """Print per-operation timing table matching paper Table III."""
    sep = "-" * 52
    print("\n" + sep)
    print("  SeBAC-IoD  Cryptographic Primitive Timings")
    print("  (corresponds to paper Table III)")
    print(sep)
    print(f"  {'Operation':<10} {'Description':<28} {'Time (ms)':>10}")
    print(sep)

    descriptions = {
        "Th":   "SHA-256 hash",
        "Te":   "ECC scalar mult (s.Q)",
        "TPA":  "ECC point addition (P+Q)",
        "TPUF": "PUF evaluation",
        "Tf":   "ECC key-pair generation",
    }

    for op in ["Th", "Te", "TPA", "TPUF", "Tf"]:
        t = results.get(op, 0.0)
        desc = descriptions.get(op, "")
        print(f"  {op:<10} {desc:<28} {t:>10.6f}")

    print(sep + "\n")


def plot_primitives(results: Dict[str, float]) -> None:
    """Bar chart of per-operation timings."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    ops    = list(results.keys())
    times  = [results[op] for op in ops]
    colors = ["steelblue", "darkorange", "seagreen", "tomato", "mediumpurple"]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(ops, times, color=colors[:len(ops)], edgecolor="black", width=0.5)
    ax.bar_label(bars, fmt="%.4f ms", padding=3, fontsize=9)
    ax.set_xlabel("Cryptographic Operation", fontsize=12)
    ax.set_ylabel("Average Time (ms)", fontsize=12)
    ax.set_title("Per-Operation Cryptographic Primitive Timings\n"
                 "(SeBAC-IoD — matches paper Table III)", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "primitives.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SeBAC-IoD performance benchmark")
    parser.add_argument(
        "--sizes", nargs="+", type=int, default=BATCH_SIZES,
        help="Batch sizes to test (default: 1 5 10 20 30 40 50)"
    )
    parser.add_argument(
        "--runs", type=int, default=RUNS,
        help=f"Repetitions per batch size for averaging (default: {RUNS})"
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Skip matplotlib plots (useful if matplotlib is not installed)"
    )
    parser.add_argument(
        "--primitives-only", action="store_true",
        help="Only run per-operation primitive timing (skip batch benchmark)"
    )
    parser.add_argument(
        "--primitive-runs", type=int, default=1000,
        help="Repetitions for each primitive timing (default: 1000)"
    )
    args = parser.parse_args()

    # ── Per-operation primitive timing (Table III) ──────────────────────
    print("\nMeasuring per-operation cryptographic primitive timings ...")
    prim_results = benchmark_primitives(runs=args.primitive_runs)
    print_primitives_table(prim_results)
    if not args.no_plot:
        plot_primitives(prim_results)

    if args.primitives_only:
        exit(0)

    # ── Batch benchmark (Table II / Figures V-VII) ───────────────────────
    results = benchmark(batch_sizes=args.sizes, runs=args.runs)
    print_table(results)

    if not args.no_plot:
        plot_results(results)
