# SeBAC-IoD — Secure and Efficient Batch Access Control for Internet of Drones

Python implementation of the **SeBAC-IoD** protocol from the IEEE TVT paper:
> *"SeBAC-IoD: A Secure and Efficient Batch Access Control Technique for Internet of Drones"*
> Chaudhry et al., IEEE Transactions on Vehicular Technology, 2026. DOI: 10.1109/TVT.2026.3651651

---

## What is SeBAC-IoD?

SeBAC-IoD is a lightweight **batch authentication protocol** for the Internet of Drones (IoD). Instead of authenticating drones one-by-one, a single user can authenticate a whole swarm in **one protocol session** using:

- **PUF** (Physical Unclonable Function) — resists physical drone capture
- **ECC** (Elliptic Curve Cryptography, NIST P-256) — lightweight key agreement
- **Batch Schnorr verification** with random weights (Vecj) — O(1) EC checks for N drones
- **Divide-and-conquer fault isolation** — locates rogue drones in O(log N) checks

---

## Project Structure

```
.
├── crypto.py            # ECC + SHA-256 primitive layer
├── puf.py               # Simulated PUF (SHA-256 keyed hash model)
├── database.py          # In-memory CS database
├── utils.py             # Logging, timestamps, byte helpers
├── control_server.py    # CS protocol logic (all 4 phases)
├── drone.py             # Drone entity (enrollment + mu2/mu3)
├── user.py              # User entity (enrollment + batch verify + SKij)
├── main.py              # Local simulation — runs full protocol, prints everything
├── protocol_messages.py # JSON framing for socket transport
├── server_app.py        # CS as TCP server (ports 9000 + 9001)
├── drone_app.py         # Drone as TCP client/server
├── user_app.py          # User as TCP client
├── socket_main.py       # All-in-one socket simulation launcher
├── benchmark.py         # Performance benchmarking + matplotlib plots
├── tests/
│   ├── test_crypto.py       # 24 tests — crypto primitives
│   ├── test_puf.py          # 17 tests — PUF simulation
│   ├── test_enrollment.py   # 27 tests — Phase 1/2/3
│   └── test_authentication.py # 37 tests — Phase 4, batch verify, fault isolation
├── diagrams/
│   ├── architecture.txt
│   └── sequence_diagram.txt
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
```

`requirements.txt` only needs:
```
ecdsa==0.19.2
matplotlib>=3.7   # only for benchmark plots
```

---

## Run Commands

### 1. Local simulation (full protocol, all values printed)
```bash
python main.py
```

### 2. Unit tests (105 tests)
```bash
python -m pytest tests/ -v
```

### 3. Performance benchmark (timing table + 3 plots)
```bash
python benchmark.py
python benchmark.py --sizes 1 5 10 20 50 --runs 5
```

### 4. Socket simulation — single terminal
```bash
python socket_main.py --drones 3
```

### 5. Socket simulation — separate terminals (real network model)
```bash
# Terminal 1
python server_app.py

# Terminal 2, 3, 4
python drone_app.py --id Drone-00 --port 9100
python drone_app.py --id Drone-01 --port 9101
python drone_app.py --id Drone-02 --port 9102

# Terminal 5 (run last)
python user_app.py --drones 3
```

---

## Protocol Phases

| Phase | Who | What |
|-------|-----|------|
| 1 — Initialization | CS | Generates master key `s`, public key `Ppub = s.Q` |
| 2 — Drone Enrollment | Drone → CS | PUF challenge/response, computes DIDj, Ksd, Kj, Ej, Wj, Nj |
| 3 — User Enrollment | User → CS | Computes UID, PIDu, FIDs, local verifiers (gamma_u, delta_u, Au, Du) |
| 4 — Batch Authentication | User ↔ CS ↔ Drones | mu1 → mu2 → mu3, batch Schnorr verify, ECDH session keys SKij |

---

## Security Properties Demonstrated

- **Mutual authentication** — CS, User, and each Drone authenticate each other
- **Anonymity & unlinkability** — PIDu pseudo-identity updated every session
- **Replay attack resistance** — timestamps T1/T2 with freshness window
- **Physical capture resistance** — PUF makes session keys unrecoverable from memory
- **Perfect forward secrecy** — ephemeral ECDH keys per session
- **Batch fault isolation** — divide-and-conquer locates rogue drones in O(log N)

---

## Test Results

```
105 passed in ~2s
```

## Benchmark (n = batch size, averaged over 5 runs)

| n  | Total (ms) | EED/drone (ms) | UserComm (bits) |
|----|-----------|----------------|-----------------|
| 1  | ~11       | ~11.0          | 992             |
| 10 | ~63       | ~6.3           | 2432            |
| 50 | ~286      | ~5.7           | 8832            |

---

## Paper Reference

S. A. Chaudhry, A. Irshad, Z. Akhtar, S. O. Gilani, Y. B. Zikria, A. K. Das,
*"SeBAC-IoD: A Secure and Efficient Batch Access Control Technique for Internet of Drones,"*
IEEE Transactions on Vehicular Technology, 2026.
DOI: [10.1109/TVT.2026.3651651](https://doi.org/10.1109/TVT.2026.3651651)
