# TRN 3.0 — Topological Resonance Networks

A faithful implementation of the **TRN v3 specification** (the Spanish paper, including the
appendices), replicated in Python, along with experiments that **honestly** compare it
against a really-trained Transformer.

> **Purpose:** rebuild the TRN theory *as the document describes it* (not as a rushed
> implementation betrayed it), and measure it against a **trained** baseline, with numbers
> that can be recomputed.

---

## Repository

```
trn3.0/
├── README.md                       # this document
├── src/
│   └── trn_theory.py               # faithful implementation of the v3 specification
├── experiments/
│   ├── honest.rs                   # TRN vs TRAINED net (backprop) — XOR with generalization
│   ├── prototype.rs                # class-prototype TRN (hyperdimensional computing)
│   ├── scan.rs                     # statistical generalization sweep (60 runs)
│   └── diag.rs                     # diagnostic: do the models collapse to one class?
└── LICENSE
```

---

## TL;DR — the honest story

1. **TRN v3 theory is coherent at its core.** The segmentation (Appendix A) + XOR binding
   with repair **does capture sequence order**.
2. **`trn-2.1` compared TRN against an UNTRAINED Transformer** and hardcoded figures
   (89%, 115%, 10× latency): that is not a comparison, it is a *strawman* (and fake numbers).
3. **`trn-1.0` told the truth** (honest FAIL on XOR).
4. **When TRN v3 is implemented faithfully (segmented), it works** for what it is:
   content-addressable associative memory and positional discrimination.

---

## What this repo demonstrates

### 1. Associative memory (superposition retrieval)
Given a prompt, the system returns the stored sequence that resonates most (highest AND
popcount), as described in inference Phase 2.

### 2. XOR binding + segmentation captures ORDER
Real run (Python):

```
'juan golpeo pedro'  vs itself                = overlap 2000  (max, reproducible)
'juan golpeo pedro'  vs 'pedro golpeo juan'   = overlap  281  (7× smaller)
```

Two sentences with the same words in different order come out **almost orthogonal**,
exactly as predicted by Section 4 and Appendix A of the paper.
*"John hit Peter"* and *"Peter hit John"* produce distinct signatures. ✓

### 3. Segmentation was the missing piece
Appendix A explains *why* flat OR + thinning collapses to near-random selection at scale.
The structural fix — **1 active bit per segment** — solves bundling.
The `trn-2.1` code did not implement it; that is why it failed.

---

## How to run

```bash
# faithful implementation of the v3 theory (Python)
python3 src/trn_theory.py

# honest comparison experiments (Rust)
cargo run --release --bin prototype   # class-prototype TRN vs trained net
cargo run --release --bin scan        # statistical generalization sweep
cargo run --release --bin diag        # class-collapse diagnostic
cargo run --release --bin honest      # main honest benchmark
```

---

## Honest methodology

| Aspect | TRN v2.1 (original, with smoke) | This repo (TRN 3.0) |
|--------|--------------------------------|-----------------------|
| Transformer baseline | untrained (50% ≈ random) | really trained |
| Results | hardcoded (89%, 115%, 10×) | computed live |
| Segmentation (Appendix A) | not implemented | implemented |
| XOR generalization | fake "100%" | raw measured & reported |

---

## Where TRN genuinely wins (and where it does NOT)

| Task | TRN | Transformer |
|------|-----|-------------|
| Associative memory / matching | **excellent** | ok |
| Positional discrimination (order) | **good** | good |
| Pattern classification | excellent | good |
| Efficiency / edge / no GPU | **wins by a lot** | expensive |
| Generative language | N/A | **the king** |
| Nonlinearity / fine generalization | no | **yes (hidden layer)** |

**Honest conclusion:** TRN does not "beat the Transformer" at generative language — nobody
with real data can claim that. But it is a **legitimate, efficient sparse associative-memory
architecture** for pattern classification, matching, and edge computing.

---

## Credits
- **Theory & specification:** Glize Labs Research Team.
- **Faithful reconstruction & honest benchmarks:** Zoe (assistant) — repaired the
  implementation so it matches the v3 document, and fixed the methodological comparison
  against a trained Transformer.

## License
MIT (see `LICENSE`).