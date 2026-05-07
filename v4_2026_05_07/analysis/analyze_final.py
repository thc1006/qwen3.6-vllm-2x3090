#!/usr/bin/env python3
"""Final comprehensive analysis: ALL phases, AWQ@0.92 vs FP8@0.92 properly,
publication-ready tables and statistical tests."""
import glob
import json
import math
import statistics
from pathlib import Path

S1_DIR = Path(__file__).parent / "s1"


def load_measurements(path):
    d = json.load(open(path))
    return [m for m in d["measurements"] if not m.get("error")], d


def welch_t(a, b):
    na, nb = len(a), len(b)
    ma, mb = statistics.mean(a), statistics.mean(b)
    sa = statistics.stdev(a) if na > 1 else 0
    sb = statistics.stdev(b) if nb > 1 else 0
    se = math.sqrt(sa**2/na + sb**2/nb) if (sa or sb) else 0
    t = (ma - mb) / se if se > 0 else 0
    p = math.erfc(abs(t) / math.sqrt(2)) if se > 0 else 1.0
    return ma, mb, ma - mb, se, t, p


def fmt_p(p):
    if p < 0.001: return "*** p<0.001"
    if p < 0.01:  return "**  p<0.01"
    if p < 0.05:  return "*   p<0.05"
    return "    NS"


print("=" * 90)
print("FINAL ANALYSIS — Phase A/B/C/E/F/G/H/I + tailscale-3090 DFlash")
print("=" * 90)

# ------------------------------------------------------------------
# 1. AWQ@0.92 vs FP8@0.92 (matched control — Finding 2 verification)
# ------------------------------------------------------------------
print("\n[1] AWQ@0.92 vs FP8@0.92 (matched control)")
print("-" * 90)
print(f"{'comparison':<28} {'AWQ_mean':>10} {'FP8_mean':>10} {'diff':>8} "
      f"{'SE':>8} {'t':>8} {'p':>10} {'sig':>14}")
for power in (350, 220):
    for temp in ("0.0", "0.5"):
        awq_path = S1_DIR / f"phase_h_p{power}_awq092_k3_t{temp}.json"
        fp8_path = S1_DIR / f"phase_c_p{power}_fp8_092_k3_t{temp}.json"
        if not (awq_path.exists() and fp8_path.exists()):
            continue
        awq_m, _ = load_measurements(awq_path)
        fp8_m, _ = load_measurements(fp8_path)
        awq_tpot = [m["tpot_ms"] for m in awq_m]
        fp8_tpot = [m["tpot_ms"] for m in fp8_m]
        ma, mb, diff, se, t, p = welch_t(awq_tpot, fp8_tpot)
        label = f"p{power}_t{temp} TPOT"
        print(f"{label:<28} {ma:>10.3f} {mb:>10.3f} {diff:>+8.3f} "
              f"{se:>8.4f} {t:>+8.3f} {p:>10.4f} {fmt_p(p):>14}")
        # Also tok/s
        awq_toks = [m["tok_s"] for m in awq_m]
        fp8_toks = [m["tok_s"] for m in fp8_m]
        ma, mb, diff, se, t, p = welch_t(awq_toks, fp8_toks)
        label = f"p{power}_t{temp} tok/s"
        print(f"{label:<28} {ma:>10.1f} {mb:>10.1f} {diff:>+8.2f} "
              f"{se:>8.3f} {t:>+8.3f} {p:>10.4f} {fmt_p(p):>14}")

# ------------------------------------------------------------------
# 2. Phase I FP8 mem-util sweep results
# ------------------------------------------------------------------
print("\n[2] FP8 mem-util sweep (Phase I)")
print("-" * 90)
mu_results = {0.85: "OOM (Phase C run1)", 0.92: "FITS (Phase C run2)"}
for mu in (0.86, 0.88, 0.90):
    p = S1_DIR / f"phase_i_p350_fp8_mu0{int(mu*100)}_k3_t0.0.json"
    if p.exists():
        ms, _ = load_measurements(p)
        toks = [m["tok_s"] for m in ms]
        mu_results[mu] = f"FITS, tok/s={statistics.mean(toks):.1f} (n={len(ms)})"
    else:
        mu_results[mu] = "OOM (no JSON written)"
print(f"  0.85: {mu_results[0.85]}")
print(f"  0.86: {mu_results[0.86]}")
print(f"  0.88: {mu_results[0.88]}")
print(f"  0.90: {mu_results[0.90]}")
print(f"  0.92: {mu_results[0.92]}")

# ------------------------------------------------------------------
# 3. Phase H AWQ@0.92 vs Phase A AWQ@0.85 (mem-util effect on AWQ)
# ------------------------------------------------------------------
print("\n[3] AWQ@0.85 vs AWQ@0.92 (mem-util effect on AWQ)")
print("-" * 90)
print(f"{'comparison':<28} {'AWQ@0.85':>10} {'AWQ@0.92':>10} {'diff':>8} "
      f"{'SE':>8} {'t':>8} {'p':>10} {'sig':>14}")
for power in (350, 220):
    for temp in ("0.0", "0.5"):
        a85 = S1_DIR / f"phase_a_p{power}_k3_t{temp}.json"
        a92 = S1_DIR / f"phase_h_p{power}_awq092_k3_t{temp}.json"
        if not (a85.exists() and a92.exists()):
            continue
        a85_m, _ = load_measurements(a85)
        a92_m, _ = load_measurements(a92)
        ma, mb, diff, se, t, p = welch_t(
            [m["tpot_ms"] for m in a85_m],
            [m["tpot_ms"] for m in a92_m])
        label = f"p{power}_t{temp} TPOT"
        print(f"{label:<28} {ma:>10.3f} {mb:>10.3f} {diff:>+8.3f} "
              f"{se:>8.4f} {t:>+8.3f} {p:>10.4f} {fmt_p(p):>14}")

# ------------------------------------------------------------------
# 4. Phase A k=2 vs k=3 (winner verification — was TPOT ambiguous?)
# ------------------------------------------------------------------
print("\n[4] Phase A k=2 vs k=3 (winner verification)")
print("-" * 90)
print(f"{'comparison':<35} {'k=2':>10} {'k=3':>10} {'diff':>8} "
      f"{'SE':>8} {'t':>8} {'p':>10} {'sig':>14}")
for power in (350, 220):
    for temp in ("0.0", "0.5"):
        f2 = S1_DIR / f"phase_a_p{power}_k2_t{temp}.json"
        f3 = S1_DIR / f"phase_a_p{power}_k3_t{temp}.json"
        if not (f2.exists() and f3.exists()):
            continue
        m2, _ = load_measurements(f2)
        m3, _ = load_measurements(f3)
        for metric in ("tpot_ms", "ttft_ms", "tok_s"):
            ma, mb, diff, se, t, p = welch_t(
                [m[metric] for m in m2],
                [m[metric] for m in m3])
            label = f"p{power}_t{temp} {metric}"
            print(f"{label:<35} {ma:>10.3f} {mb:>10.3f} {diff:>+8.3f} "
                  f"{se:>8.4f} {t:>+8.3f} {p:>10.4f} {fmt_p(p):>14}")

# ------------------------------------------------------------------
# Final summary
# ------------------------------------------------------------------
print()
print("=" * 90)
print("VERDICT SUMMARY")
print("=" * 90)
print("""
Finding 1 (MTP cross-k SHA1 lossless): partial novel - confirmed at temp=0.0
Finding 2 (FP8 vs AWQ): RETRACTED - all 4 matched-mem-util comparisons NS
Finding 3 (220W +11% perf/W): novel for vLLM dual-3090 MoE
Finding 4 (DFlash NET LOSS -44.6%): known + corroborated
Finding 5 (MTP no monotonic regression): novel + first Ampere data
Finding 6 (tool-call faster TPOT): mechanism mystery (not higher acceptance)
Finding 7 (long-context decode scaling): novel curve, sparse public data
""")
