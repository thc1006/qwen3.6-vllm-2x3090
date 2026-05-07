#!/usr/bin/env python3
"""Phase F.2: tool-call vs text-only acceptance rate comparison.

Tests Finding 6 hypothesis: structured JSON tool_call tokens have HIGHER
MTP acceptance rate than text generation, explaining the FASTER TPOT
observed in Phase F (3.04 ms) vs Phase A k=3 (4.99 ms).
"""
import json
import math
import statistics
from pathlib import Path

S1_DIR = Path(__file__).parent / "s1"


def collect_accept_rates(json_path):
    """Extract per-request length_per_cycle from a phase JSON."""
    d = json.load(open(json_path))
    rates = []
    for m in d["measurements"]:
        if m.get("error"):
            continue
        spec = m.get("spec")
        if not spec or spec.get("length_per_cycle") is None:
            continue
        rates.append(spec["length_per_cycle"])
    return rates


def main():
    # Phase F: tool-call workload (5 prompts × 5 trials)
    f_rates = collect_accept_rates(S1_DIR / "phase_f_tool_call.json")
    # Phase A k=3 t=0.0: text-only (5 prompts × 5 trials, same temp)
    a_rates = collect_accept_rates(S1_DIR / "phase_a_p350_k3_t0.0.json")

    if not f_rates or not a_rates:
        print("Missing data")
        return

    print("=" * 70)
    print("Phase F.2 — tool-call vs text-only MTP acceptance rate (k=3)")
    print("=" * 70)
    print(f"\n{'workload':<30} {'n':>4} {'mean accept/cyc':>16} {'stdev':>8} "
          f"{'as %k=3':>10}")
    print("-" * 70)
    for name, rates in [
        ("Phase F: tool-call", f_rates),
        ("Phase A k=3 t=0.0: text", a_rates),
    ]:
        m = statistics.mean(rates)
        sd = statistics.stdev(rates) if len(rates) > 1 else 0
        print(f"{name:<30} {len(rates):>4} {m:>16.3f} {sd:>8.3f} {m/3*100:>9.1f}%")

    # Welch's t-test (unequal variance)
    nf, na = len(f_rates), len(a_rates)
    mf, ma = statistics.mean(f_rates), statistics.mean(a_rates)
    sf = statistics.stdev(f_rates) if nf > 1 else 0
    sa = statistics.stdev(a_rates) if na > 1 else 0
    se_diff = math.sqrt(sf**2/nf + sa**2/na)
    t = (mf - ma) / se_diff if se_diff > 0 else 0
    p = math.erfc(abs(t) / math.sqrt(2))

    print(f"\nWelch's t-test (tool-call vs text):")
    print(f"  diff (tool - text) = {mf - ma:+.3f}")
    print(f"  SE_diff            = {se_diff:.4f}")
    print(f"  t-statistic        = {t:.3f}")
    print(f"  p-value (2-sided)  = {p:.4f}")
    print(f"  significance       = {'*** p<0.001' if p<0.001 else '** p<0.01' if p<0.01 else '* p<0.05' if p<0.05 else 'NS'}")
    print()

    if p < 0.05 and mf > ma:
        print("[OK] HYPOTHESIS CONFIRMED: tool-call has HIGHER acceptance rate (mechanism supported)")
    elif p < 0.05 and mf < ma:
        print("[X] HYPOTHESIS REFUTED: tool-call has LOWER acceptance rate")
    else:
        print("[?] INCONCLUSIVE: no significant difference at alpha=0.05")

    # Also compare TPOT directly to see consistency
    fd = json.load(open(S1_DIR / "phase_f_tool_call.json"))
    ad = json.load(open(S1_DIR / "phase_a_p350_k3_t0.0.json"))
    f_tpot = [m["tpot_ms"] for m in fd["measurements"] if not m.get("error")]
    a_tpot = [m["tpot_ms"] for m in ad["measurements"] if not m.get("error")]
    print(f"\nTPOT comparison:")
    print(f"  Phase F (tool-call): {statistics.mean(f_tpot):.3f} ± {statistics.stdev(f_tpot):.3f} ms (n={len(f_tpot)})")
    print(f"  Phase A k=3 t=0.0  : {statistics.mean(a_tpot):.3f} ± {statistics.stdev(a_tpot):.3f} ms (n={len(a_tpot)})")
    se_tpot = math.sqrt(statistics.stdev(f_tpot)**2/len(f_tpot) + statistics.stdev(a_tpot)**2/len(a_tpot))
    t_tpot = (statistics.mean(f_tpot) - statistics.mean(a_tpot)) / se_tpot
    p_tpot = math.erfc(abs(t_tpot) / math.sqrt(2))
    print(f"  Welch t={t_tpot:.3f}, p={p_tpot:.4f} ({'*** sig' if p_tpot<0.001 else 'NS'})")


if __name__ == "__main__":
    main()
