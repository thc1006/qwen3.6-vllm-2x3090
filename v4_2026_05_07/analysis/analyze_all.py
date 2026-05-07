#!/usr/bin/env python3
"""Comprehensive analysis of all bench data for v4.0 / v3.0 publication.

Computes:
  - Phase A: per-config + per-prompt TPOT/TTFT/tok-s/acceptance with stdev
  - Cross-config SHA1 verification (MTP lossless check)
  - perf/W: tok/s per watt (using gpu_after.power_w from JSONs)
  - Phase E: one-sided p-value for monotonic regression hypothesis
  - Phase B/C summary tables
  - DFlash logs aggregation
"""
import glob
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

S1_DIR = Path(__file__).parent / "s1"
TAILSCALE_DIR = Path(__file__).parent / "tailscale-3090"


def load_json(path):
    with open(path) as f:
        return json.load(f)


# ----------------------------------------------------------------------------
# 1. PHASE A — k × power × temp aggregation + per-prompt
# ----------------------------------------------------------------------------
def analyze_phase_a():
    print("=" * 80)
    print("PHASE A — k × power × temp (12 configs, N=5 trials × 5 prompts)")
    print("=" * 80)
    files = sorted(S1_DIR.glob("phase_a_*.json"))
    print(f"\n{'config':<22} {'TPOT_mean':>10} {'stdev':>8} {'TTFT_mean':>10} "
          f"{'tok/s':>8} {'accept/cyc':>10} {'mean_W_pair':>12} {'tok/s/W':>8}")
    print("-" * 100)
    by_cfg = {}
    for f in files:
        d = load_json(f)
        s = d["summary"]
        cfg = d["config_id"]
        # mean power across measurements (avg of GPU 0 + GPU 1 power_after for each request)
        powers = []
        for m in d["measurements"]:
            if m.get("error") or not m.get("gpu_after"):
                continue
            gpus = m["gpu_after"]
            if isinstance(gpus, list) and len(gpus) >= 2:
                powers.append(gpus[0].get("power_w", 0) + gpus[1].get("power_w", 0))
        mean_w_pair = statistics.mean(powers) if powers else 0
        tok_s = s["tok_s"]["mean"]
        toks_per_w = tok_s / mean_w_pair if mean_w_pair > 0 else 0
        by_cfg[cfg] = {
            "tpot_mean_ms": s["tpot_ms"]["mean"],
            "tpot_stdev_ms": s["tpot_ms"]["stdev"],
            "ttft_mean_ms": s["ttft_ms"]["mean"],
            "tok_s_mean": tok_s,
            "accept_cyc": s["spec_acceptance_per_cycle"]["mean"],
            "mean_w_pair": mean_w_pair,
            "tok_s_per_w": toks_per_w,
            "unique_sha1": s["unique_sha1"],
            "n": s["n"],
            "measurements": d["measurements"],  # for per-prompt breakdown
        }
        print(f"{cfg:<22} {by_cfg[cfg]['tpot_mean_ms']:>10.3f} "
              f"{by_cfg[cfg]['tpot_stdev_ms']:>8.3f} "
              f"{by_cfg[cfg]['ttft_mean_ms']:>10.1f} "
              f"{tok_s:>8.1f} "
              f"{by_cfg[cfg]['accept_cyc']:>10.3f} "
              f"{mean_w_pair:>12.1f} "
              f"{toks_per_w:>8.4f}")
    return by_cfg


def per_prompt_breakdown(by_cfg, cfg_name):
    print(f"\n--- Per-prompt breakdown for {cfg_name} ---")
    cfg = by_cfg[cfg_name]
    by_prompt = defaultdict(list)
    for m in cfg["measurements"]:
        if not m.get("error"):
            by_prompt[m["prompt_id"]].append(m)
    print(f"{'prompt':<10} {'TPOT_mean':>10} {'TTFT_mean':>10} {'tok/s':>8} "
          f"{'accept/cyc':>10} {'unique_sha1':>12}")
    for pid, ms in sorted(by_prompt.items()):
        tpots = [m["tpot_ms"] for m in ms]
        ttfts = [m["ttft_ms"] for m in ms]
        toks = [m["tok_s"] for m in ms]
        accepts = [m["spec"]["length_per_cycle"] for m in ms
                   if m.get("spec") and m["spec"].get("length_per_cycle") is not None]
        sha_count = len({m["sha1"] for m in ms})
        print(f"{pid:<10} {statistics.mean(tpots):>10.3f} "
              f"{statistics.mean(ttfts):>10.1f} "
              f"{statistics.mean(toks):>8.1f} "
              f"{statistics.mean(accepts) if accepts else 0:>10.3f} "
              f"{sha_count:>12}")


# ----------------------------------------------------------------------------
# 2. CROSS-CONFIG SHA1 VERIFICATION (MTP lossless check)
# ----------------------------------------------------------------------------
def cross_config_sha1(by_cfg):
    print("\n" + "=" * 80)
    print("MTP LOSSLESS CHECK — does same prompt at same temp give same SHA1 across k?")
    print("=" * 80)
    # Group by (power, temp), then compare per-prompt sha1 across k=1, k=2, k=3
    for power in (350, 220):
        for temp in ("0.0", "0.5"):
            print(f"\n--- {power}W, temp={temp} ---")
            cfg_k1 = f"p{power}_k1_t{temp}"
            cfg_k2 = f"p{power}_k2_t{temp}"
            cfg_k3 = f"p{power}_k3_t{temp}"
            if not all(c in by_cfg for c in (cfg_k1, cfg_k2, cfg_k3)):
                print(f"  missing config")
                continue
            sha_by_prompt_k = defaultdict(dict)  # prompt_id -> {k: set of sha1}
            for k, cfg in [(1, cfg_k1), (2, cfg_k2), (3, cfg_k3)]:
                for m in by_cfg[cfg]["measurements"]:
                    if not m.get("error"):
                        sha_by_prompt_k[m["prompt_id"]].setdefault(k, set()).add(m["sha1"])
            print(f"{'prompt':<10} {'k=1 unique':>10} {'k=2 unique':>10} {'k=3 unique':>10} "
                  f"{'cross-k overlap':>16}")
            for pid in sorted(sha_by_prompt_k):
                k1s = sha_by_prompt_k[pid].get(1, set())
                k2s = sha_by_prompt_k[pid].get(2, set())
                k3s = sha_by_prompt_k[pid].get(3, set())
                overlap = k1s & k2s & k3s
                print(f"{pid:<10} {len(k1s):>10} {len(k2s):>10} {len(k3s):>10} "
                      f"{len(overlap):>16} (=0 means no shared sha1 → MTP introduces stochasticity)")


# ----------------------------------------------------------------------------
# 3. PHASE B — TP=2 only (run3, valid)
# ----------------------------------------------------------------------------
def analyze_phase_b():
    print("\n" + "=" * 80)
    print("PHASE B — TP=2 (4 configs valid; TP=1 OOM categorical fail × 4)")
    print("=" * 80)
    files = sorted(S1_DIR.glob("phase_b_*.json"))
    print(f"\n{'config':<22} {'TPOT_mean':>10} {'TTFT_mean':>10} {'tok/s':>8} "
          f"{'accept/cyc':>10}")
    print("-" * 80)
    for f in files:
        d = load_json(f)
        s = d["summary"]
        cfg = d["config_id"]
        print(f"{cfg:<22} {s['tpot_ms']['mean']:>10.3f} {s['ttft_ms']['mean']:>10.1f} "
              f"{s['tok_s']['mean']:>8.1f} {s['spec_acceptance_per_cycle']['mean']:>10.3f}")


# ----------------------------------------------------------------------------
# 4. PHASE C — AWQ vs FP8
# ----------------------------------------------------------------------------
def analyze_phase_c():
    print("\n" + "=" * 80)
    print("PHASE C — AWQ (gpu-mem 0.85) vs FP8 (gpu-mem 0.92, max-with-Whisper)")
    print("=" * 80)
    files = sorted(S1_DIR.glob("phase_c_*.json"))
    print(f"\n{'config':<28} {'TPOT_mean':>10} {'TTFT_mean':>10} {'tok/s':>8} "
          f"{'accept/cyc':>10}")
    print("-" * 90)
    for f in files:
        d = load_json(f)
        s = d["summary"]
        cfg = d["config_id"]
        print(f"{cfg:<28} {s['tpot_ms']['mean']:>10.3f} {s['ttft_ms']['mean']:>10.1f} "
              f"{s['tok_s']['mean']:>8.1f} {s['spec_acceptance_per_cycle']['mean']:>10.3f}")


# ----------------------------------------------------------------------------
# 5. PHASE E — stability + one-sided p-value
# ----------------------------------------------------------------------------
def analyze_phase_e():
    print("\n" + "=" * 80)
    print("PHASE E — 60-min stability × power")
    print("=" * 80)
    files = sorted(S1_DIR.glob("phase_e_*.json"))
    for f in files:
        d = load_json(f)
        cfg = f.stem
        t = d["trend"]
        # one-sided p-value (assuming monotonic decrease hypothesis: slope < 0)
        # if t < 0, one-sided p = two-sided / 2; if t > 0, one-sided p = 1 - (two-sided / 2)
        t_stat = t["t_statistic"]
        p2 = t["p_value_two_sided"]
        if t_stat < 0:
            p_one_sided = p2 / 2
        else:
            p_one_sided = 1 - p2 / 2
        sig_one = p_one_sided < 0.05
        print(f"\n  {cfg}:")
        print(f"    n={t['n']}, slope/hr={t['slope_per_hour']:.5f}, "
              f"r^2={t['r_squared']:.5f}")
        print(f"    t={t_stat:.3f}, p_two_sided={p2:.3f}, p_one_sided={p_one_sided:.3f}")
        print(f"    monotonic (one-sided): {'SIGNIFICANT' if sig_one else 'NOT significant'}")
        print(f"    first quartile: {t['first_quartile_mean']:.3f}, "
              f"last quartile: {t['last_quartile_mean']:.3f}, "
              f"degradation: {t['degradation_pct']:.2f}%")


# ----------------------------------------------------------------------------
# 6. DFLASH — parse llama.cpp logs from tailscale-3090
# ----------------------------------------------------------------------------
def parse_llama_log(path):
    """Extract Generation tok/s from llama-cli log."""
    text = open(path, encoding="utf-8", errors="replace").read()
    for line in text.splitlines():
        if "Prompt:" in line and "Generation:" in line:
            # Format: [ Prompt: X.X t/s | Generation: Y.Y t/s ]
            try:
                gen_part = line.split("Generation:")[1].split("t/s")[0].strip()
                return float(gen_part)
            except (IndexError, ValueError):
                continue
    return None


def analyze_dflash():
    print("\n" + "=" * 80)
    print("DFLASH on tailscale-3090 (1× 3090, llama.cpp PR #22105)")
    print("=" * 80)
    out_dir = TAILSCALE_DIR / "out_20260507_183341"
    if not out_dir.exists():
        print(f"  output dir not found: {out_dir}")
        return
    cfgs = ["01_baseline", "03_oleg_draft_2_32", "04_oleg_draft_2_16",
            "05_dflash_max16", "06_dflash_max8", "07_dflash_max4"]
    print(f"\n{'config':<28} {'p1':>6} {'p2':>6} {'p3':>6} {'p4':>6} {'p5':>6} "
          f"{'mean':>8} {'stdev':>8}")
    print("-" * 90)
    results = {}
    for cfg in cfgs:
        cdir = out_dir / cfg
        if not cdir.exists():
            print(f"{cfg:<28} (no data)")
            continue
        gen_rates = []
        for i in range(1, 6):
            log = cdir / f"p{i}.log"
            if log.exists():
                rate = parse_llama_log(log)
                gen_rates.append(rate if rate else 0)
            else:
                gen_rates.append(0)
        valid = [r for r in gen_rates if r > 0]
        if valid:
            mean = statistics.mean(valid)
            stdev = statistics.stdev(valid) if len(valid) > 1 else 0
        else:
            mean, stdev = 0, 0
        results[cfg] = mean
        rates_str = " ".join(f"{r:>5.1f}" if r > 0 else "  --  " for r in gen_rates)
        print(f"{cfg:<28} {rates_str} {mean:>8.1f} {stdev:>8.2f}")

    # Summary table
    print("\n  Cross-method comparison:")
    base = results.get("01_baseline", 0)
    if base > 0:
        for cfg, mean in results.items():
            delta = (mean - base) / base * 100 if base else 0
            print(f"    {cfg:<28} {mean:>8.1f} tok/s  ({delta:+.1f}% vs baseline)")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    by_cfg = analyze_phase_a()
    per_prompt_breakdown(by_cfg, "p350_k3_t0.0")
    per_prompt_breakdown(by_cfg, "p350_k3_t0.5")
    cross_config_sha1(by_cfg)
    analyze_phase_b()
    analyze_phase_c()
    analyze_phase_e()
    analyze_dflash()
