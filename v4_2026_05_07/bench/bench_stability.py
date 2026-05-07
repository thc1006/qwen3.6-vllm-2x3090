#!/usr/bin/env python3
"""Long-time MTP draft acceptance stability test (Task #53).

Stress-test for vllm #41838 hypothesis (Eagle3 acceptance length monotonically
regresses over hours on H200). Verify whether MTP on dual 3090 also regresses.

Method:
  - 1 warmup request (discarded)
  - Pick random prompt from 5-set, generate N tokens, sleep, repeat for DURATION
  - Per-request: log timestamp, TPOT, TTFT, draft acceptance ratio
  - End: linear regression of acceptance vs time → detect monotonic trend
  - GPU snapshot every minute (mem, temp, power)
  - Trend detection: slope/hr + r^2 + p-value (normal-approximation t-test)
  - Fail-fast: abort early if last 20 requests have >75% errors

Usage:
  python3 bench_stability.py --duration-min 60 --output bench_stability.json
"""
import argparse
import json
import math
import random
import statistics
import time
from pathlib import Path

# Reuse the bench_runner module
import bench_runner

PROMPTS = bench_runner.PROMPTS


def linear_regression(xs, ys):
    """OLS with t-test: returns (slope, intercept, r_squared, t_stat, p_two_sided)."""
    n = len(xs)
    if n < 3:
        return None
    mx = statistics.mean(xs)
    my = statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return None
    slope = num / den
    intercept = my - slope * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    # t-statistic for slope = 0 hypothesis
    # SE(slope)^2 = MSE / sum((x - mean_x)^2), MSE = ss_res / (n - 2)
    mse = ss_res / (n - 2) if n > 2 else 0
    se_slope = math.sqrt(mse / den) if mse > 0 and den > 0 else 0
    t_stat = slope / se_slope if se_slope > 0 else 0
    # 2-sided p-value via normal approximation (df > 30 → student-t ≈ N(0,1))
    # erfc(|t|/sqrt(2)) gives 2 * (1 - phi(|t|))
    p_two_sided = math.erfc(abs(t_stat) / math.sqrt(2)) if abs(t_stat) > 0 else 1.0
    return slope, intercept, r2, t_stat, p_two_sided


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration-min", type=int, default=60)
    ap.add_argument("--max-tokens", type=int, default=100)
    ap.add_argument("--sleep-between-s", type=float, default=2.0,
                    help="Sleep between requests — 2.0 default for sustained load (1440 reqs/hr)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    random.seed(42)
    rng = random.Random(42)

    server_meta = {}
    import urllib.request
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:8000/v1/models", timeout=5) as r:
            server_meta = json.loads(r.read())
    except Exception as e:
        server_meta = {"error": str(e)}

    # Warmup (discard) — first request often has cold-cache effects
    print("=== warmup (1 prompt, discarded) ===", flush=True)
    try:
        bench_runner.call_streaming(PROMPTS[0][1], max_tokens=args.max_tokens,
                                    temperature=args.temperature)
    except Exception as e:
        print(f"  warmup error: {e}", flush=True)

    end_t = time.time() + args.duration_min * 60
    measurements = []
    last_gpu_log = 0
    print(f"=== Stability test: {args.duration_min} min, "
          f"{args.max_tokens} tok/req, sleep={args.sleep_between_s}s ===", flush=True)

    request_idx = 0
    t_start = time.time()
    while time.time() < end_t:
        # Fail-fast: abort if vLLM seems dead (>75% errors in recent window)
        if len(measurements) >= 20:
            recent_errs = sum(1 for m in measurements[-20:] if m.get("error"))
            if recent_errs > 15:
                print(f"!!! Aborting: {recent_errs}/20 recent requests failed — vLLM likely dead",
                      flush=True)
                break
        request_idx += 1
        pid, prompt = rng.choice(PROMPTS)
        metrics_before = bench_runner.fetch_metrics_snapshot()
        gpu_before = bench_runner.gpu_snapshot() if (time.time() - last_gpu_log) > 60 else None
        t_wall = time.time()
        try:
            res = bench_runner.call_streaming(prompt, max_tokens=args.max_tokens,
                                              temperature=args.temperature)
            err = None
        except Exception as e:
            res = {}
            err = str(e)
        metrics_after = bench_runner.fetch_metrics_snapshot()
        spec = bench_runner.parse_acceptance(metrics_before, metrics_after)
        gpu_after = bench_runner.gpu_snapshot() if gpu_before is not None else None
        if gpu_before is not None:
            last_gpu_log = time.time()

        elapsed_min = (time.time() - t_start) / 60
        row = {
            "request_idx": request_idx,
            "wall_ts": t_wall,
            "elapsed_min": elapsed_min,
            "prompt_id": pid,
            **res,
            "spec": spec,
            "gpu_before": gpu_before,
            "gpu_after": gpu_after,
            "error": err,
        }
        measurements.append(row)
        if err:
            print(f"  [{request_idx:>4}] @{elapsed_min:5.1f}min {pid}: ERR {err}", flush=True)
        else:
            length = spec.get("length_per_cycle")
            length_str = f"{length:.2f}/cyc" if length is not None else "n/a"
            print(f"  [{request_idx:>4}] @{elapsed_min:5.1f}min {pid}: "
                  f"TPOT={res['tpot_ms']:5.2f}ms accept={length_str}",
                  flush=True)

        time.sleep(args.sleep_between_s)

    # Trend analysis — use per-cycle acceptance length (more interpretable than per-token rate)
    valid = [m for m in measurements if not m.get("error")
             and m.get("spec") and m["spec"].get("length_per_cycle") is not None]
    trend = None
    if len(valid) > 10:
        xs = [m["elapsed_min"] for m in valid]
        ys = [m["spec"]["length_per_cycle"] for m in valid]
        result = linear_regression(xs, ys)
        if result:
            slope, intercept, r2, t_stat, p_value = result
            q4_size = max(len(ys) // 4, 1)
            first_q_mean = statistics.mean(ys[:q4_size])
            last_q_mean = statistics.mean(ys[-q4_size:])
            degradation_pct = ((last_q_mean - first_q_mean) / first_q_mean * 100
                               if first_q_mean > 0 else 0)
            trend = {
                "slope_per_minute": slope,
                "slope_per_hour": slope * 60,
                "intercept": intercept,
                "r_squared": r2,
                "t_statistic": t_stat,
                "p_value_two_sided": p_value,
                "significant_at_5pct": p_value < 0.05,
                "n": len(valid),
                "first_quartile_mean": first_q_mean,
                "last_quartile_mean": last_q_mean,
                "degradation_pct": degradation_pct,
                "monotonic_regression_detected": (slope < 0 and p_value < 0.05),
            }

    out = {
        "args": vars(args),
        "server_meta": server_meta,
        "trend": trend,
        "measurements": measurements,
    }
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"\n=== STABILITY TREND ===", flush=True)
    print(json.dumps(trend, indent=2), flush=True)
    print(f"\nWrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
