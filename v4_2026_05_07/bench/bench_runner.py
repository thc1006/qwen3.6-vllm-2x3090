#!/usr/bin/env python3
"""Rigorous vLLM bench runner — matched-flags methodology from v3.

Bench protocol (matched to v3 publication for direct comparability):
  - 5 prompts (sky / python / tcp_udp / tofu / haiku) — same as v3 baseline
  - N=5 trials per config (statistical validity)
  - 1 warmup discarded
  - Streaming endpoint → separate TTFT and TPOT
  - Per-request: SHA1 + preview (cross-check no degradation)
  - JSON dump per measurement + summary stats

Usage:
  python3 bench_runner.py --config-id k2_baseline --output bench_k2.json
  python3 bench_runner.py --config-id k1 --max-tokens 200 --output bench_k1.json

Designed to hit a vLLM server already running on http://127.0.0.1:8000.
Caller (orchestrator script) handles vLLM lifecycle.
"""
import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ENDPOINT = "http://127.0.0.1:8000/v1/chat/completions"
METRICS = "http://127.0.0.1:8000/metrics"
MODEL = os.getenv("BENCH_MODEL", "qwen36-awq")
TIMEOUT = 180

# v3 prompt set — DO NOT modify (preserves comparability)
PROMPTS = [
    ("sky",     "Why does the sky look blue? Answer in two sentences. /no_think"),
    ("python",  "Write a Python function fib(n) returning the first n Fibonacci numbers as a list. /no_think"),
    ("tcp_udp", "Explain TCP vs UDP in 3 concise bullet points. /no_think"),
    ("tofu",    "Give 5 numbered steps to cook firm tofu at home. /no_think"),
    ("haiku",   "Write a short haiku about debugging a memory leak at 2am. /no_think"),
]


def fetch_metrics_snapshot():
    """Capture vLLM /metrics — used for spec-decode acceptance rate."""
    try:
        with urllib.request.urlopen(METRICS, timeout=5) as r:
            return r.read().decode()
    except Exception:
        return ""


def parse_acceptance(metrics_before, metrics_after):
    """Diff before/after /metrics to extract per-request spec acceptance.

    vLLM 0.19.1 exposes (verified empirically 2026-05-07):
      vllm:spec_decode_num_drafts_total          # spec cycles (= proposals)
      vllm:spec_decode_num_draft_tokens_total    # total draft tokens proposed (k * drafts)
      vllm:spec_decode_num_accepted_tokens_total # accepted draft tokens
    NOTE: vllm:spec_decode_num_emitted_tokens_total does NOT exist in 0.19.1.

    Two useful ratios:
      ratio_per_token = accepted / draft_tokens   (0..1; "acceptance rate")
      length_per_cycle = accepted / drafts        (0..k; "acceptance length")
    """
    def grab(text, key):
        for line in text.splitlines():
            if line.startswith(key + " ") or line.startswith(key + "{"):
                try:
                    return float(line.rsplit(" ", 1)[-1])
                except ValueError:
                    continue
        return None

    def pair(key):
        a = grab(metrics_before, key)
        b = grab(metrics_after, key)
        if a is None or b is None:
            return None
        return b - a

    accepted = pair("vllm:spec_decode_num_accepted_tokens_total")
    drafts = pair("vllm:spec_decode_num_drafts_total")
    draft_tokens = pair("vllm:spec_decode_num_draft_tokens_total")

    out = {
        "accepted": accepted,
        "drafts": drafts,
        "draft_tokens": draft_tokens,
        "ratio_per_token": None,
        "length_per_cycle": None,
    }
    if accepted is not None and draft_tokens and draft_tokens > 0:
        out["ratio_per_token"] = accepted / draft_tokens
    if accepted is not None and drafts and drafts > 0:
        out["length_per_cycle"] = accepted / drafts
    return out


def call_streaming(prompt, max_tokens=200, temperature=0.0, seed=42):
    """Streaming call → measure TTFT (time to first token) and TPOT separately."""
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": seed,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(
        ENDPOINT, data=body, headers={"Content-Type": "application/json"}
    )
    full_text = []
    completion_tokens = 0
    t0 = time.perf_counter()
    ttft = None
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            payload = line[len("data: "):]
            if payload == "[DONE]":
                break
            try:
                ev = json.loads(payload)
            except json.JSONDecodeError:
                continue
            choices = ev.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                content = delta.get("content")
                if content:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    full_text.append(content)
            usage = ev.get("usage")
            if usage:
                completion_tokens = usage.get("completion_tokens", 0) or completion_tokens
    elapsed = time.perf_counter() - t0
    text = "".join(full_text)
    sha1 = hashlib.sha1(text.encode()).hexdigest()[:12]
    preview = text[:80].replace("\n", " ")
    decode_time = max(elapsed - (ttft or 0), 1e-9)
    tpot_ms = (decode_time / max(completion_tokens - 1, 1)) * 1000
    return {
        "ct": completion_tokens,
        "ttft_ms": (ttft or elapsed) * 1000,
        "tpot_ms": tpot_ms,
        "elapsed_s": elapsed,
        "tok_s": completion_tokens / elapsed if elapsed > 0 else 0,
        "sha1": sha1,
        "preview": preview,
    }


def gpu_snapshot():
    """nvidia-smi snapshot for memory + power + temp."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used,power.draw,temperature.gpu",
             "--format=csv,noheader,nounits"], timeout=5
        ).decode().strip()
        gpus = []
        for line in out.split("\n"):
            parts = [p.strip() for p in line.split(",")]
            gpus.append({
                "idx": int(parts[0]),
                "mem_mb": int(parts[1]),
                "power_w": float(parts[2]),
                "temp_c": int(parts[3]),
            })
        return gpus
    except Exception as e:
        return [{"error": str(e)}]


def run_config(config_id, max_tokens=200, n_trials=5, temperature=0.0):
    """Run N trials × len(PROMPTS) measurements for a single config."""
    print(f"\n=== Config {config_id} | trials={n_trials} | max_tokens={max_tokens} | temp={temperature} ===", flush=True)
    print("warmup...", flush=True)
    call_streaming("Hi", max_tokens=20, temperature=temperature)
    print("---", flush=True)

    measurements = []
    for trial in range(1, n_trials + 1):
        for pid, prompt in PROMPTS:
            metrics_before = fetch_metrics_snapshot()
            gpu_before = gpu_snapshot()
            t_wall = time.time()
            try:
                res = call_streaming(prompt, max_tokens=max_tokens, temperature=temperature)
                err = None
            except Exception as e:
                res = {}
                err = str(e)
            metrics_after = fetch_metrics_snapshot()
            gpu_after = gpu_snapshot()
            spec = parse_acceptance(metrics_before, metrics_after)

            row = {
                "config_id": config_id,
                "trial": trial,
                "prompt_id": pid,
                "wall_ts": t_wall,
                **res,
                "spec": spec,
                "gpu_before": gpu_before,
                "gpu_after": gpu_after,
                "error": err,
            }
            measurements.append(row)
            if err:
                print(f"  [trial {trial}] {pid}: ERR {err}", flush=True)
            else:
                accept_str = (f"{spec['length_per_cycle']:.2f}/cycle"
                              if spec.get('length_per_cycle') is not None else "n/a")
                print(f"  [trial {trial}] {pid}: ct={res['ct']:>3} "
                      f"TTFT={res['ttft_ms']:6.1f}ms TPOT={res['tpot_ms']:5.2f}ms "
                      f"tok/s={res['tok_s']:5.1f} sha1={res['sha1']} accept={accept_str}",
                      flush=True)
    return measurements


def summarize(measurements):
    """Per-config summary stats."""
    valid = [m for m in measurements if not m.get("error") and m.get("ct", 0) > 0]
    if not valid:
        return {"n": 0}
    tpot = [m["tpot_ms"] for m in valid]
    ttft = [m["ttft_ms"] for m in valid]
    tok_s = [m["tok_s"] for m in valid]
    accept_per_token = [m["spec"]["ratio_per_token"] for m in valid
                        if m.get("spec") and m["spec"].get("ratio_per_token") is not None]
    accept_per_cycle = [m["spec"]["length_per_cycle"] for m in valid
                        if m.get("spec") and m["spec"].get("length_per_cycle") is not None]
    sha_unique = len({m["sha1"] for m in valid})
    return {
        "n": len(valid),
        "tpot_ms": {
            "mean": statistics.mean(tpot),
            "median": statistics.median(tpot),
            "stdev": statistics.stdev(tpot) if len(tpot) > 1 else 0,
            "min": min(tpot),
            "max": max(tpot),
        },
        "ttft_ms": {
            "mean": statistics.mean(ttft),
            "median": statistics.median(ttft),
            "min": min(ttft),
            "max": max(ttft),
        },
        "tok_s": {
            "mean": statistics.mean(tok_s),
            "median": statistics.median(tok_s),
        },
        "spec_acceptance_per_token": {
            "n": len(accept_per_token),
            "mean": statistics.mean(accept_per_token) if accept_per_token else None,
            "median": statistics.median(accept_per_token) if accept_per_token else None,
        },
        "spec_acceptance_per_cycle": {
            "n": len(accept_per_cycle),
            "mean": statistics.mean(accept_per_cycle) if accept_per_cycle else None,
            "median": statistics.median(accept_per_cycle) if accept_per_cycle else None,
        },
        "unique_sha1": sha_unique,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-id", required=True, help="Label for this run")
    ap.add_argument("--output", required=True, help="JSON output path")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0.0 for greedy/deterministic; v3 used 0.5 (re-check before publishing)")
    args = ap.parse_args()

    server_meta = {}
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:8000/v1/models", timeout=5) as r:
            server_meta = json.loads(r.read())
    except Exception as e:
        server_meta = {"error": str(e)}

    measurements = run_config(args.config_id, args.max_tokens, args.trials, args.temperature)
    summary = summarize(measurements)

    out = {
        "config_id": args.config_id,
        "wall_start": measurements[0]["wall_ts"] if measurements else None,
        "wall_end": measurements[-1]["wall_ts"] if measurements else None,
        "args": vars(args),
        "server_meta": server_meta,
        "summary": summary,
        "measurements": measurements,
    }
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"\n=== SUMMARY {args.config_id} ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"\nWrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
