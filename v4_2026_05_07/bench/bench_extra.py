#!/usr/bin/env python3
"""Phase F + G: extra workload benches against production vLLM (port 8000).

Phase F: Tool-call workload — measures TPOT when LLM emits tool_calls (real
voice agent typical workload). 5 prompts × 5 trials, matched to v3 methodology.

Phase G: Long-context — measures TPOT vs context length (1k, 4k, 8k tokens of
prepended conversation history). Same 'sky' prompt at end, different ctx loads.

Hits production vllm-server.service WITHOUT stopping it. Reuses bench_runner
helpers for streaming + spec metrics.

Output:
  ~/bench_2026_05_07/phase_f_tool_call_*.json
  ~/bench_2026_05_07/phase_g_longctx_*.json
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

# Reuse bench_runner internals
sys.path.insert(0, "/home/reachym/dev/reachy-agent/robot/scripts")
import bench_runner

ENDPOINT = "http://127.0.0.1:8000/v1/chat/completions"
METRICS = "http://127.0.0.1:8000/metrics"
MODEL = "qwen36-awq"

# ----------------------------------------------------------------------------
# Phase F: tool-call workload
# ----------------------------------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "move_head",
            "description": "Move the robot head to face a direction.",
            "parameters": {
                "type": "object",
                "properties": {
                    "yaw": {"type": "number", "description": "horizontal -45 to +45, +right"},
                    "pitch": {"type": "number", "description": "vertical -30 to +30, +up"},
                },
                "required": ["yaw", "pitch"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "play_emotion",
            "description": "Display emotion on the robot's face.",
            "parameters": {
                "type": "object",
                "properties": {
                    "emotion": {"type": "string", "enum": ["happy", "sad", "surprised", "thinking"]}
                },
                "required": ["emotion"],
            },
        },
    },
]

TOOL_PROMPTS = [
    ("look_right",   "Look right"),
    ("look_up",      "Look up at the ceiling"),
    ("show_happy",   "Show me a happy face"),
    ("show_sad",     "Make a sad expression"),
    ("turn_left",    "Turn your head left"),
]


def call_streaming_with_tools(prompt, tools, max_tokens=200, temperature=0.0, seed=42):
    """Streaming call with tools; captures everything including tool_call deltas."""
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "tools": tools,
        "tool_choice": "auto",
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
    tool_calls_observed = []
    completion_tokens = 0
    t0 = time.perf_counter()
    ttft = None
    with urllib.request.urlopen(req, timeout=180) as r:
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
                tc = delta.get("tool_calls")
                if content:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    full_text.append(content)
                if tc:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    tool_calls_observed.append(tc)
            usage = ev.get("usage")
            if usage:
                completion_tokens = usage.get("completion_tokens", 0) or completion_tokens
    elapsed = time.perf_counter() - t0
    text = "".join(full_text)
    sha1 = hashlib.sha1((text + json.dumps(tool_calls_observed)).encode()).hexdigest()[:12]
    decode_time = max(elapsed - (ttft or 0), 1e-9)
    tpot_ms = (decode_time / max(completion_tokens - 1, 1)) * 1000
    return {
        "ct": completion_tokens,
        "ttft_ms": (ttft or elapsed) * 1000,
        "tpot_ms": tpot_ms,
        "elapsed_s": elapsed,
        "tok_s": completion_tokens / elapsed if elapsed > 0 else 0,
        "sha1": sha1,
        "n_tool_call_deltas": len(tool_calls_observed),
        "preview": text[:80].replace("\n", " "),
    }


def run_phase_f(output_path, n_trials=5, max_tokens=200, temperature=0.0):
    print(f"\n=== PHASE F — tool-call workload ===", flush=True)
    print(f"   trials={n_trials}, max_tokens={max_tokens}, temp={temperature}", flush=True)
    print("warmup...", flush=True)
    call_streaming_with_tools("Hi", TOOLS, max_tokens=20, temperature=temperature)

    measurements = []
    for trial in range(1, n_trials + 1):
        for pid, prompt in TOOL_PROMPTS:
            metrics_before = bench_runner.fetch_metrics_snapshot()
            t_wall = time.time()
            try:
                res = call_streaming_with_tools(prompt, TOOLS, max_tokens, temperature)
                err = None
            except Exception as e:
                res = {}
                err = str(e)
            metrics_after = bench_runner.fetch_metrics_snapshot()
            spec = bench_runner.parse_acceptance(metrics_before, metrics_after)
            row = {
                "trial": trial, "prompt_id": pid, "wall_ts": t_wall,
                **res, "spec": spec, "error": err,
            }
            measurements.append(row)
            if err:
                print(f"  [trial {trial}] {pid}: ERR {err}", flush=True)
            else:
                print(f"  [trial {trial}] {pid}: ct={res['ct']:>3} "
                      f"TTFT={res['ttft_ms']:6.1f}ms TPOT={res['tpot_ms']:5.2f}ms "
                      f"tok/s={res['tok_s']:5.1f} ntc={res['n_tool_call_deltas']} sha1={res['sha1']}",
                      flush=True)
    valid = [m for m in measurements if not m.get("error") and m.get("ct", 0) > 0]
    summary = {}
    if valid:
        summary = {
            "n": len(valid),
            "tpot_ms": {"mean": statistics.mean(m["tpot_ms"] for m in valid),
                        "stdev": statistics.stdev([m["tpot_ms"] for m in valid]) if len(valid) > 1 else 0},
            "ttft_ms": {"mean": statistics.mean(m["ttft_ms"] for m in valid)},
            "tok_s":   {"mean": statistics.mean(m["tok_s"]   for m in valid)},
            "ct_mean": statistics.mean(m["ct"] for m in valid),
            "tool_call_present": sum(1 for m in valid if m.get("n_tool_call_deltas", 0) > 0),
        }
    Path(output_path).write_text(json.dumps({
        "phase": "F_tool_call", "config": {"trials": n_trials, "max_tokens": max_tokens,
                                            "temperature": temperature, "tools": [t["function"]["name"] for t in TOOLS]},
        "summary": summary, "measurements": measurements,
    }, indent=2))
    print(f"\n=== PHASE F summary ===\n{json.dumps(summary, indent=2)}", flush=True)


# ----------------------------------------------------------------------------
# Phase G: long-context
# ----------------------------------------------------------------------------
def make_history(target_token_count):
    """Generate fake conversation history padded to ~target_token_count tokens."""
    history = []
    pad = "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor "
    # Each pad iteration ~10 tokens. Per turn pair (user+assistant) ~ 30-40 tokens.
    n_turns = max(target_token_count // 60, 0)
    for i in range(n_turns):
        history.append({"role": "user", "content": f"Q{i}: " + pad * 3})
        history.append({"role": "assistant", "content": f"A{i}: " + pad * 3})
    history.append({"role": "user",
                    "content": "Why does the sky look blue? Answer in two sentences. /no_think"})
    return history


def call_streaming_with_history(history, max_tokens=200, temperature=0.0, seed=42):
    body = json.dumps({
        "model": MODEL,
        "messages": history,
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
    prompt_tokens = 0
    t0 = time.perf_counter()
    ttft = None
    with urllib.request.urlopen(req, timeout=300) as r:
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
                prompt_tokens = usage.get("prompt_tokens", 0) or prompt_tokens
    elapsed = time.perf_counter() - t0
    text = "".join(full_text)
    sha1 = hashlib.sha1(text.encode()).hexdigest()[:12]
    decode_time = max(elapsed - (ttft or 0), 1e-9)
    tpot_ms = (decode_time / max(completion_tokens - 1, 1)) * 1000
    return {
        "ct": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "ttft_ms": (ttft or elapsed) * 1000,
        "tpot_ms": tpot_ms,
        "tok_s": completion_tokens / elapsed if elapsed > 0 else 0,
        "sha1": sha1,
        "preview": text[:80].replace("\n", " "),
    }


def run_phase_g(output_path, n_trials=3, temperature=0.0):
    print(f"\n=== PHASE G — long-context (1k/4k/8k/16k) ===", flush=True)
    targets = [200, 1000, 4000, 8000, 16000]
    print("warmup...", flush=True)
    call_streaming_with_history(make_history(200), max_tokens=20, temperature=temperature)

    measurements = []
    for target in targets:
        history = make_history(target)
        # Pre-flight: send 1 to get actual prompt_tokens
        for trial in range(1, n_trials + 1):
            metrics_before = bench_runner.fetch_metrics_snapshot()
            t_wall = time.time()
            try:
                res = call_streaming_with_history(history, max_tokens=200, temperature=temperature)
                err = None
            except Exception as e:
                res = {}
                err = str(e)
            metrics_after = bench_runner.fetch_metrics_snapshot()
            spec = bench_runner.parse_acceptance(metrics_before, metrics_after)
            row = {
                "trial": trial, "target_ctx": target, "wall_ts": t_wall,
                **res, "spec": spec, "error": err,
            }
            measurements.append(row)
            if err:
                print(f"  [ctx={target} t={trial}] ERR {err}", flush=True)
            else:
                print(f"  [ctx={target} t={trial}] prompt_tok={res['prompt_tokens']} "
                      f"ct={res['ct']} TTFT={res['ttft_ms']:7.1f}ms "
                      f"TPOT={res['tpot_ms']:5.2f}ms tok/s={res['tok_s']:5.1f}",
                      flush=True)
    # Aggregate by ctx target
    by_ctx = {}
    for target in targets:
        rs = [m for m in measurements if m.get("target_ctx") == target and not m.get("error")]
        if rs:
            by_ctx[target] = {
                "n": len(rs),
                "actual_prompt_tokens_mean": statistics.mean(m["prompt_tokens"] for m in rs),
                "ttft_ms_mean": statistics.mean(m["ttft_ms"] for m in rs),
                "tpot_ms_mean": statistics.mean(m["tpot_ms"] for m in rs),
                "tok_s_mean": statistics.mean(m["tok_s"] for m in rs),
            }
    Path(output_path).write_text(json.dumps({
        "phase": "G_longctx", "config": {"trials": n_trials, "temperature": temperature},
        "by_ctx": by_ctx, "measurements": measurements,
    }, indent=2))
    print(f"\n=== PHASE G summary ===\n{json.dumps(by_ctx, indent=2)}", flush=True)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2 or sys.argv[1] not in ("F", "G", "both"):
        print("Usage: bench_extra.py {F|G|both}")
        sys.exit(1)
    target = sys.argv[1]
    out = "/home/reachym/bench_2026_05_07"
    if target in ("F", "both"):
        run_phase_f(f"{out}/phase_f_tool_call.json")
    if target in ("G", "both"):
        run_phase_g(f"{out}/phase_g_longctx.json")
