"""v3 clean A/B bench — addresses methodological concerns from v2-clean run:

  1. Streaming responses → TTFT separated from decode TPOT (decode-only is the
     fair MTP comparison, since prefill is unaffected by spec-decode method).
  2. Prefix caching DISABLED on serve side → no across-trial cache inflation.
  3. N=5 trials (was N=3) → tighter confidence on the delta.
  4. Full-prompt warmup → MTP draft heads + cuda graphs warm before measurement.
  5. Captures response text (preview + sha1 + length) → manual sanity check
     post-bench that MTP/no-MTP outputs are well-formed and content-equivalent
     (token-count divergence is a known vLLM non-determinism with chunked
     prefill — TPOT compares fairly even when token counts differ).

Reads VLLM_TAG env var ('no_mtp' or 'mtp'), writes
~/bench_clean_ab/results_v3_$VLLM_TAG.json.
"""
import asyncio
import hashlib
import json
import os
import statistics
import time
import urllib.request

import aiohttp

ENDPOINT = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "qwen36-awq"
TAG = os.environ.get("VLLM_TAG", "unknown")
OUT = f"/home/reachym/bench_clean_ab/results_v3_{TAG}.json"

PROMPTS = [
    "Why does the sky look blue? Answer in two sentences. /no_think",
    "Write a Python function fib(n) returning the first n Fibonacci numbers as a list. /no_think",
    "Explain TCP vs UDP in 3 concise bullet points. /no_think",
    "Give 5 numbered steps to cook firm tofu at home. /no_think",
    "Write a short haiku about debugging a memory leak at 2am. /no_think",
]

N_TRIALS_EXP1 = 5
N_TRIALS_EXP3 = 5
CONCURRENCIES = (1, 4, 8)


def _parse_sse_lines(raw_iter, t0):
    """Yield (delta_text, usage_dict) tuples and TTFT once."""
    ttft = None
    chunks = []
    ct = 0
    for raw in raw_iter:
        line = raw.strip()
        if not line.startswith(b"data:"):
            continue
        payload = line[5:].strip()
        if payload == b"[DONE]":
            break
        try:
            d = json.loads(payload)
        except Exception:
            continue
        choices = d.get("choices") or []
        if choices:
            delta = (choices[0].get("delta") or {}).get("content") or ""
            if delta:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                chunks.append(delta)
        if d.get("usage"):
            ct = d["usage"].get("completion_tokens") or ct
    return ttft, "".join(chunks), ct


def fire_stream_sync(prompt, max_tokens=200):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.5,
        "seed": 42,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(
        ENDPOINT, data=body, headers={"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as r:
        ttft, text, ct = _parse_sse_lines(r, t0)
    elapsed = time.perf_counter() - t0
    decode_s = max(elapsed - (ttft or 0.0), 0.0)
    decode_tpot_ms = (decode_s / max(ct - 1, 1)) * 1000 if ct > 1 else 0.0
    return {
        "ct": ct,
        "elapsed_s": elapsed,
        "ttft_s": ttft or 0.0,
        "decode_s": decode_s,
        "decode_tpot_ms": decode_tpot_ms,
        "tok_s": (ct / elapsed) if elapsed > 0 else 0.0,
        "decode_tok_s": ((ct - 1) / decode_s) if decode_s > 0 and ct > 1 else 0.0,
        "text_sha1": hashlib.sha1(text.encode()).hexdigest()[:16],
        "text_preview": text[:200],
        "text_len": len(text),
    }


async def fire_stream_async(session, prompt, max_tokens=200):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.5,
        "seed": 42,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.perf_counter()
    ttft = None
    chunks = []
    ct = 0
    async with session.post(
        ENDPOINT, json=body, timeout=aiohttp.ClientTimeout(total=180)
    ) as r:
        async for raw in r.content:
            line = raw.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                break
            try:
                d = json.loads(payload)
            except Exception:
                continue
            choices = d.get("choices") or []
            if choices:
                delta = (choices[0].get("delta") or {}).get("content") or ""
                if delta:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    chunks.append(delta)
            if d.get("usage"):
                ct = d["usage"].get("completion_tokens") or ct
    elapsed = time.perf_counter() - t0
    text = "".join(chunks)
    decode_s = max(elapsed - (ttft or 0.0), 0.0)
    return {
        "ct": ct,
        "elapsed_s": elapsed,
        "ttft_s": ttft or 0.0,
        "decode_s": decode_s,
        "tok_s": (ct / elapsed) if elapsed > 0 else 0.0,
        "text_sha1": hashlib.sha1(text.encode()).hexdigest()[:16],
        "text_preview": text[:120],
        "text_len": len(text),
    }


async def stress(N_concurrent, prompts):
    sem = asyncio.Semaphore(N_concurrent)
    async with aiohttp.ClientSession() as session:
        async def task(p):
            async with sem:
                return await fire_stream_async(session, p)

        tasks = [task(p) for p in prompts]
        t0 = time.perf_counter()
        results = await asyncio.gather(*tasks)
        total_time = time.perf_counter() - t0
    total_tokens = sum(r["ct"] for r in results)
    decode_tpots = []
    for r in results:
        if r["ct"] > 1 and r["decode_s"] > 0:
            decode_tpots.append((r["decode_s"] / (r["ct"] - 1)) * 1000)
    return {
        "concurrent": N_concurrent,
        "total_requests": len(prompts),
        "total_tokens": total_tokens,
        "wall_time_s": total_time,
        "aggregate_tok_s": (total_tokens / total_time) if total_time > 0 else 0,
        "mean_ttft_ms": statistics.mean(r["ttft_s"] for r in results) * 1000,
        "mean_decode_tpot_ms": statistics.mean(decode_tpots) if decode_tpots else 0,
        "per_request": results,
    }


def warmup():
    print("[warmup] full prompt set", flush=True)
    for p in PROMPTS:
        r = fire_stream_sync(p, max_tokens=50)
        print(
            f"  warmup ct={r['ct']} elapsed={r['elapsed_s']:.2f}s "
            f"ttft={r['ttft_s'] * 1000:.0f}ms decode_tpot={r['decode_tpot_ms']:.2f}ms",
            flush=True,
        )


def exp1_dialog(N_trials=N_TRIALS_EXP1):
    print(f"\n=== Exp 1 (dialog · streaming · N={N_trials}) ===", flush=True)
    trials = []
    for t in range(1, N_trials + 1):
        print(f"  trial {t}/{N_trials}", flush=True)
        trial_results = []
        for i, p in enumerate(PROMPTS, 1):
            r = fire_stream_sync(p)
            print(
                f"    p{i}: ct={r['ct']:3d} ttft={r['ttft_s'] * 1000:5.0f}ms "
                f"decode={r['decode_s']:.2f}s decode_tpot={r['decode_tpot_ms']:.2f}ms "
                f"tok_s={r['tok_s']:6.1f} sha1={r['text_sha1'][:10]}",
                flush=True,
            )
            trial_results.append(r)
        trials.append(trial_results)
    return {"trials": trials, "n_trials": N_trials, "n_prompts": len(PROMPTS)}


async def exp3_concurrent(concurrencies=CONCURRENCIES, N_trials=N_TRIALS_EXP3):
    print(
        f"\n=== Exp 3 (concurrent · streaming · concurrencies={concurrencies} · "
        f"N={N_trials}) ===",
        flush=True,
    )
    out = {}
    for C in concurrencies:
        prompts = (PROMPTS * 4)[:20]
        trial_results = []
        for t in range(1, N_trials + 1):
            print(f"  C={C} trial {t}/{N_trials}", flush=True)
            r = await stress(C, prompts)
            print(
                f"    aggregate={r['aggregate_tok_s']:6.1f} tok/s wall={r['wall_time_s']:.2f}s "
                f"ttft={r['mean_ttft_ms']:5.0f}ms decode_tpot={r['mean_decode_tpot_ms']:.2f}ms",
                flush=True,
            )
            trial_results.append(r)
        out[f"C{C}"] = {"trials": trial_results, "n_trials": N_trials}
    return out


async def main():
    print(f"=== bench_v3.py | TAG={TAG} ===", flush=True)
    print(f"  endpoint: {ENDPOINT}", flush=True)
    print(f"  model: {MODEL}", flush=True)
    print(f"  out: {OUT}", flush=True)
    warmup()
    e1 = exp1_dialog(N_trials=N_TRIALS_EXP1)
    e3 = await exp3_concurrent(
        concurrencies=CONCURRENCIES, N_trials=N_TRIALS_EXP3
    )
    out_obj = {
        "tag": TAG,
        "version": "v3",
        "config": {
            "n_trials_exp1": N_TRIALS_EXP1,
            "n_trials_exp3": N_TRIALS_EXP3,
            "concurrencies": list(CONCURRENCIES),
            "prompts": PROMPTS,
            "max_tokens": 200,
            "temperature": 0.5,
            "seed": 42,
            "streaming": True,
            "prefix_caching_disabled_in_serve": True,
        },
        "exp1_dialog": e1,
        "exp3_concurrent": e3,
    }
    with open(OUT, "w") as f:
        json.dump(out_obj, f, indent=2, ensure_ascii=False)
    print(f"\n=== written: {OUT} ===", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
