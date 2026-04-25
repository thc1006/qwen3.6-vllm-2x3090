#!/usr/bin/env python3
"""Bench vLLM dialog throughput — single-stream baseline.

Reproduces the same prompt-set used for llama.cpp/Ollama baselines so numbers
are directly comparable.
"""
import json
import time
import urllib.request

ENDPOINT = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "qwen36-awq"

PROMPTS = [
    "Why does the sky look blue? Answer in two sentences. /no_think",
    "Write a Python function fib(n) returning the first n Fibonacci numbers as a list. /no_think",
    "Explain TCP vs UDP in 3 concise bullet points. /no_think",
    "Give 5 numbered steps to cook firm tofu at home. /no_think",
    "Write a short haiku about debugging a memory leak at 2am. /no_think",
]


def call(prompt, max_tokens=200):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.5,
        "seed": 42,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(
        ENDPOINT, data=body, headers={"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.loads(r.read().decode())
    elapsed = time.perf_counter() - t0
    ct = data.get("usage", {}).get("completion_tokens", 0)
    return ct, elapsed


def main():
    # Warmup
    print("warmup...")
    call("Hi", max_tokens=20)
    print("---")
    results = []
    for i, p in enumerate(PROMPTS, 1):
        ct, dt = call(p)
        tps = ct / dt if dt > 0 else 0
        print(f"[{i}/{len(PROMPTS)}] ct={ct:>3} elapsed={dt:5.2f}s tok/s={tps:5.1f}")
        results.append({"prompt_idx": i, "ct": ct, "elapsed_s": dt, "tok_s": tps})
    if results:
        mean = sum(r["tok_s"] for r in results) / len(results)
        mn = min(r["tok_s"] for r in results)
        mx = max(r["tok_s"] for r in results)
        print(f"\nmean={mean:.1f} min={mn:.1f} max={mx:.1f}")
    out = {"endpoint": ENDPOINT, "model": MODEL, "results": results}
    with open("vllm_dialog_bench.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
