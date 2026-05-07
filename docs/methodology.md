# Methodology

> **2026-05-07 — v4 methodology supplement.** [`v4_2026_05_07/README.md`](../v4_2026_05_07/README.md)
> documents the 9-phase factorial sweep with explicit Welch's t-test,
> p-values, confidence intervals, and a 3-way comparison (Phase A / J / J.2)
> to control for the `VLLM_USE_FLASHINFER_MOE_FP16` env-var confound when
> comparing vLLM 0.19.1 vs 0.20.1. The methodology below remains correct
> for the v1–v3 single-test framing; v4 adds matched-flag controls and
> per-comparison statistical tests on top.

## Question

Can a single vLLM engine on **2× consumer Ampere GPUs** serve **concurrent
vision+dialog** for an embodied robot, without dialog throughput collapsing
under VL prefill load?

This matters because in the alternative — running unified vision+dialog on a
single Ollama daemon — vision calls (6–31 s VL prefill on 35B-A3B MoE) block
dialog inference, producing TTFB spikes during conversation. The fix space:

1. Partition GPUs (one runs dialog, other runs vision) — sacrifices either
   model unification or vision quality
2. Continuous batching at the engine level — vLLM's design point. **This is
   what we're testing.**

## Hypothesis

vLLM's chunked-prefill scheduler with priority should preempt large vision
prefills with dialog decode tokens, keeping single-stream dialog tok/s
nearly intact while vision runs in the same engine. AWQ-Marlin is the right
quantization for Ampere (FP8 path is broken per
[vllm#40124](https://github.com/vllm-project/vllm/issues/40124)).

## Setup

- 2× RTX 3090 24 GB, no NVLink, PCIe Gen4 x8
- vLLM 0.19.1 + transformers 5.6.2 + torch 2.10.0+cu128
- AWQ-Marlin via `QuantTrio/Qwen3.6-35B-A3B-AWQ` (35.95B param, 3B active)
- TP=2, expert-parallel, chunked prefill, prefix caching
- Vision encoder TP=data (DP across 2 GPUs)
- max_model_len=32768, max_num_seqs=8, gpu_memory_utilization=0.90

Boot stats observed:
- Model load: 25.5 s
- VRAM per GPU after load: 11.62 GB
- torch.compile of dynamic shapes: 33.1 s
- Total cold-start to ready: ~70 s (one-shot)

## Test plan

### T1 · Dialog baseline (sequential)

5 prompts (sky, fib, TCP/UDP, tofu, haiku — same set as our llama.cpp/Ollama
benches for direct comparison), each `max_tokens=200`, `temperature=0.5`,
`seed=42`. Sequential calls. Mean tok/s.

Reference: Ollama+llama.cpp on 1× 3090 = ~107 tok/s. Spark NVFP4+MTP k=2 = 55.

### T2 · Vision baseline (sequential)

3 image+text calls. Synthetic 640×480 JPEG with a "Reachy Mini" label. 80 max
tokens output, "describe this image in 30 words". Mean wall-clock.

### T3 · Concurrent (the actual question)

Fire one vision request, sleep 200 ms, fire 5 dialog requests in parallel.
All overlap. Record:
- Each dialog request's tok/s
- Vision request's wall-clock under contention

Then compare:
- `T3.dialog_mean_tok_s / T1.dialog_mean_tok_s` → degradation %
- `T3.vision_wall_s / T2.vision_wall_s` → inflation %

## Pass criteria

| Metric | Threshold | Rationale |
|---|---|---|
| Dialog degradation | < 10% | User shouldn't notice slowdown when robot is "looking" |
| Vision inflation | < 30% | Vision is non-real-time; some slowdown is fine |
| No 5xx, no engine crash | – | Hard requirement |

## What we measure but don't gate on

- Memory pressure during concurrent (KV cache growth)
- Tail latency for individual dialog requests (some may be slower than mean)
- Effect of repeated runs (warm cache, prefix caching benefit)

## Reproducibility

All scripts in `scripts/`. JSON results dropped in `results/` with raw
per-request numbers. No averaging tricks. Full vLLM serve flags pinned
in `scripts/vllm_serve.sh`.
