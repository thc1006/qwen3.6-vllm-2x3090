# qwen3.6-vllm-2x3090 v4.0 — Comprehensive bench update (2026-05-07)

## What's new vs v3.0

v3.0 (2026-04-26) measured **vLLM MTP +27.5 % decode-rate gain vs no-spec** on dual RTX 3090 PCIe with Qwen3.6-35B-A3B-AWQ at matched flags + `--no-enable-prefix-caching` (the cache-OFF setting was already recommended for latency-focused MTP serving by [vLLM Recipes](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html) as of 2026-04-24; v3 quantified the gain and linked it to the #38182 L457 mechanism).

v4.0 deepens that single point into a **9-phase factorial sweep** (~3,000 measurements across 38 configurations) with statistical analysis.

## TL;DR — five new headline numbers

1. **MTP `num_speculative_tokens=3` is the production winner on this 2× RTX 3090 PCIe + AWQ-Marlin + vLLM 0.19.1 setup** (not k=1 from v3 or k=2 from old production on the same hardware). TTFT saves 26 ms versus k=2 (p<0.001 in all 4 power×temp cells); TPOT statistically equivalent. Note: [vLLM Recipes](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html) currently recommends k=2 as the cross-hardware default; the k=3 advantage here is validated only on the above stack and may not generalize to single-card / NVLink / HBM regimes.
2. **TP=1 categorically does not fit** Qwen3.6-35B-A3B-AWQ on a single 24 GB RTX 3090 with the production stack — even with `--enforce-eager`, `--gpu-memory-utilization=0.95`, `--max-num-seqs=1`, and `--max-model-len=4096`, OOM at 22.7 GB / 24 GB. Confirmed across 3 progressive configs and verified again with no spec decode at all. Dual 3090 TP=2 is mandatory.
3. **AWQ ≈ FP8 on Ampere SM 8.6** within statistical noise. At matched `gpu-memory-utilization=0.92`, all 4 (power × temp) AWQ vs FP8 TPOT comparisons p > 0.6, NS at α=0.05. The earlier "FP8 +2.7%" was confounded by 0.85 vs 0.92 mem-util asymmetry.
4. **MTP shows NO monotonic acceptance regression in 60-min sustained load** at either 350 W or 220 W. p_one_sided > 0.10, NS at α=0.05. Counter-evidence to vllm-project/vllm#41838 (H200 Eagle3) for MTP on Ampere.
5. **220 W power cap gives +11% perf/W** vs 350 W (0.434 vs 0.391 tok/s/W) with negligible throughput loss. Direction matches the known 3090 sweet-spot curve; this is the first dual-3090 vLLM MoE datapoint.

## Methodology summary

- Hardware: NVIDIA RTX 3090 24 GB × 2 (PCIe Gen4 ×16, no NVLink). GPU 0 driver-default 390 W (Founders); GPU 1 default 350 W (stock). Both capped at test power via `nvidia-smi -i N -pl`. CUDA 12.x, vLLM 0.19.1.
- Model: `QuantTrio/Qwen3.6-35B-A3B-AWQ` (community AWQ Q4 of Qwen/Qwen3.6-35B-A3B). Plus official `Qwen/Qwen3.6-35B-A3B-FP8`.
- Prompts: same 5-prompt set as v3 (sky / python / tcp_udp / tofu / haiku, all with `/no_think`). N=5 trials × 5 prompts = 25 measurements per config.
- Streaming SSE, TTFT and TPOT separated, spec-acceptance from `/metrics` differential. All under `--no-enable-prefix-caching` (prefix cache off — see v3 for why).

## Results — Phase A: k sweep (k=1 / k=2 / k=3 × power × temp = 12 configs)

### Aggregate per config (mean over 25 measurements per cell)

| config | TPOT (ms) | TTFT (ms) | tok/s | accept/cycle | mean GPU power (W, both cards) | tok/s/W |
|---|---:|---:|---:|---:|---:|---:|
| p350_k1_t0.0 | 5.471 | 82 | 150 | 0.87 / 1 (87 %) | 437 | 0.344 |
| p350_k2_t0.0 | 4.957 | 75 | 169 | 1.50 / 2 (75 %) | 449 | 0.378 |
| **p350_k3_t0.0** | **4.988** | **50** | **183** | **1.94 / 3 (65 %)** | 470 | 0.391 |
| p350_k3_t0.5 | 5.238 | 53 | 175 | 2.01 / 3 (67 %) | 475 | 0.368 |
| p220_k1_t0.0 | 5.581 | 75 | 150 | 0.86 | 410 | 0.366 |
| p220_k2_t0.0 | 5.052 | 76 | 167 | 1.50 | 408 | 0.408 |
| **p220_k3_t0.0** | **5.107** | **49** | **180** | **1.94** | **415** | **0.434** ⭐ |
| p220_k3_t0.5 | 5.328 | 52 | 172 | 2.03 | 422 | 0.407 |

(Full 12-config table + per-prompt breakdown in [analysis/phase_a_full.md].)

### k sweep statistical summary

| metric | k=2 vs k=3 (Welch's t-test, all 4 power×temp) |
|---|---|
| TPOT | NS in all 4 cells (p > 0.18) |
| **TTFT** | **p < 0.001 in all 4 cells (k=3 saves 25-29 ms)** |
| tok/s | p < 0.05 at temp=0.5 (k=3 wins ~21 tok/s); NS at temp=0.0 |

### Per-prompt variance (p350_k3_t0.0)

| prompt | TPOT (ms) | tok/s | accept/cycle | unique sha1 |
|---|---:|---:|---:|---:|
| haiku | 7.11 | 112 | 1.14 | 2 |
| python | 3.66 | 219 | 2.86 | 1 |
| sky | 4.73 | 184 | 1.93 | 1 |
| tcp_udp | 4.70 | 200 | 1.91 | 2 |
| tofu | 4.75 | 202 | 1.86 | 2 |

**Aggregate "183 tok/s" hides 2× variance across prompts** — voice agent workloads (short, conversational) bias toward haiku-like prompts, expected production throughput closer to 140-150 tok/s.

### MTP cross-k SHA1 overlap (lossless empirical check)

At temp=0.0, seed=42, every prompt has at least one shared SHA1 across k=1, k=2, k=3 within the same temp+power cell. vLLM documents MTP as "Algorithmically lossless" with float-precision caveats; our SHA1-level cross-k empirical confirmation on Qwen3.6-35B-A3B is the first public datapoint we are aware of.

## Results — Phase B: TP=2 vs TP=1 (categorical fail)

| attempt | flags | result | OOM at |
|---|---|---|---|
| 1 | TP=1, k=3, max_len=8192, num_seqs=4, mem-util=0.85 | OOM | 22.71 GB |
| 2 | TP=1, k=3, **+ enforce-eager + max_len=4096 + num_seqs=1 + mem-util=0.95 + PYTORCH_ALLOC_CONF=expandable_segments** | OOM | 22.71 GB |
| 3 | TP=1, **k=1** (smaller MTP head), all tweaks above | OOM | 22.57 GB |
| 4 | TP=1, **NO spec decoding**, all tweaks above | engine init failed | — |

→ **Single 3090 cannot fit Qwen3.6-35B-A3B AWQ + production-class chunked-prefill / tool-call / reasoning parser stack regardless of spec config.** The fixed overhead (~22.5 GB) plus a 970 MB CUDA workspace allocation exceeds 24 GB.

→ Dual 3090 TP=2 is **architecturally required**, not a performance choice. Hypothesis "reallocate one card to vision/Whisper" is **refuted**.

## Results — Phase C+H+I: AWQ vs FP8

### Phase H: AWQ at matched gpu-memory-utilization=0.92 (control vs FP8)

| config | AWQ@0.92 TPOT | FP8@0.92 TPOT | diff | p | significance |
|---|---:|---:|---:|---:|---|
| p350_t0.0 | 4.968 | 4.853 | +0.115 | 0.72 | NS |
| p350_t0.5 | 5.184 | 5.342 | −0.159 | 0.63 | NS |
| p220_t0.0 | 5.110 | 4.959 | +0.151 | 0.63 | NS |
| p220_t0.5 | 5.340 | 5.290 | +0.050 | 0.87 | NS |

**Conclusion: AWQ ≈ FP8 on Ampere SM 8.6 within measurement noise** (Welch's t-test, p > 0.6 in all four cells). The earlier apparent "FP8 +2.7 %" was confounded by gpu-mem-util asymmetry (0.85 vs 0.92).

### Phase I: FP8 mem-util sweep (find minimum that fits with Whisper on GPU 1)

Sweep tested only at **350 W power, temp = 0.0** (not full power × temp matrix).

| gpu-mem-util | result |
|---|---|
| 0.85 | OOM (KV cache allocation) |
| 0.86 | OOM |
| 0.88 | OOM |
| **0.90** | **FITS** (181 tok/s, n=25) |
| 0.92 | FITS (187 tok/s, n=25) |

→ **FP8 minimum mem-util = 0.90 with Whisper holding 1.4 GB on GPU 1.** Below that, vLLM fails KV-cache allocation. AWQ has no such requirement (production ships at 0.85). For users running FP8 in production-class voice-agent setups with auxiliary GPU loads, set `--gpu-memory-utilization` to ≥ 0.90 explicitly.

**Caveat**: minimum was confirmed at 350 W only. At 220 W the minimum should be the same (mem-util is decoupled from power), but unverified.

## Results — Phase E: 60-minute sustained-load stability

Both 350 W and 220 W with k=3, temp=0.5, ~25 req/min sustained for 60 min, ~1490 measurements per run.

| metric | 350 W | 220 W |
|---|---:|---:|
| n | 1493 | 1489 |
| slope_per_hour | −0.0503 | −0.0544 |
| r² | 0.0008 | 0.0009 |
| t-statistic | −1.11 | −1.19 |
| p_one_sided | 0.133 | 0.118 |
| first quartile mean | 1.992 | 1.979 |
| last quartile mean | 1.959 | 1.948 |
| degradation % | −1.63 % | −1.56 % |
| **monotonic_regression_detected** | **false** | **false** |

→ **No statistically significant acceptance-length regression over 60 min at either power.** This is counter-evidence to vllm-project/vllm#41838 (H200 4× setup with Eagle3) for MTP-on-Ampere — but note our test is MTP, not Eagle3. Plausible mechanism: MTP shares the main model's hidden state, no separate draft KV cache to drift.

## Results — Phase F+G: tool-call and long-context

### Phase F: tool-call workload

5 tool-call prompts × 5 trials, with `move_head` + `play_emotion` schemas, `tool_choice=auto`. 25/25 produced tool_calls.

| metric | tool-call (Phase F) | text (Phase A k=3 t=0.0) | diff | p |
|---|---:|---:|---:|---:|
| TPOT (ms) | 3.04 | 4.99 | −1.95 | < 0.0001 *** |
| **TTFT (ms)** | **138** | **50** | **+88** | (high) |
| **tok/s** | **144** | **183** | **−39** | (lower) |
| accept/cycle | 2.000 | 1.940 | +0.060 | 0.59 NS |
| ct (mean) | 35 | 200 | (different) | n/a |

**Three competing signals — not simply "tool-call faster":**

1. **TPOT (per-token decode after first chunk) is significantly faster for tool-call** (3.04 vs 4.99 ms, p < 0.0001). Plausibly because structured-output paths avoid sampling overhead, or because tool_call output reaches "stop" condition quickly so we measure fewer trailing tokens.

2. **TTFT is ~3× higher for tool-call** (138 vs 50 ms). Tool-call validation / parsing on first chunk adds significant first-token latency.

3. **End-to-end throughput (tok/s) is LOWER for tool-call** (−21%). The TTFT increase outweighs the TPOT improvement for short outputs (35 tokens vs 200).

**Acceptance rate is NOT significantly different** (2.000 vs 1.940 / cycle, p = 0.59) — refuting the "structured tokens → higher MTP acceptance" hypothesis as a mechanism for the TPOT delta.

**Production implication for voice agents**: tool-call latency for "Look right" / "Show happy" type prompts is ~240 ms total wall-clock (~138 ms TTFT + 35 tokens × 3 ms). That's snappy and acceptable. The tok/s comparison vs text is misleading because outputs differ in length.

### Phase G: long-context decode scaling

n = 3 trials per ctx target (limited statistical power; trends only — confidence intervals not reported).

| target ctx (tokens) | actual prompt_tokens | TTFT (ms) | TPOT (ms) | tok/s |
|---:|---:|---:|---:|---:|
| 200 | ~25 | ~50 | 5.0 | 180 |
| 1000 | ~1300 | ~330 | 5.5 | 150 |
| 4000 | ~5950 | 1125 | 6.69 | 37 |
| 8000 | ~12000 | 2259 | 9.46 | 20 |
| 16000 | ~24300 | 4800 | 12.47 | 11 |

→ Decode TPOT scales **steeply with context** on dual 3090 PCIe TP=2. ~+90 % at 12 k tokens, **+150 % at 24 k**. TP=2 PCIe inter-GPU communication overhead becomes dominant past ~6 k tokens. For voice agents (typical context < 1 k tokens) this is a non-issue, but long sessions accumulating tool/vision history will visibly degrade.

**Caveat**: with n=3 per cell, the curve shape is informative but specific point estimates have high variance.

## perf/W comparison (220 W vs 350 W at MTP k=3 t=0.0)

| power | mean draw (both cards, W) | tok/s | tok/s/W |
|---|---:|---:|---:|
| 350 | 470 | 183 | 0.391 |
| **220** | **415** | **180** | **0.434** (+11.0 %) |

→ **For sustained voice-agent workloads, 220 W is strictly better in perf/W with negligible throughput loss.** Generalizes the existing 3090 sweet-spot literature (Himesh QwQ-32B 4× setup, qwertyforce blog, Puget Systems MaxQ) to dual-3090 + Qwen3.6-A3B + vLLM specifically.

## Caveats and methodology notes

0. **Acceptance rate semantics**: vLLM's `vllm:spec_decode_num_accepted_tokens_total` counts **accepted draft tokens only**, NOT including the verifier's own emitted token. For k=3, "1.94 / cycle" means 1.94 draft tokens accepted out of 3 max; total tokens delivered per cycle = 1 (verifier) + 1.94 (drafts) = 2.94. Theoretical maximum tokens/cycle is k+1 = 4. Our k=3 typical ratio is 2.94/4 = 73.5 % of the theoretical ceiling.

1. **Aggregate "183 tok/s" baseline mixes 5 prompts with 2× per-prompt TPOT variance.** Voice agent workloads (haiku-like) realistically run at 140-150 tok/s.
2. **Tool-call vs text comparison** confirms TPOT delta but **refutes** the hypothesized acceptance-rate mechanism. The cause is open.
3. **AWQ vs FP8** required matched gpu-mem-util (0.92) for fair comparison; original Phase C had 0.85 vs 0.92 confound.
4. **MTP cross-k SHA1 verification** holds at temp=0.0 with seed=42; we observe within-cell non-determinism (6-8 unique SHA1 per 25 measurements) attributable to vLLM chunked-prefill non-determinism (known issue), not MTP.
5. **Phase E p-values are computed using normal approximation** (df ≈ 1490, t-distribution ≈ N(0,1)). One-sided test (slope < 0 hypothesis) reported; two-sided is also NS.
6. **Whisper held 1.4 GB on GPU 1 throughout** (production-equivalent setup). Affects FP8 minimum gpu-mem-util but not AWQ.
7. **Phase J + J.2: vLLM 0.20.1 vs 0.19.1 with backend-confound control.**

   `VLLM_USE_FLASHINFER_MOE_FP16=1` (production env) raises `NotImplementedError` on 0.20.1 ("no FlashInfer unquantized MoE backend supports the configuration"), so 0.20.1 must be run with this env var unset. To separate version effect from backend effect we ran a 3-way comparison:

   | run | vLLM | env `VLLM_USE_FLASHINFER_MOE_FP16` | TPOT (ms) | TTFT (ms) | tok/s |
   |---|---|---|---:|---:|---:|
   | Phase A k=3 t=0.0 | 0.19.1 | =1 (production) | 4.988 | 49.95 | 183.5 |
   | Phase J.2 | 0.19.1 | unset (matched J) | 4.984 | 53.05 | 182.0 |
   | Phase J | 0.20.1 | unset (required) | 4.680 | 61.90 | 191.1 |

   **(a) Backend effect (A vs J.2, 0.19.1 with vs without FP16 MoE):** all 3 metrics NS (p > 0.57). Conclusion: `VLLM_USE_FLASHINFER_MOE_FP16` is essentially a **no-op for AWQ-Marlin Qwen3.6 on Ampere SM 8.6**.

   **(b) Version effect (J.2 vs J, both with FP16 MoE off):** all 3 metrics NS (p > 0.34). Conclusion: **vLLM 0.20.1 is statistically equivalent to 0.19.1 on this path**.

   **Practical recommendation**: `vLLM 0.20.1` is safe to upgrade for production AWQ-Marlin Qwen3.6 dual 3090, after unsetting `VLLM_USE_FLASHINFER_MOE_FP16` (or removing it from the systemd unit). The #41306 MoE-backend regression that hits Mixtral 8×7B / DeepSeek-V4 / NVFP4 paths does NOT manifest on AWQ-Marlin path for this model on Ampere. Note: vllm-project/vllm#38182 (MTP × prefix-caching block-drop behaviour for Qwen3.5/3.6 A3B) is by-design — a vLLM maintainer later [confirmed it is "not really a bug"](https://github.com/vllm-project/vllm/issues/38182#issuecomment-4500246930), just expected behaviour (the MTP head is one token ahead and can't reuse the last token's KV); the `--no-enable-prefix-caching` recommendation — [documented by vLLM Recipes since 2026-04-24 for latency-focused MTP serving](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html) and connected to the #38182 L457 mechanism in our v3.0 retest — therefore applies permanently, not pending a fix.

## Reproduction

All bench scripts in `bench/`:
- `bench_runner.py` — N=5 × 5-prompt streaming bench, captures TTFT/TPOT/acceptance per request
- `bench_stability.py` — long-time monotonic-regression check with Welch's t and one-sided p-value
- `bench_extra.py` — Phase F (tool-call) and Phase G (long-context) modules
- `bench_orchestrate.sh` — vLLM lifecycle (preflight cleanup, power cap, launch, measure, stop) for all 9 phases

Raw JSON outputs in `data/2026-05-07/`:
- 12 phase_a, 4 phase_b, 8 phase_c, 4 phase_h, 1 phase_i, 2 phase_e, 1 phase_f, 1 phase_g
- Plus llama.cpp logs from cross-engine 3090 reference (single-card baseline + draft-spec).

Total: ~3,000 measurements, ~30 MB of structured JSON.

## Cross-references

- v3 publication: vLLM MTP +27.5 % NET WIN on dual 3090 PCIe (still valid, k=1 specifically — k=3 now recommended)
- Sister repo `qwen3.6-speculative-decoding-rtx3090` v3.0: llama.cpp draft-spec NET LOSS reproduction + first 3090 + DFlash + Q4 datapoint (NET LOSS −44.6 %)

## License

Apache 2.0 (data + code). Findings free to cite.
