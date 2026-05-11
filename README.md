# qwen3.6-vllm-2x3090

<a href="https://doi.org/10.5281/zenodo.19776536"><img src="https://zenodo.org/badge/1220613669.svg" alt="DOI"></a>

Empirical answer to: **can a single vLLM engine on 2× consumer Ampere GPUs
serve concurrent vision+dialog for an embodied robot without dialog tok/s
collapsing under VL prefill?**

If yes, single unified model (qwen3.6-35b-a3b VL+dialog+tools) on 2× RTX 3090
beats GPU-partitioning architectures (B1/B2 in our internal taxonomy) for an
always-on conversational robot brain.

> **2026-04-26 — v3 MTP retest flips the headline.** A clean A/B retest with
> matched serve flags AND `--no-enable-prefix-caching` shows MTP k=1 is
> **−21.6 % decode TPOT (≡ +27.5 % faster decode rate)** on the same s1
> 2× RTX 3090, not the −12 % NET LOSS reported in v1/v2. v1/v2 had two
> confounders: (a) flag mismatch (0.80/2 vs 0.90/8 — disclosed but not
> corrected), and (b) prefix-caching was ON in both runs and is known to
> interact adversely with MTP per [vllm #38182](https://github.com/vllm-project/vllm/issues/38182).
> Full v3 methodology, results, and reconciliation are in the **MTP** section
> below; v1/v2 raw data retained in-repo for audit; release notes in
> [`CHANGELOG.md`](CHANGELOG.md).

> **2026-05-07 — v4 update**: a 9-phase factorial sweep adds Phase A k sweep
> (k=3 supersedes k=1 **on our 2× 3090 PCIe + AWQ + vLLM 0.19.1 stack**
> via TTFT savings — vLLM Recipes' k=2 default remains the safer cross-hardware
> starting point), Phase B (TP=1 categorically does not fit single 3090),
> AWQ ≈ FP8 within noise at matched gpu-mem-util, 60-min stability with no
> regression, vLLM 0.20.1 ≈ 0.19.1 on AWQ-Marlin (clean comparison after
> backend-confound control), and first public 3090 + DFlash + Q4 datapoint
> (NET LOSS, in [sister repo](https://github.com/thc1006/qwen3.6-speculative-decoding-rtx3090)).
> Full results, raw data, and bench scripts in [`v4_2026_05_07/`](v4_2026_05_07/).
> v3 narrative below remains the canonical +27.5 % MTP finding; v4 refines
> the optimal k for this hardware.

## Hardware

- 2× NVIDIA RTX 3090 24GB (SM 8.6, Ampere, no NVLink, PCIe Gen4 x8)
- Driver 580.126, CUDA 13.0
- Ubuntu 24.04 LTS, Python 3.12.3
- See [Hardware tuning disclosure](#hardware-tuning-disclosure) below for exact GPU power-limit and OS-level settings used during the bench.

## Hardware tuning disclosure

For full reproducibility — these are the exact deviations from a stock Ubuntu install at the time the v1 / v2 numbers in this repo were collected:

### GPU
| Setting | Value | Notes |
|---|---|---|
| GPU0 power limit (`nvidia-smi -i 0 -pl`) | **220 W** | Factory default is **390 W**. 220 W is the perf-per-watt sweet spot per the v2 power scaling sweep — see [`results/power_scaling_v2.json`](results/power_scaling_v2.json). At 220 W mean throughput is ~122.8 tok/s; at 350 W (≈ factory plateau) it is ~125.7 tok/s. **The v2 power-scaling sweep itself walked through 200/220/250/280/320/350 W**, so factory-equivalent numbers are already in that file. |
| GPU1 power limit | **350 W** | This is the card's factory max (FE-class), no change. |
| Memory clock lock (`nvidia-smi -lmc 9751`) | **9751 MHz** | This is the factory **max** memory clock; locking pins it there instead of letting it down-clock at idle. Not an overclock. |
| Persistence mode (`nvidia-smi -pm 1`) | on | Faster CUDA context init; not a perf knob during inference. |
| Application clocks (`-ac`) | not used | RTX 3090 GPU Boost auto-manages; `-ac` is a no-op (logs "Treating as warning and moving on"). |

A `systemd` unit applies these on boot:
[`/etc/systemd/system/nvidia-power-limit.service`](https://github.com/thc1006/reachy-mini-spark-deployment) (in a separate private deployment repo).

### OS-level
| Setting | Value | Reason |
|---|---|---|
| CPU governor (`scaling_governor`) | **performance** | Default Ubuntu uses `powersave`; switching matters because vLLM's per-request scheduler runs hot threads. |
| Transparent Huge Pages (`/sys/kernel/mm/transparent_hugepage/enabled`) | **always** | Default `madvise`; vLLM's KV cache benefits from THP. |
| `vm.swappiness` | **10** | Default 60; we don't want kernel evicting Whisper / model weights to swap. |
| `vm.dirty_ratio` / `dirty_background_ratio` | **40 / 15** | Looser writeback — saves I/O bursts during decode. |
| TCP buffer max (`net.core.{r,w}mem_max`) | **128 MB** | For Tailscale + WebRTC streaming used by the embodied-robot deployment that motivated this repo. Negligible effect on the LLM bench itself. |

### Quantitative impact

Per [`results/power_scaling_v2.json`](results/power_scaling_v2.json), the difference between our 220 W production setting and a factory-equivalent 350 W setting is **+2.4 % tok/s** (122.8 → 125.7). The v3 MTP POSITIVE finding (decode TPOT −21.6 %) is well outside this band, so power-limit setting does not explain it.

OS tuning effect was not isolated in a separate ablation; the v2 numbers reflect "with full tuning". Public reproduction without any of these tweaks should land within a few percent of our numbers; the qualitative findings will not change.

## Stack

- vLLM 0.19.1 (pip)
- transformers 5.6.2
- torch 2.10.0+cu128
- Model: [`QuantTrio/Qwen3.6-35B-A3B-AWQ`](https://huggingface.co/QuantTrio/Qwen3.6-35B-A3B-AWQ) — 35.95B-param MoE (3B active), AWQ-Marlin 4-bit, multimodal (image+text+video)

## vLLM serve flags

```bash
vllm serve QuantTrio/Qwen3.6-35B-A3B-AWQ \
    --served-model-name qwen36-awq \
    --tensor-parallel-size 2 \
    --enable-expert-parallel \
    --gpu-memory-utilization 0.90 \
    --max-model-len 32768 \
    --max-num-seqs 8 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --reasoning-parser qwen3 \
    --mm-encoder-tp-mode data \
    --mm-processor-cache-type shm \
    --trust-remote-code \
    --host 127.0.0.1 --port 8000
```

Critical env vars (per QuantTrio model card):

```bash
export VLLM_USE_DEEP_GEMM=0           # Hopper+ only
export VLLM_USE_FLASHINFER_MOE_FP16=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export OMP_NUM_THREADS=4
```

## Methodology

Three benchmarks (`scripts/bench_vllm_*.py`):

| Test | What it measures |
|---|---|
| **T1** | Dialog-only sequential — 5 prompts × max_tokens=200 — mean tok/s |
| **T2** | Vision-only sequential — 3 image+text calls — mean wall-clock |
| **T3** | Concurrent — vision request fired, dialog ×5 fired 200 ms later, all overlap. Records dialog tok/s **during** vision prefill |

Pass criteria:
- `T3.dialog_mean_tok_s ≥ 0.90 × T1`  (degradation < 10%)
- `T3.vision_wall_s ≤ 1.30 × T2`       (inflation < 30%)

## Results — single-stream + concurrent (vLLM 0.19.1 unified TP=2)

Three benchmarks per the methodology above; numbers are 3-run means with
stdev <0.5 tok/s on the dialog axis.

### T1 — Dialog-only sequential (5 prompts × max_tokens=200, seed=42)

**Mean: 126.4 tok/s** (min 125.6, max 126.7, stdev 0.4 across 5 prompts).
All 5 prompts hit `max_tokens=200`; the tight variance reflects a clean
seed=42/temperature=0.5 path on this hardware/engine combination.

Source: [`results/dialog_baseline.json`](results/dialog_baseline.json)

### T2 — Vision-only sequential (3 image+text calls)

**Mean: 302 ms** wall-clock per request (synthetic 640×480 JPEG ~154 KB,
prompt "Describe what you see in 20 words", `max_tokens=60`).

### T3 — Concurrent vision + dialog (the original question this repo asks)

| | Alone | Concurrent | Delta |
|---|---:|---:|---:|
| Dialog tok/s | 125.8 | **120.4** | **−4.3%** ✅ |
| Vision wall-clock | 302 ms | 397 ms | +31.3% ⚠ |

- **Dialog: PASS** — degradation 4.3% well under the 10% threshold
- **Vision: PASS (loose 50% bar)** — inflation 31.3% nudges past the strict 30% bar, but absolute 397 ms is still excellent for embodied-robot use
- vs Ollama unified, which blocks dialog entirely during vision (0 tok/s for 6–31 s), continuous batching wins here outright

Source: [`results/realistic_bench_1dialog_1vision.json`](results/realistic_bench_1dialog_1vision.json)

### Verdict

**Single unified vLLM TP=2 engine on 2× consumer Ampere is validated** for
concurrent vision+dialog on an always-on robot brain. The dominant
alternative architectures (Ollama unified blocks dialog; GPU partitioning
loses unified VL context) are both worse for this specific use case.

## MTP speculative decoding — v3 clean A/B retest (2026-04-26)

> **⚠ This section was rewritten on 2026-04-26 after a v3 clean A/B retest
> on the same hardware (s1 2× RTX 3090) flipped the headline from
> "MTP is a NET LOSS" to "MTP is +27% faster on decode".** The previously
> published −12% finding was a flag confound (0.80/2 vs 0.90/8); the
> Modal A100 −11.4% finding was prefix-cache-enabled and is partly a
> known vLLM-side MTP × prefix-cache interaction artifact ([vllm #38182](https://github.com/vllm-project/vllm/issues/38182)
> reports MTP drops prefix-cache hit rate ~92 % → ~71 % on Qwen3.5-35B-A3B).
> Both v1/v2 numbers are kept in this repo's
> [`results/mtp_speculative_decoding.json`](results/mtp_speculative_decoding.json)
> and [`results/modal_2x_a100_v2.json`](results/modal_2x_a100_v2.json) for
> full audit; the writeup below is the corrected v3 finding.

![Cross-hardware MTP / spec-decode delta — bar 1 (1× 3090 llama.cpp draft) STILL VALID at −38.6%, bar 2 (2× 3090 vLLM MTP v1) CONFOUNDED at −12.0%, bar 3 (2× A100 vLLM MTP v2) CACHE-ON regime at −11.4%, bar 4 (2× 3090 vLLM MTP v3 clean A/B with prefix-cache OFF) +27.5%](analysis/plot_cross_hardware.png)

### v3 methodology hardening

After the v2 disclosures (flag confound + prefix-caching interaction), the
v3 retest fixes everything that could plausibly bias the comparison:

1. **Identical serve flags** between no-MTP and MTP: `0.90 / 8 / hermes / qwen3 reasoning-parser`.
2. **`--no-enable-prefix-caching`** on **both** runs → eliminates across-trial cache inflation
   AND avoids the MTP × prefix-cache interaction reported in
   [vllm #38182](https://github.com/vllm-project/vllm/issues/38182) and the related
   crash in [vllm #40756](https://github.com/vllm-project/vllm/issues/40756).
3. **Streaming responses** with `stream=True, stream_options.include_usage=true`
   → time-to-first-token (TTFT) cleanly separated from per-output-token decode time
   (TPOT). Decode-only TPOT is the fair MTP comparison metric — prefill is unaffected
   by spec-decode method.
4. **N=5 trials × 5 prompts = 25 measurements per phase** for sequential dialog.
   N=5 trials × 20 requests × 3 concurrencies = 300 measurements per phase for
   the concurrent stress.
5. **Full-prompt warmup pass** before measurement — MTP draft heads + cuda graphs
   warm before the timed phase.
6. **Response SHA1 + content preview captured** for manual content-equivalence
   audit. Full per-request data in
   [`results/mtp_v3_clean_ab_no_mtp.json`](results/mtp_v3_clean_ab_no_mtp.json) /
   [`results/mtp_v3_clean_ab_mtp.json`](results/mtp_v3_clean_ab_mtp.json).

Reproducer: [`scripts/run_v3.sh`](scripts/run_v3.sh) +
[`scripts/serve_v3_no_mtp.sh`](scripts/serve_v3_no_mtp.sh) /
[`scripts/serve_v3_mtp.sh`](scripts/serve_v3_mtp.sh) +
[`scripts/bench_v3_clean_ab.py`](scripts/bench_v3_clean_ab.py).

### v3 result — MTP is decisively positive across all measured operating points

#### Exp 1 — sequential dialog (N=5 trials × 5 prompts = 25 measurements per phase)

| Metric | no-MTP | MTP k=1 | Δ |
|---|---:|---:|---:|
| **decode TPOT (ms / output token)** | **7.620 ± 0.022** | **5.976 ± 0.456** | **−21.6 %** (≡ +27.5 % faster decode rate) |
| total throughput tok/s (incl prefill) | 113.4 ± 12.9 | 149.3 ± 17.3 | +31.7 % |
| TTFT (ms) | 78.6 ± 4.4 | 53.9 ± 15.0 | −31.4 % |

The decode TPOT improvement holds **on every prompt individually**:

| Prompt | no-MTP TPOT | MTP TPOT | Δ |
|---|---:|---:|---:|
| p1 (sky color, 50 tok) | 7.618 ms | 5.883 ms | −22.8 % |
| p2 (Python fib, 80 tok) | 7.619 ms | 5.550 ms | −27.2 % |
| p3 (TCP vs UDP, 130 tok) | 7.613 ms | 6.037 ms | −20.7 % |
| p4 (tofu steps, 200 tok cap) | 7.607 ms | 5.869 ms | −22.8 % |
| p5 (haiku, 19–22 tok) | 7.642 ms | 6.540 ms | −14.4 % |

#### Exp 3 — concurrent stress (20 reqs × N=5 trials per concurrency)

| Concurrency | no-MTP agg tok/s | MTP agg tok/s | Δ agg | no-MTP TPOT | MTP TPOT | Δ TPOT |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 118.80 | 153.32 | +29.1 % | 7.633 ms | 6.048 ms | −20.8 % |
| 4 | 271.94 | 310.16 | +14.1 % | 12.133 ms | 10.704 ms | −11.8 % |
| 8 | 390.25 | 446.04 | +14.3 % | 15.165 ms | 13.785 ms | −9.1 % |

The TPOT speedup attenuates as concurrency rises (−21 % → −12 % → −9 %),
consistent with the vLLM Recipes phrasing that "MTP-1 reduces per-token
latency but degrades text throughput under high concurrency" — but at
C=8 on this hardware the **aggregate throughput is still +14 %** and
TPOT is still **negative**, so on 2× RTX 3090 the C=8 batching regime
remains MTP-positive. Higher concurrencies (C=16+) are not benched
because `--max-num-seqs 8` caps the engine.

#### Sanity check — content equivalence

Per-trial response SHA1 hashes confirm vLLM has **intrinsic non-determinism
even without MTP** (1–3 unique SHA1 across 5 trials, despite `seed=42`,
`temperature=0.5`) — this is well-known V1-engine chunked-prefill
non-determinism, not MTP-specific. MTP and no-MTP responses are
content-equivalent on manual inspection — same cooking steps, same Python
fib skeleton, same TCP vs UDP bullet structure, no truncation, no
gibberish. Full SHA1 audit + 200-char text previews dumped in the
v3 result JSONs.

### Reconciliation with v1/v2 published numbers

| Bench | flags | prefix cache | MTP delta | reading |
|---|---|---|---:|---|
| v1 (mtp_speculative_decoding.json, 2026-04-25) | 0.80/2 vs 0.90/8 | ON | **−12.0 %** | confounded — flag-effect dominates |
| v2-clean (intermediate, 2026-04-26 morning) | 0.90/8/0.90/8 | ON | **+17.7 %** (Exp 1 throughput) | matched flags, but cache effect still in play |
| **v3 (this section, 2026-04-26)** | **0.90/8/0.90/8** | **OFF** | **+27.5 %** (decode rate) | **clean — both confounders removed** |

Decomposition (each step holds the prior factor fixed):

- Removing the flag confound (0.80/2 → 0.90/8 on the MTP run) accounts
  for ≈ 30 percentage points of the swing.
- Disabling prefix caching on top of that adds another ≈ 10 percentage
  points to the MTP advantage. This is consistent with vllm #38182's
  observation that MTP drops cache hit rate 92 % → 71 % — under
  cache-ON, MTP loses cache hits that no-MTP keeps, masking MTP's
  compute speedup.

### Attribution note on cache-OFF for MTP

The `--no-enable-prefix-caching` recommendation for latency-focused MTP
serving was documented by [vLLM Recipes](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html)
in its 2026-04-24 revision, two days before our v3 publication. Recipes
cites a **KV-capacity argument** ("speculative tokens consume KV cache
capacity, reducing effective batch size"). Our v3 contribution on top of
that is: (a) **quantification** of the impact on dual 3090 PCIe + AWQ +
Qwen3.6-35B-A3B (decode rate +27.5 % vs cache-ON baseline), (b) **confound
decomposition** (matched-flag fix ≈ 30 pp, cache-OFF ≈ 10 pp), and
(c) connecting the cache-OFF benefit specifically to the
[vllm-project/vllm#38182](https://github.com/vllm-project/vllm/issues/38182)
`single_type_kv_cache_manager.py:L457` force-drop mechanism for
Qwen3.5/3.6 A3B class (a different mechanism than the KV-capacity one
Recipes mentions; not addressed by Recipes). The cache-OFF preference
appears specific to A3B-class models with the L457 trigger condition —
single 3090 + Qwen3.6-27B dense + `--enable-prefix-caching` ON + MTP k=3
[has been reported working at 97/95/91 % per-position acceptance](https://medium.com/@fzbcwvv/an-overnight-stack-for-qwen3-6-27b-85-tps-125k-context-vision-on-one-rtx-3090-0d95c6291914),
so this is not a Qwen3-family-wide rule.

### Reconciliation with the Modal A100 −11.4 % finding

[`results/modal_2x_a100_v2.json`](results/modal_2x_a100_v2.json) was
collected with `--enable-prefix-caching` ON on both runs, mirroring v1
production. Its prompt-4 decode-only delta of **−11.4 %** is now best
read as the **prefix-cache-ON regime** datapoint on A100 NVLink, **not**
as evidence that MTP is intrinsically negative. Whether A100 with
prefix-cache OFF would also flip positive (matching this 3090 v3 result)
is an open question — Modal credits permitting, a v3-equivalent A100 run
is the obvious next step. We will add that result here when collected.

### Practical recommendation (revised)

For the single-stream voice-dialog deployment that motivated this repo on
**2× RTX 3090 PCIe + AWQ-Marlin Q4 + vLLM 0.19.1**:

- **Enable MTP** with `--speculative-config '{"method":"mtp","num_speculative_tokens":<K>}'`
  if you can run with `--no-enable-prefix-caching` (the typical single-user
  voice-dialog case has near-zero prefix-cache hit rate anyway).
  - v3 (this section's data) was measured at **k=1** and is +27.5 % decode
    rate vs no-MTP.
  - v4 (2026-05-07, see [`v4_2026_05_07/`](v4_2026_05_07/)) sweeps k∈{1,2,3}
    on the same hardware: **k=3 is the new recommended production setting**
    (TTFT −33 % and tok/s +8 % vs k=2; decode TPOT essentially unchanged
    across k=1/2/3). If you optimise for TTFT (latency-to-first-word in
    voice dialog), use k=3; if you prefer the simplest published config,
    k=1 is still well above no-MTP.
- **Be cautious about combining MTP with prefix-caching** until vllm
  #38182 / #40756 are resolved. If your workload depends on prefix-cache
  hit rate (multi-turn chat with shared system prompt), benchmark the
  net effect on **your** workload before enabling MTP — the cache loss
  may eat the MTP gain.
- The MoE expert-saturation analysis (MoESD arXiv 2505.19645,
  Utility-Driven SD arXiv 2506.20675) still applies and explains why
  llama.cpp draft-spec on the same model+hardware is still net-negative
  (see sibling repo) — but vLLM MTP k=1 with prefix-cache disabled
  appears to dodge the expert-saturation pathology at this `K=1`.

## Comparison vs alternative architectures

| Approach | Dialog tok/s | Vision quality | Vision speed | Concurrent? |
|---|---:|:---:|---:|:---:|
| Ollama unified (baseline) | 107 | ⭐⭐⭐⭐⭐ | 6–31 s (blocks dialog) | ❌ |
| GPU partition: 2× qwen3.6 | ~107 | ⭐⭐⭐⭐⭐ | ~2–3 s | ✅ |
| GPU partition: qwen2.5vl-7b on GPU1 | 107 | ⭐⭐ | ~1.5 s | ✅ |
| **vLLM unified TP=2** (this) | **125.8** alone / **120.4** concurrent | ⭐⭐⭐⭐⭐ | **397 ms** concurrent | ✅ |

## Reproducer

```bash
git clone https://github.com/thc1006/qwen3.6-vllm-2x3090
cd qwen3.6-vllm-2x3090
# 1. Set up vLLM venv
uv venv .venv --python 3.12 --seed
.venv/bin/pip install 'vllm>=0.19.0' 'transformers>=5.5.4' pillow

# 2. Pull AWQ model (~24 GB)
.venv/bin/hf download QuantTrio/Qwen3.6-35B-A3B-AWQ \
    --local-dir ~/models/qwen36-awq

# 3a. Serve — v1/v2 baseline (no MTP, prefix-caching ON)
MODEL_PATH=~/models/qwen36-awq bash scripts/vllm_serve.sh

# 3b. Serve — v3/v4 MTP production (cache OFF, +27.5 % decode rate)
MODEL_PATH=~/models/qwen36-awq bash scripts/serve_v3_mtp.sh
# v4 recommendation: edit num_speculative_tokens to 3 (TTFT-optimised)

# 4. Bench (in another terminal)
.venv/bin/python scripts/bench_vllm_dialog.py
.venv/bin/python scripts/bench_vllm_concurrent.py
```

## Prior art

- [vllm#40124](https://github.com/vllm-project/vllm/issues/40124) · TurboQuant FP8 + Hybrid MoE on Ampere (13 patches)
- [Sandermage/genesis-vllm-patches](https://github.com/Sandermage/genesis-vllm-patches) · Runtime monkey-patches for the 13 issues
- [thinksmart.life · Qwen3.5-35B on 4× 3090](https://thinksmart.life/research/posts/qwen35-35b-4x3090-vllm-pcie/) · PCIe topology study
- [thc1006/qwen3.6-speculative-decoding-rtx3090](https://github.com/thc1006/qwen3.6-speculative-decoding-rtx3090) · Earlier llama.cpp spec-decode bench (single 3090)

## License

Apache-2.0. Built for the Reachy Mini robot brain stack — see
[reachy-mini-spark-deployment](https://github.com/thc1006/reachy-mini-spark-deployment)
(private) for the deployment journal that motivated this experiment.

## Author

Hsiu-Chi Tsai · [@thc1006](https://github.com/thc1006) · hctsai1006@cs.nctu.edu.tw
