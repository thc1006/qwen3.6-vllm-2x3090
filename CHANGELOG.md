# Changelog

All notable changes to this benchmark are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning is not strictly semver — each numbered release is a public
publication point with its own data set.

## [v5.0] — 2026-05-17

> **ERRATA (2026-05-17, same-day audit, fix commit on master)**: an audit
> pass after the initial v5.0 release surfaced three issues. (1)
> `analysis/analyze_v5.py` shipped with a unidirectional zh-TW classifier
> that collapsed SIMP into "shared"; corrected counts: MoE chat replies
> are **6 TRAD / 3 SIMP / 3 MIX** (the 3 SIMP are all c1 "你好" trials),
> Dense's 5 chat replies are **5 TRAD / 0 SIMP / 0 MIX**. (2) The bench
> SYSTEM_PROMPT is a 4-line stub with 4 tools, not the actual production
> `robot_brain.py:1292` SYSTEM_PROMPT, which has 8 tools, JSON output
> schema, and explicit "你好 / hi → speech only, use 'greet' animation"
> guidance. So the **77 % vs 100 % tool-accuracy gap is partly an
> artifact of bench prompt under-specification**, not pure model
> behavior; a follow-up retest with the production SYSTEM_PROMPT is the
> right way to confirm whether Dense's chat false-fires survive proper
> steering. (3) vLLM-config confounds beyond TP and spec-decode were not
> originally disclosed: prefix-caching default ON for Dense vs explicit
> OFF for MoE, `--enforce-eager` (no CUDA graphs) for Dense vs CUDA
> graphs for MoE, `max-num-seqs` 4 vs 1, chunked-prefill on vs off.
> **Latency findings (TTFT 4.34×, tok/s 5.42×, e2e 6.13×) are dominated
> by compute and remain robust**; the tool-accuracy and zh-TW purity
> findings need the prompt-matched retest. The v5.0 tag is preserved
> unchanged for reproducibility; this is a documentation-only fix
> committed after the tag. See
> [`v5_2026_05_17/README.md`](v5_2026_05_17/README.md) for the expanded
> caveats.

### Added

- **Dense Qwen3.6-27B vs MoE Qwen3.6-35B-A3B head-to-head on voice agent
  workload** with the production stack fixed at v4.0 settings (MoE + MTP
  k=3 + TP=2 + `--no-enable-prefix-caching`). Full content in
  [`v5_2026_05_17/`](v5_2026_05_17/). 10 prompts × 3 trials = 30 samples
  per model with streaming SSE TTFT capture, tool-call accuracy scoring,
  and zh-TW vs zh-CN purity analysis via OpenCC `t2s`.
- [`v5_2026_05_17/data/`](v5_2026_05_17/data/) — raw rows (60 total).
- [`v5_2026_05_17/bench/v5_voice_bench.py`](v5_2026_05_17/bench/v5_voice_bench.py)
  — bench harness.
- [`v5_2026_05_17/analysis/analyze_v5.py`](v5_2026_05_17/analysis/analyze_v5.py)
  + [`aggregate.json`](v5_2026_05_17/analysis/aggregate.json) — analyzer
  and aggregate metrics.

### Headline numbers

| metric | MoE 35B-A3B + MTP k=3 + TP=2 | Dense 27B no-spec TP=1 | MoE win factor |
|---|---:|---:|---:|
| TTFT mean (ms) | **178** | 771 | **4.34×** |
| e2e mean (ms) | **274** | 1684 | **6.13×** |
| tok/s mean | **88.0** | 16.2 | **5.42×** |
| tool accuracy | **30/30 (100 %)** | 23/30 (77 %) | +23.3 pp |
| chat false-fires | 0/12 | 7/12 | — |

### Decision

- **No production swap.** Keep MoE + MTP k=3 + TP=2 (v4.0 config).
- Dense 27B-AWQ is not a free upgrade on this hardware for this workload:
  it loses on TTFT (4.3×), throughput (5.4×), and tool-call discrimination
  (over-fires `play_emotion` on greetings/smalltalk 7/12 chat prompts).
- The only Dense win — cleaner raw zh-TW on the chat outputs it does
  produce (5/5 TRAD vs MoE 6/9 TRAD with c1 leaking SIMP) — is partly an
  artifact of Dense refusing to chat in the first place, and is
  independently solved by `robot_brain.py`'s OpenCC `s2t` post-processor
  (shipped on commit `a7912c7`).

### Scope and caveats

(See ERRATA above for the expanded caveat list added in the audit pass.)

- N=3 trials per cell; sufficient for the 4× / 5× / 23 pp gaps shown; not
  sufficient for close calls.
- Dense ran TP=1 + `--enforce-eager` + `--limit-mm-per-prompt '{"image":0,"video":0}'`
  + 0.95 mem-util (single 3090 budget; production was paused for this
  arm). A "Dense TP=2 + no enforce-eager" follow-up arm is out of scope
  for this release — textbook scaling suggests ≤ 2× tok/s, which would
  still not close the 5.4× throughput gap.
- Dense ran without spec decoding (no public MTP draft head exists for
  Qwen3.6-27B yet). This is *the right* comparison for "what should I
  serve in production"; it is the wrong comparison for "is dense's base
  decode faster than MoE's base decode", and the latter is moot because
  nothing in production runs base.
- Single hardware (2× RTX 3090 PCIe, no NVLink, SM 8.6). NVLink / HBM /
  H100-class hardware would change every absolute number; the
  *direction* of "MoE+MTP wins by a wide margin on voice-agent shape" is
  unlikely to flip but is unverified here.

## [v4.0] — 2026-05-07

### Added

- **9-phase factorial sweep** with statistical analysis (~3000 measurements
  across 38 configurations). Full content lives in
  [`v4_2026_05_07/`](v4_2026_05_07/). Highlights:
  - **Phase A** k sweep (k=1/2/3 × power × temp = 12 configs): k=3 winner
    via TTFT (−33 % p<0.001), TPOT statistically equivalent to k=2.
  - **Phase B** TP=1 vs TP=2: TP=1 categorically does not fit on single
    24 GB RTX 3090 (3 progressive memory configs all OOM, even no-spec).
  - **Phase C+H+I** AWQ vs FP8 with matched gpu-memory-utilization control:
    AWQ ≈ FP8 within statistical noise (all 4 cells p > 0.6). FP8 minimum
    mem-util on dual 3090 with Whisper sidecar = 0.90.
  - **Phase E** 60-min sustained-load × 2 power configs: no monotonic
    acceptance regression detected (NS at α=0.05). Counter-evidence to
    [vllm-project/vllm#41838](https://github.com/vllm-project/vllm/issues/41838)
    on Ampere — but note we tested MTP, the issue is about Eagle3.
  - **Phase F** tool-call workload: 25/25 produced tool_calls; 3-metric
    nuance — TPOT lower (p<0.0001) but TTFT higher (~3×) and tok/s lower.
    Acceptance rate is NOT significantly different — refutes "structured
    tokens → higher MTP acceptance" hypothesis.
  - **Phase G** long-context decode scaling: TPOT scales steeply on dual
    3090 PCIe TP=2 (+150 % at 24 k tokens). TP=2 inter-GPU communication
    overhead dominant past ~6 k.
  - **Phase J + J.2** vLLM 0.20.1 vs 0.19.1 with backend-confound control:
    `VLLM_USE_FLASHINFER_MOE_FP16` is essentially a no-op for AWQ-Marlin
    Qwen3.6 (NS p > 0.57), and version effect is also NS (p > 0.34). The
    [#41306 MoE-backend regression](https://github.com/vllm-project/vllm/issues/41306)
    does not manifest on AWQ-Marlin path on Ampere SM 8.6. **Caveat**:
    0.20.1 raises `NotImplementedError` if `VLLM_USE_FLASHINFER_MOE_FP16=1`
    is set; must unset it.
- [`v4_2026_05_07/data/`](v4_2026_05_07/data/) — 27 phase JSONs (~1 MB).
- [`v4_2026_05_07/bench/`](v4_2026_05_07/bench/) — bench scripts (matched-flag
  methodology, streaming SSE, spec-acceptance from /metrics, t-test, p-value).
- [`v4_2026_05_07/analysis/`](v4_2026_05_07/analysis/) — statistical analysis
  (Welch's t-test, perf/W, per-prompt breakdown, MTP cross-k SHA1 lossless
  check).

### Changed

- **k=3 is the new production recommendation on this 2× RTX 3090 PCIe + AWQ
  + vLLM 0.19.1 hardware** (replacing k=1 from v3.0 analysis and k=2 from
  earlier production deploy on the same machine). Reason: TTFT savings
  ~26 ms at p<0.001 in all 4 (power × temp) cells; TPOT statistically
  equivalent. The v3.0 analysis correctly identified that MTP gives a net
  speedup; the v4.0 analysis refines the optimal k for voice-agent TTFB on
  this stack. **Cross-hardware caveat**: [vLLM Recipes](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html)
  recommends k=2 as the Qwen3.5/3.6 default; we have not validated k=3 on
  single-card (TP=1), NVLink, HBM, or different quant paths, and would
  recommend testing both on your setup before deviating from the Recipes
  default.

### Caveats

- **Finding "FP8 +2.7 % vs AWQ" — RETRACTED.** Earlier impression that FP8
  beats AWQ was driven by `gpu-memory-utilization=0.85` (AWQ) vs `0.92`
  (FP8) confound. With matched mem-util at 0.92, all 4 cells NS (p > 0.6).
  Additionally, AWQ at 0.85 and at 0.92 are themselves NS (p > 0.95) —
  mem-util setting is decoupled from decode speed at concurrency=1.
- **Tool-call "faster" framing — clarified.** Tool-call has lower TPOT
  (good) but higher TTFT and lower tok/s (less good). For voice-agent
  short outputs, total wall-clock is acceptable (~240 ms for 35-token
  tool_call), but readers should not infer "tool-call uniformly faster".

### Reproduction

- See [`v4_2026_05_07/README.md`](v4_2026_05_07/README.md) for full
  publication writeup, methodology, and full per-config tables.
- Bench scripts require setting `S1_SUDO_PW` env var for nvidia-smi power
  cap; no hardcoded passwords in repo.

## [v3.0] — 2026-04-26

### Changed

- **MAJOR — MTP headline finding flipped from `−12 % NET LOSS` to `+27 %
  faster decode rate`** after a clean A/B retest on the same hardware
  (s1 2× RTX 3090 PCIe). The v3 retest fixes two confounders that biased
  the v1/v2 numbers: (a) MTP run used `--gpu-memory-utilization 0.80
  --max-num-seqs 2` while the no-MTP baseline used `0.90 / 8` (flag
  confound — disclosed in v2.x but not yet corrected), and (b) prefix
  caching was ON in both v1/v2 runs, which interacts adversely with MTP
  per [vllm #38182](https://github.com/vllm-project/vllm/issues/38182)
  (MTP drops prefix-cache hit rate ~92 % → ~71 %). Under matched flags
  AND `--no-enable-prefix-caching`, the per-output-token decode time
  drops from **7.620 ± 0.022 ms (no-MTP)** to **5.976 ± 0.456 ms (MTP
  k=1)**, a robust −21.6 % delta that holds on every individual prompt
  (range −14 % to −27 %) and across concurrencies C ∈ {1, 4, 8} on the
  concurrent stress test.
- README MTP section: completely rewritten. v1/v2 numbers retained in-repo
  for full audit (`results/mtp_speculative_decoding.json`,
  `results/modal_2x_a100_v2.json`); the prose now leads with v3 and
  explicitly reconciles v1 (confounded), v2-clean intermediate
  (matched flags but cache-ON, +17.7 %), and v3 (matched flags + cache-OFF,
  +27.5 %).
- README "Hardware tuning disclosure → quantitative impact" — the line
  that previously said "the MTP NEGATIVE finding is completely
  insensitive to power-limit setting" is now updated to point at the
  v3 POSITIVE finding (also outside the power-limit band, just on the
  other side).

### Added

- [`results/mtp_v3_clean_ab_no_mtp.json`](results/mtp_v3_clean_ab_no_mtp.json)
  + [`results/mtp_v3_clean_ab_mtp.json`](results/mtp_v3_clean_ab_mtp.json) —
  full per-request data with TTFT, decode-only TPOT, response SHA1, and
  200-char text preview for content-equivalence audit. 25 measurements
  per phase for sequential dialog + 300 measurements per phase for
  concurrent stress at C ∈ {1, 4, 8}.
- [`results/mtp_v3_summary.json`](results/mtp_v3_summary.json) — aggregated
  summary statistics + interpretation.
- [`results/mtp_v3_master.txt`](results/mtp_v3_master.txt) — orchestration
  log for the boot → bench → kill → boot → bench → kill flow.
- [`scripts/run_v3.sh`](scripts/run_v3.sh) — orchestration that stops the
  production vllm-server systemd unit, runs no-MTP then MTP back to back
  with clean process boundaries, restarts production at the end.
- [`scripts/serve_v3_no_mtp.sh`](scripts/serve_v3_no_mtp.sh) /
  [`scripts/serve_v3_mtp.sh`](scripts/serve_v3_mtp.sh) — matched-flag serve
  scripts with `--no-enable-prefix-caching` on both.
- [`scripts/bench_v3_clean_ab.py`](scripts/bench_v3_clean_ab.py) — streaming
  bench client. Captures TTFT separately, computes decode-only TPOT,
  preserves response SHA1 + first 200 chars per request.
- [`scripts/aggregate_v3.py`](scripts/aggregate_v3.py) — statistics +
  determinism check + sanity-check pretty-printer.

## [v2.x] — 2026-04-25 → 2026-04-26 (intermediate, see git log)

- Hardware tuning disclosure (220 W power-limit, persistence mode, OS
  knobs).
- v2 voice latency budget retest with N=5 N=3 trials and corrected
  methodology (real STT engine, not VAD bug).
- Power scaling sweep N=5 across 200/220/250/280/320/350 W with both
  cards.
- Modal 2× A100-80GB SXM4 NVLink cross-hardware bench
  ([`results/modal_2x_a100_v2.json`](results/modal_2x_a100_v2.json)) —
  prompt-4 decode-only delta −11.4 % under prefix-cache-ON. Now read as
  the prefix-cache-ON regime A100 datapoint, **not** as evidence that
  MTP is intrinsically negative; a v3-equivalent A100 run with
  prefix-cache-OFF is the open follow-up.
- Cross-hardware comparison plot
  ([`analysis/plot_cross_hardware.png`](analysis/plot_cross_hardware.png)).
- v1 MTP feasibility bench
  ([`results/mtp_speculative_decoding.json`](results/mtp_speculative_decoding.json))
  with `_WARNING_config_confound` block disclosing 0.80/2 vs 0.90/8.

## [v1.0] — 2026-04-25

- Initial public release of vLLM TP=2 unified vision+dialog bench.
- T1 dialog baseline 126.4 tok/s, T2 vision 302 ms, T3 concurrent
  4.3 % dialog degradation under VL prefill.
- Decision: vLLM unified TP=2 on 2× consumer Ampere validated for
  embodied-robot dialog+vision.

[Unreleased]: https://github.com/thc1006/qwen3.6-vllm-2x3090/compare/v4.0...HEAD
[v4.0]: https://github.com/thc1006/qwen3.6-vllm-2x3090/releases/tag/v4.0
[v3.0]: https://github.com/thc1006/qwen3.6-vllm-2x3090/releases/tag/v3.0
[v2.0]: https://github.com/thc1006/qwen3.6-vllm-2x3090/releases/tag/v2.0
[v1.0]: https://github.com/thc1006/qwen3.6-vllm-2x3090/releases/tag/v1.0
