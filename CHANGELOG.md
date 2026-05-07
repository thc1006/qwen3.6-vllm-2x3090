# Changelog

All notable changes to this benchmark are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning is not strictly semver — each numbered release is a public
publication point with its own data set.

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

- **k=3 is the new production recommendation** (replacing k=1 from v3.0
  analysis and k=2 from earlier production deploy). Reason: TTFT savings
  ~26 ms at p<0.001 in all 4 (power × temp) cells; TPOT statistically
  equivalent. The v3.0 analysis correctly identified that MTP gives a net
  speedup; the v4.0 analysis refines the optimal k for voice-agent TTFB.

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
