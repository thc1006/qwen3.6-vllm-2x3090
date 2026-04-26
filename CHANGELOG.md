# Changelog

All notable changes to this benchmark are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning is not strictly semver — each numbered release is a public
publication point with its own data set.

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
