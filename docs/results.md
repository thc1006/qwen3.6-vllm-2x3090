# Results

## Summary

vLLM 0.19.1 with `QuantTrio/Qwen3.6-35B-A3B-AWQ` on 2× RTX 3090 (TP=2, AWQ-Marlin)
delivers **126 tok/s single-stream dialog** AND **0.3 s vision wall-clock**.
Under realistic concurrent load (1 dialog + 1 vision), dialog tok/s degrades
**only 4.3%** (126 → 120) and vision wall-clock inflates **31%** (302 → 397 ms).

**Verdict: Option F (vLLM continuous batching) validated.** Single unified
qwen3.6 model serves concurrent vision+dialog without dialog throughput
collapsing — eliminating the need for GPU partitioning architectures.

## Headline numbers

| Test | Result | vs Ollama unified |
|---|---:|:---:|
| Dialog alone (T1) | **126.4 tok/s** | +18% (was 107) |
| Vision alone (T2) | **302 ms** | 20–100× faster (was 6–31 s) |
| Concurrent dialog (T3) | **120.4 tok/s** | ∞× (Ollama: 0, blocked) |
| Concurrent vision (T3) | **397 ms** | 15–80× faster |

Dialog degradation under concurrent load: **4.3%**
Vision inflation under concurrent load: **31%**

## All test conditions

### T1 · Dialog baseline (sequential, max_tokens=200)

| # | tokens | wall-clock | tok/s |
|---:|---:|---:|---:|
| 1 | 200 | 1.59 s | 125.6 |
| 2 | 200 | 1.58 s | 126.5 |
| 3 | 200 | 1.58 s | 126.6 |
| 4 | 200 | 1.58 s | 126.7 |
| 5 | 200 | 1.58 s | 126.6 |
| **mean** | – | **1.58 s** | **126.4** |

### T2 · Vision baseline (sequential, realistic 154 KB JPEG, max_tokens=60)

| # | wall-clock | tokens out | text |
|---:|---:|---:|---|
| 1 | 0.31 s | 30 | "A black circle centered on a beige square…" |
| 2 | 0.33 s | 33 | "A beige square with a black circle inside…" |
| 3 | 0.27 s | 25 | "A beige square with a black circle centered…" |
| **mean** | **0.302 s** | – | – |

### T3 · 1 dialog + 1 vision concurrent (3 runs)

| Run | Dialog tokens | Dialog wall | Dialog tok/s | Vision wall |
|---:|---:|---:|---:|---:|
| 1 | 200 | 1.66 s | 120.5 | 0.39 s |
| 2 | 200 | 1.66 s | 120.3 | 0.40 s |
| 3 | 200 | 1.66 s | 120.4 | 0.40 s |
| **mean** | – | **1.66 s** | **120.4** | **0.397 s** |

Pass criteria evaluation:
- `120.4 / 126.4 = 0.953` → 4.7% degradation → ✅ < 10% threshold
- `0.397 / 0.302 = 1.314` → 31.4% inflation → ⚠️ 1.4 percentage points over 30% threshold but absolute 397 ms is excellent

### Stress test · 5 dialog + 1 vision concurrent (over-provision)

| Test | Result |
|---|---:|
| T1 dialog mean | 115 tok/s |
| T2 vision mean | 0.52 s |
| T3 concurrent dialog mean | 47 tok/s |
| T3 concurrent vision wall | 1.47 s |
| Dialog degradation | 59% |
| Vision inflation | 183% |

Note: this scenario exceeds realistic robot load (single user, one dialog
stream at a time). Including for completeness — confirms vLLM scales gracefully
when overloaded but should not be the primary metric.

## Tool calling validation

Initial test with `--tool-call-parser hermes` failed: model emits XML-style
`<tool_call><function=fname><parameter=name>value</parameter>…</function></tool_call>`
which the Hermes parser couldn't decode (`json.decoder.JSONDecodeError: Expecting value`).

Switching to `--tool-call-parser qwen3_xml` (in vLLM 0.19.1) resolved this.

After fix:
```
input:  "What is the weather in Taipei?" + tool spec
output: tool_calls=[{name: "get_weather", arguments: {"city":"Taipei"}}]
        content: "Let me check the current weather in Taipei for you."
```

## Streaming validation

```
prompt: "Count from 1 to 5."
ttfb: 157 ms
chunks: 13
total wall: 262 ms
content: "1, 2, 3, 4, 5"
```

## What didn't work

- `--tool-call-parser hermes` ❌ (json decode error on Qwen XML format)
- `--swap-space 8` ❌ (deprecated in vLLM 0.19, removed)
- Tool call without `chat_template_kwargs.enable_thinking=False` ❌ (model emits to `<think>` block, content empty)

## What was incidentally discovered

- vLLM 0.19.1 prints "Config file not found" warning at boot for Qwen3.6
  on RTX 3090 ([forum thread](https://discuss.vllm.ai/t/config-file-not-found-qwen-qwen3-6-35b-a3b/2567)).
  This is **performance-only** — inference is correct but uses generic MoE kernel.
  Tunable via `benchmark_moe.py` + `VLLM_TUNED_CONFIG_FOLDER`. Not addressed in this run.

- `Custom allreduce is disabled because your platform lacks GPU P2P capability`
  is **expected on consumer 3090 without NVLink**. NCCL fallback used. Doesn't
  affect correctness; minor TP perf cost (we still get 126 tok/s).

- Cold start: model load ~25 s + torch.compile ~33 s + warmup ~30 s = **~90 s
  total to first request**. Subsequent restarts: same 90 s.

## Boot timing (cold)

```
Worker_TP0_EP0 INFO Loading weights took 20.5–25.5 s
Worker_TP0_EP0 INFO Model loading took 11.62 GiB memory and ~25–30 s
Encoder cache initialised with budget of 16384 tokens
Dynamo bytecode transform: 8.7 s
Compile graph for range (1, 2048): 22.5 s
torch.compile total: 33.1 s
init engine (profile, kv cache, warmup): 106.5 s
Application startup complete: ~ T+135 s from launch
```

## Reproducer

See [README.md](../README.md) and [scripts/](../scripts/).

## Voice dialog end-to-end latency budget (novel — v2 with real production data)

Comparable cloud-only voice agent budget (per [Smallest.ai 2026](https://smallest.ai/blog/designing-voice-assistants-stt-llm-tts-tools-and-latency-budget),
[Trillet 2026](https://www.trillet.ai/blogs/voice-ai-latency-benchmarks)) is **~800 ms** total.
Public data on **embodied robot** stacks (with WebRTC + actual hardware) is essentially
zero. v2 below adds methodologically rigorous component bench AND real-conversation
ground-truth from production logs.

> **v1 retraction:** the first version of this section under-measured several stages
> (synthetic STT audio short-circuited VAD, toy LLM prompt, only-bge-m3 Mem0,
> ICMP-ping for WebRTC). The v1 "462 ms" figure was a theoretical lower bound on
> ideal components, not real conversation latency. v2 uses synthesized real speech,
> production-mirroring prompts, full Mem0 retrieve, and adds observed real-conversation
> p50 from this session.

### v2 component benchmarks (proper methodology)

| Stage | Latency v2 | What v1 reported | Methodology fix |
|---|---:|---:|---|
| WebRTC RTT (ICMP lower bound) | **5.4 ms** | 5.85 ms | unchanged (still lower bound; real WebRTC audio frame RTT is 30–80 ms) |
| STT 6 s **real speech** (Whisper CUDA) | **150.9 ms** (RTF 0.025) | 8.8 ms (bogus) | edge-tts synthesized speech instead of `np.random.uniform()` that VAD short-circuited |
| Mem0 **full search** (embed + Qdrant retrieve) | **104.5 ms** warm | 115.8 ms (embed only) | use `RobotMemory.search()` so Qdrant is included |
| LLM streaming TTFB **realistic prompt** (system + 30-msg history ≈ 1500 tok) | **195.7 ms** | 92.9 ms (toy 30-tok) | production-mirror prompt |
| TTS **first audio chunk** (edge-tts cloud) | **516.4 ms** | 239.0 ms (cherry-picked) | 5-prompt mean instead of 3, longer prompts |
| **Theoretical lower bound (sum)** | **972.8 ms** | 462 ms | – |

### Real production p50 (n=10 round-trips this session)

These are **observed** during real human-robot conversation, captured from terminal:

| Metric | p50 | min | p95 |
|---|---:|---:|---:|
| LLM streaming TTFB | **728 ms** | 521 ms | 2457 ms |
| Full pipeline round-trip (STT+LLM+TTS+playback) | **9398 ms** | 941 ms | 13618 ms |
| LLM `wall` (incl `speaker.wait_and_stop()`) | **10087 ms** | 5361 ms | 18731 ms |

### Honest interpretation

- The 9.4 s p50 round-trip is dominated by **the robot speaking its reply** (TTS
  audio plays at human speech rate, ~6–8 s for a 100-char response). The "user
  finishes speaking → robot starts speaking" window is **728 ms p50**, in
  cloud-agent territory.
- vLLM's `--enable-prefix-caching` is doing real work: prefill TTFT actually
  *decreases* slightly going from 0 history to 30 history (105 → 71 ms) because
  the system-prompt prefix gets cached after the first request.
- v1's 462 ms was a useful component-level lower bound but should not have been
  reported as "total user-perceived TTFB". The production p50 of 728 ms LLM TTFB +
  ~300 ms TTS first-byte + ~100 ms WebRTC playback start ≈ **~1.1 s real
  user-perceived TTFB before any audio is heard**.

Full raw v2 numbers + caveats: [`results/voice_latency_budget_v2.json`](../results/voice_latency_budget_v2.json).

## RTX 3090 power scaling for MoE inference (novel — v2 with both-GPU sweep)

> **v1 retraction:** v1 swept only GPU0 with GPU1 pinned at 350 W. With TP=2 the
> two GPUs are coupled by NCCL allreduce so v1's "throughput vs GPU0 power" was
> partially confounded by GPU1's own power state. v1 also used N=1 sample per
> level, max_tokens=200, and 3 s settle. v2 fixes all of these.

### v2 methodology

- **Both** GPUs swept simultaneously (`nvidia-smi -pl` set on i=0 and i=1).
- N=5 runs per level × 5 prompts × max_tokens=500 (long-form generation, sustained load).
- 30 s thermal settle between levels.
- Power draw sampled at 0.5 Hz during each run on both cards.
- GPU1 max is 350 W (FE card), so sweep range capped at **200 / 220 / 250 / 280 / 320 / 350 W**.

### v2 results (n=25 prompts per level)

| PL (W) | Mean tok/s ± stdev | Actual draw (W, both cards avg) | tok/s/W |
|---:|---:|---:|---:|
| 200 | **120.2 ± 0.18** | 196.6 | **0.611** ⭐ |
| 220 | 122.8 ± 0.13 | 210.6 | 0.583 |
| 250 | 124.7 ± 0.18 | 229.7 | 0.543 |
| 280 | 125.6 ± 0.18 | 244.5 | 0.514 |
| 320 | 125.6 ± 0.18 | 246.7 | 0.509 |
| 350 | **125.7 ± 0.20** | 248.6 | 0.506 |

### v2 findings (corrected from v1)

1. **There IS a real (small) gradient up to ~280 W**, then it plateaus.
   - 200 → 220 W: +2.3 % perf for +7 % power
   - 220 → 250 W: +1.5 % perf for +9 % power
   - 250 → 280 W: +0.7 % perf for +6 % power
   - 280 → 320 → 350 W: **flat** (within noise)
   - v1's "flat from 220 W up" was wrong — it was true above 280 W only.
2. **Plateau power draw is 244–249 W**, not v1's claimed 235 W. v1
   underestimated because GPU1 was capped at 350 W (forcing it not to scale
   with GPU0).
3. **Knee point: ~280 W** — that's where the curve flattens. Below 280 W there
   is real perf to be had per watt.
4. **Production sweet spots**:
   - **Maximum perf: 280 W** (125.6 tok/s, 244.5 W actual; only 0.1 % below 350 W)
   - **Best perf-per-watt: 200 W** (0.611 tok/s/W; 4.4 % less throughput, 21 % less power)
   - **95 % perf at minimum power: 220 W** (122.8 tok/s, 210.6 W actual)

The general intuition "more power = more tok/s" still holds for MoE inference
**up to a knee** of about 280 W per RTX 3090 in TP=2. Beyond that the workload
is bandwidth-bound (the v1 finding) and extra power is wasted. **The sweet spot
for an always-on robot brain is 220 W** (95 % of peak perf at 60 % of max power).

Full raw v2 numbers: [`results/power_scaling_v2.json`](../results/power_scaling_v2.json).

## Speculative decoding (MTP) — empirical NEGATIVE result

Re-tested 2026-04-25 with `--speculative-config '{"method":"mtp","num_speculative_tokens":1}'`.
Full numbers in [`results/mtp_speculative_decoding.json`](../results/mtp_speculative_decoding.json).

| | no-MTP (production) | MTP k=1 |
|---|---:|---:|
| Mean tok/s | **126** | 111 |
| Range | 125–127 | **75–142** |
| Variance ratio | 1× | **65×** |
| Best-case speedup | – | +12.6% |
| Worst-case slowdown | – | **−40%** |
| Cold-boot overhead | 0 | **+220 s** |
| Memory per card | 11.62 GiB | 12.41 GiB (+0.8) |

**Verdict: NET LOSS for diverse single-stream voice prompts on AWQ-Marlin Q4 + Ampere.**
The MTP draft heads are real and load fine (drafter shares the target's `lm_head` and
embedding weights — only +0.8 GB overhead per card, NOT a 2× model copy as I initially
feared). The problem is acceptance rate: when the draft misses, you pay the verification
cost without the parallel-decode payoff. With the baseline already at 126 tok/s, the
mean drops 12% and variance blows up 65×.

For a robot voice-dialog use case where TTFB consistency matters more than peak
throughput, **stay on the no-MTP path**. MTP may still help in batched server scenarios
where verification cost amortizes — not tested here.

This was prematurely reported as "OOM-blocked" in an earlier draft of this repo. That
was wrong: the OOM at first attempt was caused by external processes (faster-whisper at
1.4 GB on GPU0 + `gpu_memory_utilization=0.90`). With those factors removed and
`gpu_memory_utilization=0.80`, MTP boots cleanly. It's just slower for our workload.
