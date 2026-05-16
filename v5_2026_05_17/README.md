# qwen3.6-vllm-2x3090 v5.0 — Dense Qwen3.6-27B vs MoE Qwen3.6-35B-A3B on voice agent workload (2026-05-17)

A natural extension of [v4.0](../v4_2026_05_07/) (which fixed the production stack at MoE + MTP k=3 + TP=2) to ask a single follow-up: **does Qwen3.6-27B, the new dense sibling that fits TP=1, give us a cheaper-to-run alternative for a voice agent?**

Spoiler: on this stack and this workload, **no**. The MoE production stack wins decisively on every axis that matters for a real-time voice agent — TTFT, throughput, and tool-call discrimination. The Dense 27B's only win (cleaner raw zh-TW on the chat responses it *does* produce) is contingent on it answering far fewer chat prompts due to over-eager tool firing.

## TL;DR — four headline numbers

1. **TTFT: MoE 178 ms vs Dense 772 ms (MoE 4.34× faster).** With production MTP k=3, the MoE returns first token in under 200 ms on the same hardware where Dense 27B-AWQ on a single 3090 (no MTP head exists) takes 0.77 s. For a conversational robot this is the difference between "responsive" and "noticeable lag".
2. **Throughput: MoE 88 tok/s vs Dense 16 tok/s (MoE 5.42× faster).** Dense 27B forced to `--enforce-eager` on TP=1 plus the absence of any draft head means it loses on the steady-state decode rate as well, not just on TTFT.
3. **Tool-call discrimination: MoE 100 % (30/30) vs Dense 77 % (23/30).** The 7 Dense misses are all in the same class — chat prompts ("你好" / "你今天好嗎" / "跟我講個短笑話") where Dense over-eagerly fires `play_emotion` instead of replying. Production hit-rate at MoE matches our [v3 synth E2E bench](https://github.com/thc1006/qwen3.6-vllm-2x3090) at 10/10 tool decisions.
4. **Raw zh-TW purity: Dense 27B wins on the responses it gives, but the sample is biased.** 0/5 of Dense's actual chat responses leaked simplified characters; MoE leaked simplified on c1 (all 3 trials of "你好，我是瑞奇"). However Dense only produced 5 chat outputs vs MoE's 12, because Dense converted 7 chat prompts into spurious tool calls — so the cleaner Dense distribution is partly an artifact of refusing to chat. **The robot_brain.py `OpenCC s2t` post-processor that we shipped on commit `a7912c7` is independently justified for either model.**

## Decision

**No swap.** Keep MoE Qwen3.6-35B-A3B + MTP k=3 + TP=2 + `--no-enable-prefix-caching` as the production LLM (the v4.0 config). The dense sibling is not a free upgrade on this hardware for this workload.

This release is the falsification of a tempting "dense 27B is smaller, simpler, fits TP=1, should be cheaper to serve" intuition with a same-hardware A/B. It is **not** a generalizable claim that dense always loses to MoE+MTP — it is the specific result for **this stack** (vLLM 0.19.1, dual RTX 3090 PCIe no NVLink, AWQ-Marlin, voice-agent prompt distribution).

## Hardware + software

- Workstation `s1`: 2 × NVIDIA RTX 3090 24 GB (PCIe Gen4 ×16, no NVLink). Power capped 220 W via `nvidia-smi -pl`. CUDA 12.x, vLLM 0.19.1.
- MoE arm: `tclf90/Qwen3.6-35B-A3B-AWQ` served via `vllm serve --tensor-parallel-size 2 --gpu-memory-utilization 0.85 --max-model-len 32768 --max-num-seqs 4 --enable-chunked-prefill --no-enable-prefix-caching --speculative-config '{"method":"mtp","num_speculative_tokens":3}' --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 --mm-encoder-tp-mode data --mm-processor-cache-type shm`.
- Dense arm: `QuantTrio/Qwen3.6-27B-AWQ` (community AWQ Q4, ~13.5 GB on disk) served via `vllm serve --tensor-parallel-size 1 --gpu-memory-utilization 0.95 --max-model-len 2048 --max-num-seqs 1 --enforce-eager --limit-mm-per-prompt '{"image":0,"video":0}' --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 --trust-remote-code`. GPU 1 only; production was paused while Dense was bench'd, then resumed.

## Workload

10 prompts × 3 trials = 30 samples per model, streaming SSE with TTFT and `usage` captured from the final `[DONE]` chunk. Prompts and tools mirror the conversational pipeline used in `robot_brain.py` against the actual production system prompt shape.

| pid | prompt (zh-TW) | expected | category |
|---|---|---|---|
| c1 | 你好，我是瑞奇。 | none (chat) | greeting |
| c2 | 你今天好嗎？ | none (chat) | smalltalk |
| c3 | 跟我講個短笑話。 | none (chat) | open generation |
| c4 | 你是誰？簡短自我介紹。 | none (chat) | identity |
| t1 | 向左轉。 | `move_head` | motion |
| t2 | 頭抬起來看上面。 | `move_head` | motion |
| t3 | 跳一下舞。 | `play_emotion` | emotion |
| t4 | 做個開心的動作。 | `play_emotion` | emotion |
| t5 | 停下來。 | `stop_motion` | stop |
| t6 | 現在幾點？ | `get_current_time` | system |

Decode: `temperature=0.2`, `max_tokens=200`, `stream=True`, `chat_template_kwargs={"enable_thinking": False}` (per the [synth bench finding](https://github.com/thc1006/qwen3.6-vllm-2x3090) that thinking-mode + tight `max_tokens` truncates `<tool_call>` tags). `tool_choice="auto"` so a model that prefers to chat is free to.

Bench script: [`bench/v5_voice_bench.py`](bench/v5_voice_bench.py). Raw rows: [`data/results_moe_35b_mtp_tp2.json`](data/results_moe_35b_mtp_tp2.json) and [`data/results_dense_27b_tp1.json`](data/results_dense_27b_tp1.json). Analyzer: [`analysis/analyze_v5.py`](analysis/analyze_v5.py), aggregated: [`analysis/aggregate.json`](analysis/aggregate.json).

## Results — full per-model aggregates

### MoE Qwen3.6-35B-A3B + MTP k=3 + TP=2 (production)

| metric | mean | median | stdev | min | max |
|---|---:|---:|---:|---:|---:|
| TTFT (ms) | **178** | 171 | 77 | 141 | 575 |
| e2e (ms) | 274 | 249 | 125 | — | — |
| tok/s | **88.0** | 74.5 | 32.4 | — | — |
| tool accuracy | **30/30 (100 %)** | — | — | — | — |
| chat false-fires | 0/12 | — | — | — | — |
| tool misses | 0/18 | — | — | — | — |
| zh-TW (TRAD/SIMP/shared) on chat | 6/3/3 | — | — | — | — |

### Dense Qwen3.6-27B (no MTP, TP=1 GPU1, enforce-eager)

| metric | mean | median | stdev | min | max |
|---|---:|---:|---:|---:|---:|
| TTFT (ms) | 771 | 770 | 201 | 455 | 1644 |
| e2e (ms) | 1684 | 1584 | 644 | — | — |
| tok/s | 16.2 | 17.0 | 2.5 | — | — |
| tool accuracy | 23/30 (77 %) | — | — | — | — |
| chat false-fires | **7/12** | — | — | — | — |
| tool misses | 0/18 | — | — | — | — |
| zh-TW (TRAD/SIMP/shared) on chat | 5/0/0 | — | — | — | — |

### Head-to-head

| metric | MoE win factor |
|---|---|
| TTFT (lower is better) | **4.34×** |
| e2e (lower is better) | **6.13×** |
| tok/s (higher is better) | **5.42×** |
| tool accuracy (higher is better) | +23.3 pp |

## Where Dense 27B's chat false-fires actually came from

The 7 chat-bucket false-fires are not scattered randomly — they cluster on c1 ("你好，我是瑞奇") and c2 ("你今天好嗎？"), both 3/3:

```
c1/1  SIMP→none (no text)  tools=['play_emotion']
c1/2  SIMP→none            tools=['play_emotion']
c1/3  SIMP→none            tools=['play_emotion']
c2/1  ...                  tools=['play_emotion']
c2/2  ...                  tools=['play_emotion']
c2/3  ...                  tools=['play_emotion']
c3/2  ...                  tools=['play_emotion']     # 1/3 trials of "tell me a joke" also fired emotion
```

The MoE replies to those same prompts with text (e.g. c2 → "我很好，謝謝！你呢？"). Dense reads "你好" as "user wants robot to react with an emotion" — plausible from a tool-only viewpoint, but wrong for a robot whose primary interface is conversation.

A system-prompt tweak ("only call a tool when the user names an action verb") would probably narrow this gap. We did not run that ablation; it is out of scope for a "production-config A/B".

## Scope and known caveats

This is a **single-day, N=3, single hardware** measurement. Read it as falsifying one specific hypothesis ("Dense 27B is a cheap free-upgrade for our voice agent"), not as a generalized "Dense vs MoE" claim.

- **TP confound.** Dense ran TP=1 on a single 3090 because its 13.5 GB AWQ weights fit cleanly there, and because routing it through TP=2 would have stolen GPU from the MoE arm during its bench window. A "Dense 27B TP=2 + no `--enforce-eager`" run is a follow-up. The likely TP=2 win for Dense is ≤ 2× tok/s based on textbook scaling and what we saw on MoE TP=1 → TP=2 failures in [v4 Phase B](../v4_2026_05_07/). Even doubling Dense's 16 tok/s does not catch the MoE's 88.
- **Spec-decode asymmetry.** MoE ran with MTP k=3 (its production winner from v4); Dense ran with no spec because no public draft head exists for Qwen3.6-27B yet. This is *the right* comparison if the question is "what should I serve in production". It is *the wrong* comparison if the question is "is dense's base decode faster than MoE's base decode" — and that question is moot, because nothing in production runs base.
- **Workload coverage.** 10 prompts cover the four shape-classes that drive 95% of the robot_brain.py call sites (greeting / smalltalk / open generation / identity / 4 tool families) but do not exercise long-context reasoning, image input, or multi-turn memory. v3+v4 long-context results carry over for MoE; Dense long-context is not measured here.
- **OpenCC dependency for zh-TW status.** The analyzer prefers `OpenCC("t2s")` for canonical traditional-vs-simplified detection; a char-marker heuristic is the fallback if `opencc` is not installed. Both agree on the present dataset; the OpenCC numbers are in the table above.
- **No statistical inference.** Sample sizes (N=3 per cell) are too small to justify Welch's-t style claims. The TTFT gap is 4×, throughput gap is 5×, tool-accuracy gap is 23 pp; these are unambiguous at this N. **A close call (e.g. ±10%) at this N would not be publishable.**

## Reproducing locally

```bash
# 1) Make sure production vllm-server.service is up and exposes qwen36-awq at :8000.
V5_MODEL=qwen36-awq V5_TRIALS=3 V5_OUT=results_moe_35b_mtp_tp2.json python3 v5_voice_bench.py

# 2) Stop production. Bring up dense 27B test instance on one GPU (single 3090).
hf download QuantTrio/Qwen3.6-27B-AWQ --local-dir ~/models/qwen36-27b-awq
CUDA_VISIBLE_DEVICES=1 vllm serve ~/models/qwen36-27b-awq \
  --served-model-name qwen36-27b --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.95 --max-model-len 2048 \
  --max-num-seqs 1 --enforce-eager \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 \
  --trust-remote-code --host 127.0.0.1 --port 8000

V5_MODEL=qwen36-27b V5_TRIALS=3 V5_OUT=results_dense_27b_tp1.json python3 v5_voice_bench.py

# 3) Analyze (OpenCC optional but recommended).
python3 analysis/analyze_v5.py
```

## Related work in this repo lineage

- [`../v4_2026_05_07/`](../v4_2026_05_07/) — 9-phase factorial sweep that fixed MTP k=3, AWQ ≈ FP8, TP=2 mandatory, 220 W sweet spot.
- [v3.0 release](https://github.com/thc1006/qwen3.6-speculative-decoding-rtx3090/releases/tag/v3.0) — original MTP +27.5 % cache-OFF measurement that started this series.
- [Reachy Mini voice agent](https://github.com/thc1006/reachy-mini-agent) — the consumer of this stack. The `robot_brain.py` script there talks to vLLM via the same payload shape used in this bench.

---

*v5.0 measurement window: 2026-05-17, single afternoon, single operator. Raw rows + bench script + analyzer all in this directory. Bug reports / replications welcome via issues against this repo.*
