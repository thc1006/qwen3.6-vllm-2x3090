# qwen3.6-vllm-2x3090

Empirical answer to: **can a single vLLM engine on 2× consumer Ampere GPUs
serve concurrent vision+dialog for an embodied robot without dialog tok/s
collapsing under VL prefill?**

If yes, single unified model (qwen3.6-35b-a3b VL+dialog+tools) on 2× RTX 3090
beats GPU-partitioning architectures (B1/B2 in our internal taxonomy) for an
always-on conversational robot brain.

## Hardware

- 2× NVIDIA RTX 3090 24GB (SM 8.6, Ampere, no NVLink, PCIe Gen4 x8)
- Driver 580.126, CUDA 13.0
- Ubuntu 24.04 LTS, Python 3.12.3
- Stock GPU clocks, no power-limit changes

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

## Results

_(populated when bench completes)_

## Comparison vs alternative architectures

| Approach | Dialog tok/s | Vision quality | Vision speed | Concurrent? |
|---|---:|:---:|---:|:---:|
| Ollama unified (baseline) | 107 | ⭐⭐⭐⭐⭐ | 6–31s (blocks dialog) | ❌ |
| GPU partition: 2× qwen3.6 | ~107 | ⭐⭐⭐⭐⭐ | ~2–3s | ✅ |
| GPU partition: qwen2.5vl-7b on GPU1 | 107 | ⭐⭐ | ~1.5s | ✅ |
| **vLLM unified TP=2** (this) | _TBD_ | ⭐⭐⭐⭐⭐ | _TBD_ | _TBD_ |

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

# 3. Serve
MODEL_PATH=~/models/qwen36-awq bash scripts/vllm_serve.sh

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
