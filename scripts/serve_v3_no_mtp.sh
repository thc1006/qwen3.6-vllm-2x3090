#!/bin/bash
# v3: Manually start vLLM with matched flags 0.90/8/hermes AND prefix caching DISABLED.
# Eliminates the across-trial prefix-cache inflation present in v2-clean run.
set -eu
MODEL_PATH=/home/reachym/models/qwen36-awq
export VLLM_USE_DEEP_GEMM=0
export VLLM_USE_FLASHINFER_MOE_FP16=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export OMP_NUM_THREADS=4
export VLLM_DISABLE_FP8=1
source /home/reachym/venvs/vllm/bin/activate
exec vllm serve "$MODEL_PATH" \
    --served-model-name qwen36-awq \
    --tensor-parallel-size 2 \
    --enable-expert-parallel \
    --gpu-memory-utilization 0.90 \
    --max-model-len 32768 \
    --max-num-seqs 8 \
    --enable-chunked-prefill \
    --no-enable-prefix-caching \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --reasoning-parser qwen3 \
    --mm-encoder-tp-mode data \
    --mm-processor-cache-type shm \
    --trust-remote-code \
    --host 127.0.0.1 --port 8000
