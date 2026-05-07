#!/usr/bin/env bash
# Phase J on s1: vLLM 0.20.x AWQ-Marlin TP=2 dual-3090 bench.
# Direct comparison to Phase A 0.19.1 numbers (same hardware, same model, same flags).
# Tests if #41306 MoE-backend regression affects AWQ-Marlin path on Ampere SM 8.6.

set -euo pipefail

S1_SUDO_PW="${S1_SUDO_PW:?Set S1_SUDO_PW env var}"
VENV020="${VENV020:-/home/reachym/venvs/vllm_020}"
VENV019="${VENV019:-/home/reachym/venvs/vllm}"
MODEL_AWQ="${MODEL_AWQ:-/home/reachym/models/qwen36-awq}"
BENCH_DIR="${BENCH_DIR:-$HOME/bench_2026_05_07}"
PORT=8000

mkdir -p "$BENCH_DIR"

# Same env as production (mirror Phase A) — only swap vLLM 0.19.1 → 0.20.x
# IMPORTANT: VLLM_USE_FLASHINFER_MOE_FP16=1 is NOT supported in 0.20.x
# (vllm 0.20.x raises NotImplementedError: "no FlashInfer unquantized MoE backend
#  supports the configuration"). For 0.20.x, let vLLM auto-choose MoE backend.
export VLLM_USE_DEEP_GEMM=0
unset VLLM_USE_FLASHINFER_MOE_FP16
export VLLM_USE_FLASHINFER_SAMPLER=0
export OMP_NUM_THREADS=4
export VLLM_DISABLE_FP8=1   # we're testing AWQ, not FP8

# Production-mirrored flags (same as Phase A k=3)
COMMON_FLAGS=(
    --served-model-name qwen36-awq
    --tensor-parallel-size 2
    --enable-expert-parallel
    --gpu-memory-utilization 0.85
    --max-model-len 32768
    --max-num-seqs 4
    --enable-chunked-prefill
    --no-enable-prefix-caching
    --enable-auto-tool-choice
    --tool-call-parser qwen3_xml
    --reasoning-parser qwen3
    --mm-encoder-tp-mode data
    --mm-processor-cache-type shm
    --trust-remote-code
    --host 127.0.0.1 --port "$PORT"
)

VLLM_PID=""

cleanup() {
    if [[ -n "${VLLM_PID:-}" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
        echo ">>> [cleanup] TERM-killing parent PID $VLLM_PID"
        kill -TERM "$VLLM_PID" 2>/dev/null || true
        sleep 3
    fi
    pkill -KILL -f "VLLM::EngineCore" 2>/dev/null || true
    pkill -KILL -f "VLLM::Worker" 2>/dev/null || true
    pkill -KILL -f "/vllm serve" 2>/dev/null || true
    pkill -KILL -f "vllm/bin/vllm" 2>/dev/null || true
    sleep 2
    # Wait for GPU release
    for _ in 1 2 3 4 5 6 7 8 9 10; do
        local m0=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 2>/dev/null || echo 99999)
        local m1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 2>/dev/null || echo 99999)
        if [[ ${m0:-99999} -lt 3000 ]] && [[ ${m1:-99999} -lt 3000 ]]; then
            break
        fi
        sleep 1
    done
}
trap cleanup EXIT

ensure_prod_stopped() {
    if systemctl --user is-active --quiet vllm-server; then
        echo ">>> Stopping production vllm-server.service"
        systemctl --user stop vllm-server
        sleep 5
    fi
    cleanup
}

set_power_350() {
    echo "$S1_SUDO_PW" | sudo -S nvidia-smi -i 0 -pl 350 > /dev/null 2>&1 || true
    echo "$S1_SUDO_PW" | sudo -S nvidia-smi -i 1 -pl 350 > /dev/null 2>&1 || true
    sleep 1
}

wait_for_ready() {
    local max=600
    local i=0
    echo ">>> Waiting for vLLM ready on port $PORT..."
    while ! curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null 2>&1; do
        sleep 5
        i=$((i + 5))
        if [[ $i -ge $max ]]; then
            echo "!!! vLLM did not become ready in ${max}s"
            return 1
        fi
        if [[ -n "$VLLM_PID" ]] && ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "!!! vLLM PID $VLLM_PID died during startup"
            return 1
        fi
        if [[ $((i % 30)) -eq 0 ]]; then
            echo "    ... still waiting (${i}s elapsed)"
        fi
    done
    echo ">>> vLLM ready after ${i}s"
}

bench_one_config() {
    local label="$1"
    local k="$2"
    local launch_log="$BENCH_DIR/vllm_${label}.log"
    echo ""
    echo "================================================================"
    echo "  PHASE J — $label (vLLM 0.20.x dual 3090 AWQ-Marlin, k=$k)"
    echo "================================================================"
    echo ">>> Launching vllm 0.20.x..."
    "$VENV020/bin/vllm" serve "$MODEL_AWQ" \
        "${COMMON_FLAGS[@]}" \
        --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${k}}" \
        > "$launch_log" 2>&1 &
    VLLM_PID=$!
    echo "    PID: $VLLM_PID; log: $launch_log"
    if ! wait_for_ready; then
        echo "!!! Aborting $label"
        echo "=== last 40 lines of $launch_log ==="
        tail -40 "$launch_log"
        cleanup
        VLLM_PID=""
        return 1
    fi
    # Run bench at 350W, t=0.0 only (matches Phase A k=3 winner cell)
    python3 /home/reachym/dev/reachy-agent/robot/scripts/bench_runner.py \
        --config-id "phase_j_${label}_t0.0" \
        --temperature 0.0 \
        --output "$BENCH_DIR/phase_j_${label}_t0.0.json" 2>&1 \
      | tee "$BENCH_DIR/phase_j_${label}_t0.0.stdout"
    echo ">>> Stopping vllm 0.20.x..."
    cleanup
    VLLM_PID=""
}

# Main: ensure prod stopped, set power, run bench at k=3
ensure_prod_stopped
set_power_350
bench_one_config "020x_p350_k3" 3 || echo "k=3 attempt failed"

# Restore production
echo ""
echo ">>> Restoring production vllm-server (0.19.1)..."
systemctl --user start vllm-server
sleep 10
systemctl --user status vllm-server --no-pager | head -8

echo ""
echo "=== PHASE J complete ==="
ls -la "$BENCH_DIR"/phase_j_*.json 2>&1
