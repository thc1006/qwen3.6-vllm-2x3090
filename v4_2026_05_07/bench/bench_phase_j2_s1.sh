#!/usr/bin/env bash
# Phase J.2: vLLM 0.19.1 baseline WITHOUT VLLM_USE_FLASHINFER_MOE_FP16=1
# (matched control vs Phase J 0.20.x which had this env var unset).
# Establishes whether the +1.6% TPOT in Phase J came from version or backend.

set -euo pipefail

S1_SUDO_PW="${S1_SUDO_PW:?Set S1_SUDO_PW env var}"
VENV019="/home/reachym/venvs/vllm"   # production 0.19.1 venv
MODEL_AWQ="/home/reachym/models/qwen36-awq"
BENCH_DIR="$HOME/bench_2026_05_07"
PORT=8000

mkdir -p "$BENCH_DIR"

# Same env as Phase A EXCEPT VLLM_USE_FLASHINFER_MOE_FP16 — UNSET (matched 0.20.x)
export VLLM_USE_DEEP_GEMM=0
unset VLLM_USE_FLASHINFER_MOE_FP16
export VLLM_USE_FLASHINFER_SAMPLER=0
export OMP_NUM_THREADS=4
export VLLM_DISABLE_FP8=1

# Same flags as production / Phase A k=3
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
        kill -TERM "$VLLM_PID" 2>/dev/null || true
        sleep 3
    fi
    pkill -KILL -f "VLLM::EngineCore" 2>/dev/null || true
    pkill -KILL -f "VLLM::Worker" 2>/dev/null || true
    pkill -KILL -f "/vllm serve" 2>/dev/null || true
    pkill -KILL -f "vllm/bin/vllm" 2>/dev/null || true
    sleep 2
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

# Stop production
if systemctl --user is-active --quiet vllm-server; then
    echo ">>> Stopping production vllm-server.service"
    systemctl --user stop vllm-server
    sleep 5
fi
cleanup

echo "$S1_SUDO_PW" | sudo -S nvidia-smi -i 0 -pl 350 > /dev/null 2>&1 || true
echo "$S1_SUDO_PW" | sudo -S nvidia-smi -i 1 -pl 350 > /dev/null 2>&1 || true
sleep 1

label="019x_unset_p350_k3"
launch_log="$BENCH_DIR/vllm_${label}.log"
echo ""
echo "================================================================"
echo "  PHASE J.2 — vLLM 0.19.1 with VLLM_USE_FLASHINFER_MOE_FP16 unset"
echo "================================================================"
echo ">>> Launching vllm 0.19.1..."
"$VENV019/bin/vllm" serve "$MODEL_AWQ" \
    "${COMMON_FLAGS[@]}" \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
    > "$launch_log" 2>&1 &
VLLM_PID=$!
echo "    PID: $VLLM_PID; log: $launch_log"

# Wait for ready
i=0; max=600
echo ">>> Waiting for vLLM ready..."
while ! curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null 2>&1; do
    sleep 5
    i=$((i + 5))
    if [[ $i -ge $max ]]; then
        echo "!!! vLLM did not become ready in ${max}s"
        tail -40 "$launch_log"
        exit 1
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "!!! vLLM died during startup"
        tail -40 "$launch_log"
        exit 1
    fi
    if [[ $((i % 30)) -eq 0 ]]; then
        echo "    ... ${i}s elapsed"
    fi
done
echo ">>> vLLM ready after ${i}s"

# Run bench: same as Phase A k=3 t=0.0
python3 /home/reachym/dev/reachy-agent/robot/scripts/bench_runner.py \
    --config-id "phase_j2_${label}_t0.0" \
    --temperature 0.0 \
    --output "$BENCH_DIR/phase_j2_${label}_t0.0.json" 2>&1 \
  | tee "$BENCH_DIR/phase_j2_${label}_t0.0.stdout"

echo ">>> Stopping vllm 0.19.1 (test instance)..."
cleanup
VLLM_PID=""

# Restore production
echo ""
echo ">>> Restoring production vllm-server (0.19.1, VLLM_USE_FLASHINFER_MOE_FP16=1 default)..."
systemctl --user start vllm-server
sleep 10
systemctl --user status vllm-server --no-pager | head -8

echo ""
echo "=== PHASE J.2 complete ==="
ls -la "$BENCH_DIR"/phase_j2_*.json 2>&1
