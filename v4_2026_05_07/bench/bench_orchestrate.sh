#!/usr/bin/env bash
# Full-matrix bench orchestrator — Tasks #50-#54.
#
# Phases:
#   phase_a    k × power × temp = 12 configs (~25 min)
#   phase_b    TP × power × temp = 8 configs (~15 min, needs BEST_K from A)
#   phase_c    quant × power × temp = 8 configs (~15 min, needs FP8 model)
#   phase_d    spec(MTP|DFlash) × power × temp = 8 configs (needs DFlash drafter)
#   phase_e    power × 60 min stability = 2 long runs (~2 hr)
#   restore    re-enable production vllm-server
#
# Sudo: needed for nvidia-smi -pl (power cap). Password via S1_SUDO_PW env or
#       hardcoded fallback.

set -euo pipefail

S1_SUDO_PW="${S1_SUDO_PW:?Set S1_SUDO_PW env var, e.g.: S1_SUDO_PW=yourpw bash bench_orchestrate.sh phase_a}"
REPO_DIR="${REPO_DIR:-$HOME/dev/reachy-agent/robot/scripts}"
BENCH_DIR="${BENCH_DIR:-$HOME/bench_2026_05_07}"
VENV="${VENV:-/home/reachym/venvs/vllm}"
MODEL_AWQ="${MODEL_AWQ:-/home/reachym/models/qwen36-awq}"
MODEL_FP8="${MODEL_FP8:-/home/reachym/models/qwen36-fp8}"
MODEL_DFLASH="${MODEL_DFLASH:-/home/reachym/models/qwen36-dflash}"
PORT=8000

mkdir -p "$BENCH_DIR"

# ---- production-aligned env (matches vllm_serve.sh; phase_c overrides FP8) ----
export VLLM_USE_DEEP_GEMM=0
export VLLM_USE_FLASHINFER_MOE_FP16=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export OMP_NUM_THREADS=4
export VLLM_DISABLE_FP8=1   # phase_c sets to 0 explicitly

# ---- common flags (production-mirrored) ----
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
TEMPS=(0.0 0.5)
POWERS=(350 220)

cleanup() {
    # vLLM 0.19.1 spawns DETACHED multiprocessing children (EngineCore + Workers).
    # kill -TERM on parent does NOT propagate. Must pkill by name pattern
    # to nuke all descendants and free GPU memory.
    if [[ -n "${VLLM_PID:-}" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
        echo ">>> [cleanup] TERM-killing parent PID $VLLM_PID"
        kill -TERM "$VLLM_PID" 2>/dev/null || true
        sleep 3
    fi
    # Pattern KILL — catches detached children that survive parent kill
    pkill -KILL -f "VLLM::EngineCore" 2>/dev/null || true
    pkill -KILL -f "VLLM::Worker" 2>/dev/null || true
    pkill -KILL -f "/vllm serve" 2>/dev/null || true
    pkill -KILL -f "vllm/bin/vllm" 2>/dev/null || true
    sleep 2
    # Wait for GPU memory release (CUDA driver async releases on process exit)
    local mem_free=false
    for _ in 1 2 3 4 5 6 7 8 9 10 11 12; do
        local mem0
        local mem1
        mem0=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 2>/dev/null || echo 99999)
        mem1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 2>/dev/null || echo 99999)
        if [[ ${mem0:-99999} -lt 3000 ]] && [[ ${mem1:-99999} -lt 3000 ]]; then
            mem_free=true
            break
        fi
        sleep 1
    done
    if ! $mem_free; then
        echo "!!! WARNING: GPU memory not released within 12s — next launch may OOM"
        nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
    fi
}
trap cleanup EXIT

set_power() {
    local watts=$1
    echo ">>> Setting GPU power cap to ${watts}W on both cards"
    # Defensive: catch sudo failure rather than letting set -e abort the whole sweep.
    # If wrong password / nvidia-smi missing, log and continue at default power.
    local rc=0
    echo "$S1_SUDO_PW" | sudo -S nvidia-smi -i 0 -pl "$watts" > /tmp/setpower.log 2>&1 || rc=$?
    echo "$S1_SUDO_PW" | sudo -S nvidia-smi -i 1 -pl "$watts" >> /tmp/setpower.log 2>&1 || rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "!!! set_power FAILED (rc=$rc) — see /tmp/setpower.log. Continuing at current power."
        cat /tmp/setpower.log
    fi
    sleep 2
    nvidia-smi --query-gpu=index,power.limit --format=csv,noheader
}

ensure_prod_stopped() {
    if systemctl --user is-active --quiet vllm-server; then
        echo ">>> Stopping production vllm-server.service"
        systemctl --user stop vllm-server
        sleep 5
    fi
    # PREFLIGHT VRAM CLEANUP — kill ANY vLLM stragglers (orphan EngineCore/Workers
    # from previous failed bench runs) and wait for GPU memory release.
    # Must do this before launching new vLLM or it will OOM.
    echo ">>> Preflight: nuking vLLM stragglers + waiting for VRAM release"
    cleanup
}

wait_for_ready() {
    # Empirically Qwen3.6-35B-A3B AWQ TP=2 cold-start with CUDA graph capture
    # takes ~4-5 min (model load + engine init 33s + graph capture + async sched).
    # Set max to 600s — generous, since failures are obvious in vllm logs quickly.
    local max=600
    local i=0
    echo ">>> Waiting for vLLM ready on port $PORT (max ${max}s)..."
    while ! curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null 2>&1; do
        sleep 5
        i=$((i + 5))
        if [[ $i -ge $max ]]; then
            echo "!!! vLLM did not become ready in ${max}s"
            return 1
        fi
        # Detect parent death (children may linger but parent gone = no API server)
        if [[ -n "$VLLM_PID" ]] && ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "!!! vLLM PID $VLLM_PID died during startup"
            return 1
        fi
        # Periodic progress hint every 30s
        if [[ $((i % 30)) -eq 0 ]]; then
            echo "    ... still waiting (${i}s elapsed)"
        fi
    done
    echo ">>> vLLM ready after ${i}s"
}

launch_vllm() {
    local label="$1"
    shift
    local logfile="$BENCH_DIR/vllm_${label}.log"
    echo ">>> Launching vLLM ($label)"
    echo "    Log: $logfile"
    "$VENV/bin/vllm" serve "$@" > "$logfile" 2>&1 &
    VLLM_PID=$!
    echo "    PID: $VLLM_PID"
    if ! wait_for_ready; then
        echo "!!! Aborting this config; check $logfile for vLLM startup errors"
        cleanup
        VLLM_PID=""
        return 1
    fi
    return 0
}

stop_vllm() {
    # Use the same comprehensive cleanup as the EXIT trap.
    # Kills detached vLLM children (EngineCore + Workers) + waits for GPU release.
    echo ">>> Stopping bench vLLM (PID ${VLLM_PID:-none}) + descendants"
    cleanup
    VLLM_PID=""
}

# =========================================================================
# Phase A: k × power × temp = 12 configs
# Inner loop: temp shares vLLM (request-level); k requires vLLM restart
# Outer loop: power (sudo nvidia-smi -pl)
# Wall: ~25 min
# =========================================================================
phase_a_k_sweep() {
    ensure_prod_stopped
    echo ""
    echo "================================================================"
    echo "  PHASE A — k × power × temp = 3 × 2 × 2 = 12 configs"
    echo "================================================================"
    for POWER in "${POWERS[@]}"; do
        set_power "$POWER"
        for K in 1 2 3; do
            local label="p${POWER}_k${K}"
            if launch_vllm "$label" "$MODEL_AWQ" \
                "${COMMON_FLAGS[@]}" \
                --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${K}}"; then
                for TEMP in "${TEMPS[@]}"; do
                    local cfg="${label}_t${TEMP}"
                    echo ""
                    echo "  --- bench config: $cfg ---"
                    python3 "$REPO_DIR/bench_runner.py" \
                        --config-id "$cfg" \
                        --temperature "$TEMP" \
                        --output "$BENCH_DIR/phase_a_${cfg}.json" 2>&1 \
                      | tee "$BENCH_DIR/phase_a_${cfg}.stdout"
                done
                stop_vllm
            else
                echo "!!! Skipping $label (vLLM startup failed)"
            fi
        done
    done
    set_power 350   # restore default
    echo ""
    echo "=== PHASE A complete ==="
    ls -la "$BENCH_DIR"/phase_a_*.json 2>&1
}

# =========================================================================
# Phase B: TP × power × temp = 2 × 2 × 2 = 8 configs
# BEST_K env var picks the winning k from Phase A (default 2)
# TP=1 uses GPU 0 only (CUDA_VISIBLE_DEVICES=0), drops --enable-expert-parallel,
# and reduces max-model-len to 8192 (fits 24GB).
# =========================================================================
phase_b_tp() {
    ensure_prod_stopped
    local K="${BEST_K:-2}"
    echo ""
    echo "================================================================"
    echo "  PHASE B — TP × power × temp = 2 × 2 × 2 = 8 configs (k=$K)"
    echo "================================================================"

    # Build TP=1 flags. Single 3090 24GB with 35B-A3B AWQ + MTP k=3 spec decoding
    # is VERY tight (~24GB needed, exact margin). Empirical OOM analysis 2026-05-07:
    #   weights ~19GB + MTP k=3 head ~1.5GB + activations ~2GB + CUDA graph ~1GB
    #   ≈ 23.5GB → no safety margin → random OOM
    #
    # Five-layer survival kit:
    #   --gpu-memory-utilization 0.85 → 0.95   (no Whisper on this card)
    #   --max-num-seqs 4 → 1                   (concurrency=1 anyway)
    #   --max-model-len 32768 → 4096           (lower KV ceiling)
    #   --enforce-eager                        (skip CUDA graph capture, saves ~1GB)
    #   PYTORCH_ALLOC_CONF=expandable_segments (anti-fragmentation, vLLM-recommended)
    # Also drop --enable-expert-parallel (no MoE parallelism on single card)
    # and --mm-encoder-tp-mode (no TP at all).
    #
    # NOTE: --enforce-eager makes TP=1 skip CUDA graphs that TP=2 uses.
    # This is necessary for the model to fit at all, but does mean TP=1 results
    # include both "no parallelism" and "no CUDA graph" effects. We document both.
    TP1_FLAGS=()
    local skip_next=false
    for f in "${COMMON_FLAGS[@]}"; do
        if $skip_next; then skip_next=false; continue; fi
        case "$f" in
            "--tensor-parallel-size") TP1_FLAGS+=("--tensor-parallel-size" "1"); skip_next=true ;;
            "--enable-expert-parallel") ;;   # drop
            "--gpu-memory-utilization") TP1_FLAGS+=("--gpu-memory-utilization" "0.95"); skip_next=true ;;
            "--max-model-len") TP1_FLAGS+=("--max-model-len" "4096"); skip_next=true ;;
            "--max-num-seqs") TP1_FLAGS+=("--max-num-seqs" "1"); skip_next=true ;;
            "--mm-encoder-tp-mode") skip_next=true ;;   # drop
            *) TP1_FLAGS+=("$f") ;;
        esac
    done
    TP1_FLAGS+=(--enforce-eager)
    echo ">>> TP=1 flags (overrides for single-card memory budget): ${TP1_FLAGS[*]}"

    for POWER in "${POWERS[@]}"; do
        set_power "$POWER"
        # TP=2
        local label="p${POWER}_tp2_k${K}"
        if launch_vllm "$label" "$MODEL_AWQ" \
            "${COMMON_FLAGS[@]}" \
            --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${K}}"; then
            for TEMP in "${TEMPS[@]}"; do
                local cfg="${label}_t${TEMP}"
                python3 "$REPO_DIR/bench_runner.py" \
                    --config-id "$cfg" --temperature "$TEMP" \
                    --output "$BENCH_DIR/phase_b_${cfg}.json" 2>&1 \
                  | tee "$BENCH_DIR/phase_b_${cfg}.stdout"
            done
            stop_vllm
        fi
        # TP=1 (GPU 0 only) — try k=$K first, fall back to k=1 if OOM
        export CUDA_VISIBLE_DEVICES=0
        export PYTORCH_ALLOC_CONF=expandable_segments:True
        local tp1_succeeded=false
        for K_TRY in "$K" 1; do
            # Skip k=1 fallback if K already 1 and first attempt succeeded/done
            if $tp1_succeeded; then break; fi
            if [[ "$K_TRY" == "1" ]] && [[ "$K" == "1" ]]; then break; fi
            label="p${POWER}_tp1_k${K_TRY}"
            if launch_vllm "$label" "$MODEL_AWQ" \
                "${TP1_FLAGS[@]}" \
                --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${K_TRY}}"; then
                tp1_succeeded=true
                for TEMP in "${TEMPS[@]}"; do
                    local cfg="${label}_t${TEMP}"
                    python3 "$REPO_DIR/bench_runner.py" \
                        --config-id "$cfg" --temperature "$TEMP" \
                        --output "$BENCH_DIR/phase_b_${cfg}.json" 2>&1 \
                      | tee "$BENCH_DIR/phase_b_${cfg}.stdout"
                done
                stop_vllm
            else
                echo "!!! TP=1 launch failed at k=$K_TRY (OOM expected); trying k=1 fallback"
            fi
        done
        if ! $tp1_succeeded; then
            echo "!!! TP=1 FAILED entirely at power ${POWER}W — single 3090 cannot fit this model+spec config"
        fi
        unset CUDA_VISIBLE_DEVICES
        unset PYTORCH_ALLOC_CONF
    done
    set_power 350
    echo ""
    echo "=== PHASE B complete ==="
}

# =========================================================================
# Phase C: quant × power × temp = 2 × 2 × 2 = 8 configs
# Needs Qwen3.6-35B-A3B-FP8 downloaded
# VLLM_DISABLE_FP8 must be unset for FP8 path
# =========================================================================
phase_c_quant() {
    if [[ ! -d "$MODEL_FP8" ]]; then
        echo "!!! FP8 model missing at $MODEL_FP8"
        echo "    Run: $VENV/bin/python $REPO_DIR/download_models.py fp8"
        return 1
    fi
    ensure_prod_stopped
    local K="${BEST_K:-2}"
    echo ""
    echo "================================================================"
    echo "  PHASE C — quant × power × temp = 2 × 2 × 2 = 8 configs (k=$K)"
    echo "================================================================"
    for POWER in "${POWERS[@]}"; do
        set_power "$POWER"
        # AWQ
        local label="p${POWER}_awq_k${K}"
        if launch_vllm "$label" "$MODEL_AWQ" \
            "${COMMON_FLAGS[@]}" \
            --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${K}}"; then
            for TEMP in "${TEMPS[@]}"; do
                local cfg="${label}_t${TEMP}"
                python3 "$REPO_DIR/bench_runner.py" \
                    --config-id "$cfg" --temperature "$TEMP" \
                    --output "$BENCH_DIR/phase_c_${cfg}.json" 2>&1 \
                  | tee "$BENCH_DIR/phase_c_${cfg}.stdout"
            done
            stop_vllm
        fi
        # FP8 (override env, swap model). FP8 weights are 8 bits/param vs AWQ's 4
        # bits/param → ~2× weight footprint. Production 0.85 mem-util fits AWQ but
        # NOT FP8 (KV cache allocation OOM at ~22.5GB used). Bump to 0.95 for FP8.
        # Last --gpu-memory-utilization wins in argparse (verified with vllm 0.19.1).
        label="p${POWER}_fp8_k${K}"
        if VLLM_DISABLE_FP8=0 launch_vllm "$label" "$MODEL_FP8" \
            "${COMMON_FLAGS[@]}" \
            --gpu-memory-utilization 0.95 \
            --quantization fp8 \
            --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${K}}"; then
            for TEMP in "${TEMPS[@]}"; do
                local cfg="${label}_t${TEMP}"
                python3 "$REPO_DIR/bench_runner.py" \
                    --config-id "$cfg" --temperature "$TEMP" \
                    --output "$BENCH_DIR/phase_c_${cfg}.json" 2>&1 \
                  | tee "$BENCH_DIR/phase_c_${cfg}.stdout"
            done
            stop_vllm
        else
            echo "!!! FP8 launch failed at power ${POWER}W (expected on Ampere SM 8.6 per vllm#40124)"
        fi
    done
    set_power 350
    echo ""
    echo "=== PHASE C complete ==="
}

# =========================================================================
# Phase D: spec(DFlash vs MTP) × power × temp = 2 × 2 × 2 = 8 configs
# Needs z-lab/Qwen3.6-35B-A3B-DFlash drafter and possibly vLLM source build
# (PR #40898) — try in-tree first; if "unknown method", fall back to building
# =========================================================================
phase_d_dflash() {
    if [[ ! -d "$MODEL_DFLASH" ]]; then
        echo "!!! DFlash drafter missing at $MODEL_DFLASH"
        echo "    Run: $VENV/bin/python $REPO_DIR/download_models.py dflash"
        return 1
    fi
    ensure_prod_stopped
    local K="${BEST_K:-2}"
    echo ""
    echo "================================================================"
    echo "  PHASE D — DFlash vs MTP-k${K} × power × temp = 8 configs"
    echo "================================================================"
    for POWER in "${POWERS[@]}"; do
        set_power "$POWER"
        # MTP baseline (winning k)
        local label="p${POWER}_mtp_k${K}"
        if launch_vllm "$label" "$MODEL_AWQ" \
            "${COMMON_FLAGS[@]}" \
            --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${K}}"; then
            for TEMP in "${TEMPS[@]}"; do
                local cfg="${label}_t${TEMP}"
                python3 "$REPO_DIR/bench_runner.py" \
                    --config-id "$cfg" --temperature "$TEMP" \
                    --output "$BENCH_DIR/phase_d_${cfg}.json"
            done
            stop_vllm
        fi
        # DFlash
        label="p${POWER}_dflash"
        if launch_vllm "$label" "$MODEL_AWQ" \
            "${COMMON_FLAGS[@]}" \
            --speculative-config "{\"method\":\"dflash\",\"model\":\"$MODEL_DFLASH\",\"num_speculative_tokens\":4}"; then
            for TEMP in "${TEMPS[@]}"; do
                local cfg="${label}_t${TEMP}"
                python3 "$REPO_DIR/bench_runner.py" \
                    --config-id "$cfg" --temperature "$TEMP" \
                    --output "$BENCH_DIR/phase_d_${cfg}.json"
            done
            stop_vllm
        else
            echo "!!! DFlash launch failed — likely unsupported in vllm 0.19.1, needs PR #40898 build"
        fi
    done
    set_power 350
    echo ""
    echo "=== PHASE D complete ==="
}

# =========================================================================
# Phase E: long-time stability × power = 2 long runs (~1 hr each)
# Use BEST_K and BEST_TEMP (default 0.5 — closer to real voice workload)
# =========================================================================
phase_e_stability() {
    ensure_prod_stopped
    # Default to k=3 (Phase A winner 2026-05-07). Override via BEST_K env if needed.
    local K="${BEST_K:-3}"
    local TEMP="${BEST_TEMP:-0.5}"
    local DURATION="${DURATION:-60}"
    echo ""
    echo "================================================================"
    echo "  PHASE E — stability × power = 2 × ${DURATION} min"
    echo "================================================================"
    for POWER in "${POWERS[@]}"; do
        set_power "$POWER"
        local label="p${POWER}_stability_k${K}"
        if launch_vllm "$label" "$MODEL_AWQ" \
            "${COMMON_FLAGS[@]}" \
            --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${K}}"; then
            python3 "$REPO_DIR/bench_stability.py" \
                --duration-min "$DURATION" \
                --temperature "$TEMP" \
                --output "$BENCH_DIR/phase_e_${label}_t${TEMP}.json" 2>&1 \
              | tee "$BENCH_DIR/phase_e_${label}_t${TEMP}.stdout"
            stop_vllm
        fi
    done
    set_power 350
    echo ""
    echo "=== PHASE E complete ==="
}

# =========================================================================
# Phase H: AWQ at gpu-memory-utilization=0.92 (matched control vs FP8 0.92).
# Tests Finding 2 (FP8 > AWQ) properly: same mem-util, same model class.
# OUTCOME (post-run, see analysis/analyze_final.py): RETRACTED — all 4 cells
# Welch p > 0.6 (NS). The +2.7% gap from v3 was confounded by 0.85 vs 0.92.
# =========================================================================
phase_h_awq_092() {
    ensure_prod_stopped
    local K="${BEST_K:-3}"
    echo ""
    echo "================================================================"
    echo "  PHASE H -- AWQ @ gpu-mem-util=0.92 (matched FP8 control), k=$K"
    echo "================================================================"
    for POWER in "${POWERS[@]}"; do
        set_power "$POWER"
        local label="p${POWER}_awq092_k${K}"
        if launch_vllm "$label" "$MODEL_AWQ" \
            "${COMMON_FLAGS[@]}" \
            --gpu-memory-utilization 0.92 \
            --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${K}}"; then
            for TEMP in "${TEMPS[@]}"; do
                local cfg="${label}_t${TEMP}"
                python3 "$REPO_DIR/bench_runner.py" \
                    --config-id "$cfg" --temperature "$TEMP" \
                    --output "$BENCH_DIR/phase_h_${cfg}.json" 2>&1 \
                  | tee "$BENCH_DIR/phase_h_${cfg}.stdout"
            done
            stop_vllm
        fi
    done
    set_power 350
    echo ""
    echo "=== PHASE H complete ==="
}

# =========================================================================
# Phase I: FP8 gpu-memory-utilization sweep (0.86 -> 0.88 -> 0.90).
# Find true minimum that fits FP8 + Whisper on dual 3090. We know 0.85 OOMs
# (KV cache), 0.92 fits. Narrow the bound.
# =========================================================================
phase_i_fp8_memutil_sweep() {
    if [[ ! -d "$MODEL_FP8" ]]; then
        echo "!!! FP8 model missing at $MODEL_FP8"
        return 1
    fi
    ensure_prod_stopped
    local K="${BEST_K:-3}"
    echo ""
    echo "================================================================"
    echo "  PHASE I -- FP8 gpu-mem-util sweep at 350W, t=0.0, k=$K"
    echo "================================================================"
    set_power 350
    for MEMUTIL in 0.86 0.88 0.90; do
        local label_tag="${MEMUTIL//./}"
        local label="p350_fp8_mu${label_tag}_k${K}"
        echo ""
        echo "  --- Trying FP8 at gpu-memory-utilization=$MEMUTIL ---"
        if VLLM_DISABLE_FP8=0 launch_vllm "$label" "$MODEL_FP8" \
            "${COMMON_FLAGS[@]}" \
            --gpu-memory-utilization "$MEMUTIL" \
            --quantization fp8 \
            --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${K}}"; then
            local cfg="${label}_t0.0"
            python3 "$REPO_DIR/bench_runner.py" \
                --config-id "$cfg" --temperature 0.0 \
                --output "$BENCH_DIR/phase_i_${cfg}.json" 2>&1 \
              | tee "$BENCH_DIR/phase_i_${cfg}.stdout"
            stop_vllm
        else
            echo "!!! FP8 OOM at mem-util=$MEMUTIL -- minimum is > $MEMUTIL"
        fi
    done
    set_power 350
    echo ""
    echo "=== PHASE I complete ==="
}

# =========================================================================
# Phase C FP8-only retry: gpu-mem-util 0.92 (max with Whisper coexistence).
# Original phase_c at 0.95 fails because Whisper holds 1.4 GB on GPU 1, leaving
# only 21.76 GiB free vs 22.38 GiB requested. 0.92 × 23.56 = 21.68 GiB ≤ 21.76 free.
# Tight but should fit. If still OOM, we have definitive answer "FP8 not viable
# in production-class config on Ampere SM 8.6 dual 3090".
# =========================================================================
phase_c_fp8_only() {
    if [[ ! -d "$MODEL_FP8" ]]; then
        echo "!!! FP8 model missing at $MODEL_FP8"
        return 1
    fi
    ensure_prod_stopped
    local K="${BEST_K:-3}"
    echo ""
    echo "================================================================"
    echo "  PHASE C FP8 ONLY — gpu-mem-util 0.92 retry × power × temp = 4 configs (k=$K)"
    echo "================================================================"
    for POWER in "${POWERS[@]}"; do
        set_power "$POWER"
        local label="p${POWER}_fp8_092_k${K}"
        if VLLM_DISABLE_FP8=0 launch_vllm "$label" "$MODEL_FP8" \
            "${COMMON_FLAGS[@]}" \
            --gpu-memory-utilization 0.92 \
            --quantization fp8 \
            --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${K}}"; then
            for TEMP in "${TEMPS[@]}"; do
                local cfg="${label}_t${TEMP}"
                python3 "$REPO_DIR/bench_runner.py" \
                    --config-id "$cfg" --temperature "$TEMP" \
                    --output "$BENCH_DIR/phase_c_${cfg}.json" 2>&1 \
                  | tee "$BENCH_DIR/phase_c_${cfg}.stdout"
            done
            stop_vllm
        else
            echo "!!! FP8 launch failed at power ${POWER}W with mem-util 0.92 (max with Whisper) — definitive: not viable in prod config"
        fi
    done
    set_power 350
    echo ""
    echo "=== PHASE C FP8 ONLY complete ==="
}

# ---- Restore production ----
restore() {
    set_power 350
    echo ">>> Re-enabling and starting production vllm-server.service"
    systemctl --user start vllm-server
    sleep 10
    systemctl --user status vllm-server --no-pager | head -8
}

case "${1:-}" in
    phase_a) phase_a_k_sweep ;;
    phase_b) phase_b_tp ;;
    phase_c) phase_c_quant ;;
    phase_c_fp8) phase_c_fp8_only ;;
    phase_d) phase_d_dflash ;;
    phase_e) phase_e_stability ;;
    phase_h) phase_h_awq_092 ;;
    phase_i) phase_i_fp8_memutil_sweep ;;
    phase_h_then_i) phase_h_awq_092 && phase_i_fp8_memutil_sweep ;;
    restore) restore ;;
    set_power) set_power "${2:-350}" ;;
    *)
        echo "Usage: $0 {phase_a|phase_b|phase_c|phase_c_fp8|phase_d|phase_e|restore|set_power <W>}"
        echo "Env: BEST_K (default 3 for B/C/D/E), BEST_TEMP (default 0.5 for E),"
        echo "     DURATION (default 60 for E)"
        exit 1
        ;;
esac
