#!/bin/bash
# 3090 power-scaling sweep for Qwen3.6-35B-A3B-AWQ on vLLM TP=2.
#
# Question: where is the perf-per-watt sweet spot for MoE inference on 3090?
# Existing public bench (Himesh, 2025) found 220W sweet spot — but for dense
# models. MoE expert routing has lower compute density per token, may saturate
# at lower power.
#
# Sweeps GPU0 power limit at: 200, 220, 250, 280, 320, 350, 390, 420, 450 W.
# (GPU1 stays at its max 350W because it's a different model with smaller cap.)
#
# Records: mean tok/s + max GPU temp + actual avg power draw during run.
# Output: results/power_scaling.json
#
# Usage:  PASSWORD=123@nctu bash bench_power_scaling.sh
# Requires sudo for nvidia-smi -pl.
set -eu

PASSWORD=${PASSWORD:?set PASSWORD env var}
BENCH_SCRIPT=${BENCH_SCRIPT:-/home/reachym/dev/reachy-agent/robot/scripts/bench_vllm_dialog.py}
RESULTS_FILE=${RESULTS_FILE:-/tmp/power_scaling.json}

POWER_LIMITS=(200 220 250 280 320 350 390 420 450)

# Ensure GPU1 stays at its 350W max for the whole sweep (don't accidentally lower)
echo "$PASSWORD" | sudo -S nvidia-smi -i 1 -pl 350 >/dev/null

echo "[" > "$RESULTS_FILE"

for i in "${!POWER_LIMITS[@]}"; do
    pl="${POWER_LIMITS[$i]}"
    echo "=== GPU0 power limit = ${pl}W ==="
    echo "$PASSWORD" | sudo -S nvidia-smi -i 0 -pl "$pl" 2>&1 | tail -1

    # Allow GPU to settle
    sleep 3

    # Sample idle baseline (5 samples avg)
    idle_temp=$(nvidia-smi -i 0 --query-gpu=temperature.gpu --format=csv,noheader,nounits)

    # Run dialog bench (capture only the summary line)
    start_temp=$(nvidia-smi -i 0 --query-gpu=temperature.gpu --format=csv,noheader,nounits)
    bench_out=$(/home/reachym/venvs/vllm/bin/python "$BENCH_SCRIPT" 2>&1 | tail -3)
    end_temp=$(nvidia-smi -i 0 --query-gpu=temperature.gpu --format=csv,noheader,nounits)

    # Sample power during a quick re-run (best effort)
    power_during=$(nvidia-smi -i 0 --query-gpu=power.draw --format=csv,noheader,nounits | head -1)

    # Parse mean tok/s from "mean=XX.X min=YY.Y max=ZZ.Z"
    mean=$(echo "$bench_out" | grep -oP 'mean=\K[0-9.]+' | head -1)
    bmin=$(echo "$bench_out" | grep -oP 'min=\K[0-9.]+' | head -1)
    bmax=$(echo "$bench_out" | grep -oP 'max=\K[0-9.]+' | head -1)

    # Append JSON record
    sep=","; [ "$i" -eq 0 ] && sep=""
    cat >> "$RESULTS_FILE" <<EOF
$sep  {
    "power_limit_w": $pl,
    "mean_tok_s": ${mean:-null},
    "min_tok_s": ${bmin:-null},
    "max_tok_s": ${bmax:-null},
    "start_temp_c": $start_temp,
    "end_temp_c": $end_temp,
    "power_during_w": ${power_during:-null}
  }
EOF
done

echo "]" >> "$RESULTS_FILE"

# Restore GPU0 to 450W (production setting)
echo "$PASSWORD" | sudo -S nvidia-smi -i 0 -pl 450 >/dev/null
echo "=== sweep complete ==="
cat "$RESULTS_FILE"
