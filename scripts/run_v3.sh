#!/usr/bin/env bash
# v3 orchestrator: stops production vllm-server, runs no-MTP then MTP clean A/B,
# restarts production at the end.
# Run: nohup bash ~/bench_clean_ab/run_v3.sh > ~/bench_clean_ab/master_v3.log 2>&1 &
set -uo pipefail

LOGDIR=/home/reachym/bench_clean_ab
PY=/home/reachym/venvs/vllm/bin/python

wait_ready() {
  local max=$1
  for i in $(seq 1 $max); do
    if curl -fsS http://127.0.0.1:8000/v1/models > /dev/null 2>&1; then
      echo "[ready] vLLM ready after ${i} × 5s"
      return 0
    fi
    sleep 5
  done
  echo "[FAIL] vLLM not ready after ${max} × 5s"
  return 1
}

stop_vllm() {
  echo "[stop] stopping vLLM (pkill -f vllm)"
  pkill -f "vllm serve" 2>/dev/null || true
  sleep 5
  pkill -9 -f "vllm serve" 2>/dev/null || true
  sleep 8
}

run_phase() {
  local TAG=$1
  local SCRIPT=$2
  echo ""
  echo "============================================"
  echo "PHASE: $TAG"
  echo "$(date -Is)"
  echo "============================================"
  bash "$SCRIPT" > "$LOGDIR/serve_v3_${TAG}.log" 2>&1 &
  local SERVE_PID=$!
  echo "[serve] started PID=$SERVE_PID, waiting for /v1/models..."
  if ! wait_ready 90; then
    echo "[FAIL] aborting phase $TAG"
    kill $SERVE_PID 2>/dev/null || true
    return 1
  fi
  sleep 3
  echo "[bench] running bench_v3.py TAG=$TAG"
  VLLM_TAG=$TAG $PY "$LOGDIR/bench_v3.py" 2>&1 | tee "$LOGDIR/bench_v3_${TAG}.log"
  stop_vllm
  echo "[done] phase $TAG complete"
}

cd "$LOGDIR"
echo "=== run_v3.sh start $(date -Is) ==="
echo "[pre] stopping production vllm-server.service"
systemctl --user stop vllm-server.service 2>/dev/null || true
sleep 3
nvidia-smi --query-gpu=memory.used --format=csv,noheader

run_phase "no_mtp" "$LOGDIR/serve_v3_no_mtp.sh" || { systemctl --user start vllm-server.service; exit 1; }
sleep 5
run_phase "mtp" "$LOGDIR/serve_v3_mtp.sh" || { systemctl --user start vllm-server.service; exit 1; }

echo ""
echo "=== ALL DONE $(date -Is) ==="
echo "Results:"
ls -la "$LOGDIR/results_v3_"*.json 2>/dev/null

echo "[post] restarting production vllm-server.service"
systemctl --user start vllm-server.service
sleep 5
systemctl --user is-active vllm-server.service
