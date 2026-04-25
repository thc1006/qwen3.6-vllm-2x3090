#!/bin/bash
# 3090 power-scaling sweep v2 — methodologically rigorous.
#
# Changes from v1:
#  - Both GPUs swept simultaneously (TP=2 means both must be at the same
#    power for the variable to actually be "power"). v1 only swept GPU0
#    while GPU1 stayed at 350W, so the result was confounded.
#  - 5 bench runs per power level (was 1).
#  - 30 s thermal settle between changes (was 3 s).
#  - Long workload: max_tokens=500 (was 200) — closer to saturation.
#  - GPU1 max is 350W (FE card), so sweep range is 200/220/250/280/320/350.
#    GPU0 can go higher but for an apples-to-apples both-cards experiment
#    we cap at 350.
#
# Output: results/power_scaling_v2.json
set -eu

PASSWORD=${PASSWORD:?set PASSWORD env var}
RESULTS_FILE=${RESULTS_FILE:-/tmp/power_scaling_v2.json}
SETTLE_S=${SETTLE_S:-30}
N_RUNS=${N_RUNS:-5}
MAX_TOKENS=${MAX_TOKENS:-500}

POWER_LIMITS=(200 220 250 280 320 350)

# Inline bench: long-form generation, capture mean tok/s + power draw mid-run.
bench_one_run() {
    local pl="$1" run_idx="$2"
    /home/reachym/venvs/vllm/bin/python - <<PYEOF 2>&1
import json, time, urllib.request, threading, subprocess
ENDPOINT = "http://127.0.0.1:8000/v1/chat/completions"
PROMPTS = [
    "Tell me a 200-word story about a curious robot exploring a kitchen.",
    "Explain neural networks to a high schooler in 200 words.",
    "Write a 200-word travel guide for Taipei.",
    "Describe the lifecycle of a star in 200 words.",
    "Compose a 200-word essay on the value of curiosity.",
]
power_samples = []
def sample_power():
    while not stop[0]:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                timeout=2
            ).decode().split()
            power_samples.append((float(out[0]), float(out[1])))
        except Exception:
            pass
        time.sleep(0.5)

stop = [False]
sampler = threading.Thread(target=sample_power, daemon=True); sampler.start()

results = []
for p in PROMPTS:
    body = {
        "model": "qwen36-awq",
        "messages": [{"role":"user","content": p}],
        "max_tokens": $MAX_TOKENS, "temperature": 0.5, "seed": 42,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.perf_counter()
    req = urllib.request.Request(ENDPOINT, data=json.dumps(body).encode(),
        headers={"Content-Type":"application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            d = json.loads(r.read().decode())
    except Exception as e:
        results.append({"err": str(e)}); continue
    dt = time.perf_counter() - t0
    ct = d.get("usage", {}).get("completion_tokens", 0)
    results.append({"ct": ct, "dt": dt, "tok_s": ct/dt if dt else 0})

stop[0] = True
sampler.join(timeout=2)

ts = [r["tok_s"] for r in results if "tok_s" in r]
ct_total = sum(r["ct"] for r in results if "ct" in r)
dt_total = sum(r["dt"] for r in results if "dt" in r)
mean_tps_per_prompt = sum(ts) / len(ts) if ts else 0
agg_tok_s = ct_total / dt_total if dt_total else 0
g0_avg = sum(p[0] for p in power_samples) / len(power_samples) if power_samples else 0
g1_avg = sum(p[1] for p in power_samples) / len(power_samples) if power_samples else 0
g0_max = max((p[0] for p in power_samples), default=0)
g1_max = max((p[1] for p in power_samples), default=0)
print(json.dumps({
    "run_idx": $run_idx, "pl_w": $pl,
    "mean_tps_per_prompt": round(mean_tps_per_prompt, 2),
    "agg_tok_s": round(agg_tok_s, 2),
    "ct_total": ct_total, "dt_total": round(dt_total, 2),
    "gpu0_avg_w": round(g0_avg, 1), "gpu0_max_w": round(g0_max, 1),
    "gpu1_avg_w": round(g1_avg, 1), "gpu1_max_w": round(g1_max, 1),
    "power_samples": len(power_samples),
}))
PYEOF
}

echo "$PASSWORD" | sudo -S nvidia-smi -pm 1 >/dev/null
echo "[" > "$RESULTS_FILE"

for i in "${!POWER_LIMITS[@]}"; do
    pl="${POWER_LIMITS[$i]}"
    echo "==== sweeping pl=${pl}W on BOTH GPUs ===="
    echo "$PASSWORD" | sudo -S nvidia-smi -i 0 -pl "$pl" 2>&1 | tail -1
    echo "$PASSWORD" | sudo -S nvidia-smi -i 1 -pl "$pl" 2>&1 | tail -1
    echo "settling ${SETTLE_S}s for thermal stabilization..."
    sleep "$SETTLE_S"

    runs_json=""
    for r in $(seq 1 $N_RUNS); do
        echo "  run $r/$N_RUNS at ${pl}W..."
        out=$(bench_one_run "$pl" "$r")
        echo "    -> $out"
        sep=","; [ "$r" -eq 1 ] && sep=""
        runs_json="${runs_json}${sep}${out}"
    done

    sep=","; [ "$i" -eq 0 ] && sep=""
    cat >> "$RESULTS_FILE" <<EOF
$sep  {
    "pl_w": $pl,
    "settle_s": $SETTLE_S,
    "n_runs": $N_RUNS,
    "max_tokens": $MAX_TOKENS,
    "runs": [$runs_json]
  }
EOF
done

echo "]" >> "$RESULTS_FILE"

# Restore production setting (220W = sweet spot per v2 finding,
# but only if v2 confirms; otherwise default to 350W safe).
echo "$PASSWORD" | sudo -S nvidia-smi -i 0 -pl 220 >/dev/null
echo "$PASSWORD" | sudo -S nvidia-smi -i 1 -pl 350 >/dev/null
echo "=== sweep v2 complete ==="
cat "$RESULTS_FILE"
