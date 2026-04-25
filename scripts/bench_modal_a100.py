#!/usr/bin/env python3
"""Cross-hardware MTP bench on 2x A100 80GB via Modal.

Why this exists
---------------
Our v2 bench established that on 2x RTX 3090 (Ampere, GDDR6X 936 GB/s,
PCIe Gen4 x8 interconnect), vLLM `--speculative-config method=mtp k=1`
is mean -12% vs no-MTP, with variance ~65x larger. The open question
is whether this is a property of:

  (a) consumer-class GPU memory bandwidth (GDDR6X), or
  (b) the inter-GPU TP allreduce interconnect being the bottleneck
      (PCIe ~16 GB/s vs HGX-class NVLink ~600 GB/s), or
  (c) the 3B-active MoE itself, regardless of hardware.

Two-variable trap
-----------------
Naively running this on 2x A100-80GB SXM (Modal default) changes BOTH
GPU memory type (GDDR6X → HBM2e) AND interconnect (PCIe → NVLink) at
the same time, so a "MTP flips to positive on A100" result cannot
distinguish (a) from (b).

To disentangle, this script supports a 4-condition matrix:

    no-MTP / MTP   ×   NVLink-default / NCCL_P2P_DISABLE=1 (PCIe-forced)

The PCIe-forced variant runs on the same A100 hardware but downgrades
TP allreduce to PCIe path, isolating interconnect as a control
variable. Combined with the 3090 baseline (which is GDDR6X + PCIe),
the 4 conditions span:

    3090           : GDDR6X  + PCIe   (existing)
    A100 NVLink    : HBM2e   + NVLink (this run, default)
    A100 PCIe-fcd  : HBM2e   + PCIe   (this run, --four-condition)

Cost
----
A100-80GB:2 on Modal is ~$6.80/hr. Each run is ~10-15 min wall-clock
(model already cached in Volume after first run). 2-condition default
~= $2-3. 4-condition full matrix ~= $5-6.

Run
---
    pip install modal
    modal setup       # first-time auth (browser flow)
    modal run scripts/bench_modal_a100.py                 # 2-condition (NVLink only)
    modal run scripts/bench_modal_a100.py --four-condition  # 4-condition full matrix

Output
------
results/modal_2x_a100_v2.json — `nvlink` block always present, `pcie_forced`
block present only with --four-condition.
"""
import modal

app = modal.App("qwen36-vllm-mtp-cross-hw")

# Persistent volume so model weights (~24 GB AWQ) are downloaded once
# and reused across runs.
hf_cache = modal.Volume.from_name(
    "hf-cache-qwen36-awq", create_if_missing=True
)

# vLLM official image, version-pinned to match the 3090 bench exactly.
# `.entrypoint([])` clears the upstream image's `vllm` ENTRYPOINT — without
# this, Modal's container bootstrap (`python -u -R ... -m
# modal._container_entrypoint`) is shoved through the `vllm` CLI and fails
# with "unrecognized arguments: -u -R --check-hash-based-pycs ...".
vllm_image = (
    modal.Image.from_registry("vllm/vllm-openai:v0.19.1", add_python="3.12")
    .entrypoint([])
    .apt_install("curl")
    .pip_install("huggingface-hub>=1.3", "requests")
    .env({"HF_HOME": "/cache/hf", "HF_HUB_CACHE": "/cache/hf/hub"})
)

PROMPTS = [
    "Why does the sky look blue? Answer in two sentences. /no_think",
    "Write a Python function fib(n) returning the first n Fibonacci numbers as a list. /no_think",
    "Explain TCP vs UDP in 3 concise bullet points. /no_think",
    "Give 5 numbered steps to cook firm tofu at home. /no_think",
    "Write a short haiku about debugging a memory leak at 2am. /no_think",
]


@app.function(
    image=vllm_image,
    gpu="A100-80GB:2",
    volumes={"/cache": hf_cache},
    timeout=2400,  # 40 min headroom
)
def run_one(use_mtp: bool, force_pcie: bool = False):
    """Boot vLLM, run the 5-prompt bench, return JSON results.

    `force_pcie=True` sets `NCCL_P2P_DISABLE=1` so TP=2 allreduce is forced
    over PCIe instead of NVLink, isolating "interconnect bandwidth" from
    "GPU memory bandwidth" as a control variable on the same hardware.
    """
    import json
    import os
    import socket
    import subprocess
    import time
    import urllib.request

    from huggingface_hub import snapshot_download

    # GPU + topology sanity snapshot — surface any silent topology drift
    # (e.g. Modal allocating 2x A100 SXM but with NVLink off, or wrong SKU).
    # Topology query needs CAP_SYS_ADMIN-ish privileges that Modal containers
    # don't always have, so make it defensive — degrade to <unavailable>
    # instead of failing the whole bench.
    def _safe_run(cmd):
        try:
            return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
        except Exception as e:
            return f"<unavailable: {type(e).__name__}: {e}>"

    nvsmi = _safe_run([
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,driver_version",
        "--format=csv,noheader",
    ])
    topo = _safe_run(["nvidia-smi", "topo", "-m"])
    nvlink_status = _safe_run(["nvidia-smi", "nvlink", "--status"])
    print(f"[bench] GPUs:\n{nvsmi}")
    print(f"[bench] Topology:\n{topo}")
    print(f"[bench] NVLink:\n{nvlink_status}")

    print(f"[bench] use_mtp={use_mtp} force_pcie={force_pcie} — downloading model if needed")
    model_path = snapshot_download(
        "QuantTrio/Qwen3.6-35B-A3B-AWQ",
        cache_dir="/cache/hf/hub",
        max_workers=4,
    )
    print(f"[bench] model_path={model_path}")

    # Match published 3090 README exactly (see qwen3.6-vllm-2x3090/README.md
    # "vLLM serve flags"). Any divergence here breaks the cross-hardware
    # control claim — the only intentional variable is hardware (and
    # optionally interconnect via NCCL_P2P_DISABLE).
    cmd = [
        "vllm", "serve", model_path,
        "--served-model-name", "qwen36-awq",
        "--tensor-parallel-size", "2",
        "--enable-expert-parallel",
        "--gpu-memory-utilization", "0.90",
        "--max-model-len", "32768",
        "--max-num-seqs", "8",
        "--enable-chunked-prefill",
        "--enable-prefix-caching",
        "--enable-auto-tool-choice",
        "--tool-call-parser", "hermes",
        "--reasoning-parser", "qwen3",
        "--mm-encoder-tp-mode", "data",
        "--mm-processor-cache-type", "shm",
        "--trust-remote-code",
        "--host", "127.0.0.1", "--port", "8000",
    ]
    if use_mtp:
        cmd += [
            "--speculative-config",
            '{"method":"mtp","num_speculative_tokens":1}',
        ]
    print("[bench] launching vLLM:", " ".join(cmd[:6]), "...")

    env = os.environ.copy()
    if force_pcie:
        env["NCCL_P2P_DISABLE"] = "1"
        env["NCCL_DEBUG"] = "INFO"  # confirm in log that P2P really off
        print("[bench] FORCE_PCIE=1 set — NCCL_P2P_DISABLE=1")

    log_path = f"/tmp/vllm_serve_{'mtp' if use_mtp else 'nomtp'}_{'pcie' if force_pcie else 'p2p'}.log"
    log_fh = open(log_path, "w")
    proc = subprocess.Popen(
        cmd,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        env=env,
    )

    # Wait for /v1/models to respond. Cold load on 2x A100 SXM with first
    # weight load + AWQ-Marlin unmarshal + torch.compile + CUDA-graph
    # capture is typically 4-7 min; allow 12 min headroom.
    boot_t0 = time.perf_counter()
    ready = False
    for _ in range(360):  # 12 min @ 2 s/step
        if proc.poll() is not None:
            log_fh.flush()
            tail = subprocess.run(
                ["tail", "-c", "8000", log_path], capture_output=True, text=True,
            ).stdout
            raise RuntimeError(
                f"vLLM exited early with code {proc.returncode}.\nTail of {log_path}:\n{tail}"
            )
        try:
            with urllib.request.urlopen(
                "http://127.0.0.1:8000/v1/models", timeout=3
            ) as r:
                if r.status == 200:
                    ready = True
                    break
        except Exception:
            pass
        time.sleep(2)
    if not ready:
        proc.terminate()
        log_fh.flush()
        tail = subprocess.run(
            ["tail", "-c", "8000", log_path], capture_output=True, text=True,
        ).stdout
        raise RuntimeError(f"vLLM did not become ready within 12 min.\nTail:\n{tail}")
    boot_s = time.perf_counter() - boot_t0
    print(f"[bench] vLLM ready in {boot_s:.1f}s")

    def _post(body, timeout=120):
        req = urllib.request.Request(
            "http://127.0.0.1:8000/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        return json.loads(urllib.request.urlopen(req, timeout=timeout).read().decode())

    # Warmup
    _post({
        "model": "qwen36-awq",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 20,
        "chat_template_kwargs": {"enable_thinking": False},
    }, timeout=60)

    print("[bench] running 5 prompts ...")
    results = []
    for i, p in enumerate(PROMPTS, 1):
        body = {
            "model": "qwen36-awq",
            "messages": [{"role": "user", "content": p}],
            "max_tokens": 200,
            "temperature": 0.5,
            "seed": 42,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        t0 = time.perf_counter()
        d = _post(body)
        dt = time.perf_counter() - t0
        ct = d.get("usage", {}).get("completion_tokens", 0)
        tps = ct / dt if dt > 0 else 0
        results.append({"prompt_idx": i, "ct": ct, "elapsed_s": dt, "tok_s": tps})
        print(f"  [{i}/5] ct={ct} {dt:.2f}s {tps:.1f} tok/s")

    proc.terminate()
    try:
        proc.wait(timeout=45)
    except subprocess.TimeoutExpired:
        proc.kill()
    log_fh.close()

    ts = [r["tok_s"] for r in results]
    mean = sum(ts) / len(ts)
    # Population stdev (divisor=N) — matches the formula used in the 3090
    # v2 bench's `power_scaling_v2.json` so the comparison is apples-to-apples.
    pstdev = (sum((x - mean) ** 2 for x in ts) / len(ts)) ** 0.5
    return {
        "use_mtp": use_mtp,
        "force_pcie": force_pcie,
        "hardware": (
            "Modal 2x A100-80GB TP=2 (HBM2e, "
            + ("NCCL P2P disabled — PCIe forced" if force_pcie else "NVLink")
            + ")"
        ),
        "nvidia_smi": nvsmi,
        "topology": topo,
        "nvlink_status": nvlink_status,
        "vllm": "0.19.1",
        "model": "QuantTrio/Qwen3.6-35B-A3B-AWQ (AWQ-Marlin Q4)",
        "serve_flags": cmd[1:],  # for postmortem, full command captured
        "boot_s": round(boot_s, 1),
        "results": results,
        "mean_tok_s": round(mean, 2),
        "min_tok_s": round(min(ts), 2),
        "max_tok_s": round(max(ts), 2),
        "stdev_tok_s": round(pstdev, 2),
    }


def _summarize(label, r):
    print(
        f"  [{label}] mean={r['mean_tok_s']} min={r['min_tok_s']} "
        f"max={r['max_tok_s']} stdev={r['stdev_tok_s']} boot={r['boot_s']}s"
    )


def _delta(mtp_r, base_r):
    return {
        "mean_delta_pct": round(100 * (mtp_r["mean_tok_s"] / base_r["mean_tok_s"] - 1), 1),
        "stdev_blowup_factor": round(
            mtp_r["stdev_tok_s"] / max(base_r["stdev_tok_s"], 0.01), 1
        ),
        "best_case_speedup_pct": round(
            100 * (mtp_r["max_tok_s"] / base_r["mean_tok_s"] - 1), 1
        ),
        "worst_case_slowdown_pct": round(
            100 * (mtp_r["min_tok_s"] / base_r["mean_tok_s"] - 1), 1
        ),
    }


@app.local_entrypoint()
def main(four_condition: bool = False):
    """Run no-MTP + MTP k=1 on Modal A100. Default = 2-condition (NVLink only).

    Pass `--four-condition` to also run NCCL_P2P_DISABLE=1 PCIe-forced
    variants — that disentangles "HBM bandwidth" from "NVLink interconnect"
    as the cause of any cross-hardware MTP-result divergence vs 3090.
    """
    import json
    import pathlib

    out = {
        "v": 2,
        "harness": "5 prompts × max_tokens=200 × temperature=0.5 × seed=42",
        "vs_2x_3090_baseline": {
            "3090_no_mtp_mean_tok_s": 126.4,
            "3090_mtp_k1_mean_tok_s": 111.2,
            "3090_mtp_delta_pct": -12.0,
            "3090_stdev_blowup": 65.0,
            "3090_interconnect": "PCIe Gen4 x8 (no NVLink)",
            "note": (
                "Cross-hardware comparison disentangles two variables: "
                "(1) GPU memory bandwidth (3090 GDDR6X 936 GB/s vs A100 HBM2e 2 TB/s), "
                "and (2) TP allreduce interconnect (3090 PCIe ~16 GB/s vs A100 NVLink ~600 GB/s). "
                "The 4-condition matrix isolates each."
            ),
        },
    }

    print("=== A100 NVLink: No-MTP baseline ===")
    no_mtp_nv = run_one.remote(use_mtp=False, force_pcie=False)
    _summarize("nvlink/no-mtp", no_mtp_nv)
    print("\n=== A100 NVLink: MTP k=1 ===")
    mtp_nv = run_one.remote(use_mtp=True, force_pcie=False)
    _summarize("nvlink/mtp", mtp_nv)

    out["nvlink"] = {
        "no_mtp_baseline": no_mtp_nv,
        "mtp_k1": mtp_nv,
        "comparison": _delta(mtp_nv, no_mtp_nv),
    }

    if four_condition:
        print("\n=== A100 PCIe-forced (NCCL_P2P_DISABLE=1): No-MTP baseline ===")
        no_mtp_pcie = run_one.remote(use_mtp=False, force_pcie=True)
        _summarize("pcie/no-mtp", no_mtp_pcie)
        print("\n=== A100 PCIe-forced: MTP k=1 ===")
        mtp_pcie = run_one.remote(use_mtp=True, force_pcie=True)
        _summarize("pcie/mtp", mtp_pcie)

        out["pcie_forced"] = {
            "no_mtp_baseline": no_mtp_pcie,
            "mtp_k1": mtp_pcie,
            "comparison": _delta(mtp_pcie, no_mtp_pcie),
        }

    out_path = pathlib.Path("results/modal_2x_a100_v2.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    print("\n=== Summary ===")
    nv = out["nvlink"]["comparison"]
    print(f"A100 NVLink   MTP delta: {nv['mean_delta_pct']}% (3090 was -12.0%, stdev {nv['stdev_blowup_factor']}x vs 3090's 65x)")
    if four_condition:
        pc = out["pcie_forced"]["comparison"]
        print(f"A100 PCIe-fcd MTP delta: {pc['mean_delta_pct']}% (stdev {pc['stdev_blowup_factor']}x)")
    print(f"\nFull results saved to: {out_path.resolve()}")
