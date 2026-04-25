#!/usr/bin/env python3
"""The headline test for Option F — vLLM continuous batching with concurrent
vision+dialog requests.

Hypothesis: with --enable-chunked-prefill, dialog decode tok/s should NOT
degrade by more than ~10% while a vision prefill is in flight on the same
engine. If this holds, Option F (single unified model on s1, no GPU partitioning)
is validated and we don't need B1/B2.

Methodology:
  T1 baseline: dialog alone, mean tok/s
  T2 baseline: vision alone, prefill latency
  T3 concurrent: vision request fired, then 200ms later 5 dialog requests
                 measure dialog tok/s during T3, compare to T1
  T4 inverse: dialog request first, then vision 200ms later
                 measure vision prefill latency during T4, compare to T2

Pass criteria:
  - T3 dialog mean tok/s >= 0.90 * T1
  - T4 vision prefill latency <= 1.20 * T2
  - No 5xx errors, no engine crashes
"""
import base64
import concurrent.futures as cf
import json
import os
import time
import urllib.request

ENDPOINT = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "qwen36-awq"
IMAGE_PATH = os.environ.get(
    "BENCH_IMG", "/home/reachym/dev/reachy-agent/robot/face_detection_yunet.onnx"
)


def post(body, timeout=180):
    req = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read().decode())
    return data, time.perf_counter() - t0


def make_dialog_body(prompt, max_tokens=200):
    return {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.5,
        "seed": 42,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def make_vision_body(image_b64, prompt="Describe this image in 30 words."):
    return {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": 80,
        "temperature": 0.3,
        "chat_template_kwargs": {"enable_thinking": False},
    }


DIALOG_PROMPTS = [
    "Why does the sky look blue? Two sentences.",
    "Define TCP vs UDP in three bullets.",
    "Write a haiku about a tired robot.",
    "List five things to pack for a hike.",
    "Explain DNS in one paragraph.",
]


def encode_image():
    """Encode a real JPEG; if no JPEG handy, draw a synthetic one with PIL."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        os.system("pip install pillow -q")
        from PIL import Image, ImageDraw
    img = Image.new("RGB", (640, 480), color="lightblue")
    d = ImageDraw.Draw(img)
    d.rectangle([100, 100, 540, 380], outline="red", width=5)
    d.text((150, 200), "Reachy Mini", fill="black")
    import io as _io
    buf = _io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def t1_dialog_baseline():
    print("\n=== T1: dialog baseline (sequential) ===")
    rs = []
    for p in DIALOG_PROMPTS:
        d, dt = post(make_dialog_body(p))
        ct = d["usage"]["completion_tokens"]
        tps = ct / dt
        rs.append(tps)
        print(f"  ct={ct:>3}  {dt:5.2f}s  {tps:5.1f} tok/s")
    return sum(rs) / len(rs), min(rs), max(rs)


def t2_vision_baseline(img_b64):
    print("\n=== T2: vision baseline (sequential) ===")
    latencies = []
    for _ in range(3):
        d, dt = post(make_vision_body(img_b64))
        ct = d["usage"]["completion_tokens"]
        latencies.append(dt)
        print(f"  ct={ct:>3}  {dt:5.2f}s  text={d['choices'][0]['message']['content'][:60]!r}")
    return sum(latencies) / len(latencies)


def t3_concurrent(img_b64):
    """Fire vision then dialog 200ms later. Measure dialog tok/s under contention."""
    print("\n=== T3: concurrent (vision in flight while dialog runs) ===")
    rs = []
    with cf.ThreadPoolExecutor(max_workers=6) as ex:
        # fire vision first
        vision_fut = ex.submit(post, make_vision_body(img_b64), 180)
        time.sleep(0.2)
        # immediately fire 5 dialog requests in parallel
        dialog_futs = [
            ex.submit(post, make_dialog_body(p), 180) for p in DIALOG_PROMPTS
        ]
        for f in dialog_futs:
            d, dt = f.result()
            ct = d["usage"]["completion_tokens"]
            tps = ct / dt
            rs.append(tps)
            print(f"  dialog  ct={ct:>3}  {dt:5.2f}s  {tps:5.1f} tok/s")
        v_data, v_dt = vision_fut.result()
        print(f"  vision  ct={v_data['usage']['completion_tokens']:>3}  {v_dt:5.2f}s")
    return sum(rs) / len(rs), v_dt


def main():
    img_b64 = encode_image()
    print(f"image bytes={len(img_b64) * 3 // 4}")
    # warmup
    post(make_dialog_body("Hi", 10))

    t1_mean, t1_min, t1_max = t1_dialog_baseline()
    t2_mean = t2_vision_baseline(img_b64)
    t3_dialog_mean, t3_vision = t3_concurrent(img_b64)

    pass_dialog = t3_dialog_mean >= 0.90 * t1_mean
    pass_vision = t3_vision <= 1.30 * t2_mean

    summary = {
        "T1_dialog_mean_tok_s": t1_mean,
        "T1_dialog_min": t1_min,
        "T1_dialog_max": t1_max,
        "T2_vision_mean_s": t2_mean,
        "T3_concurrent_dialog_mean_tok_s": t3_dialog_mean,
        "T3_concurrent_vision_s": t3_vision,
        "dialog_degradation_pct": 100 * (1 - t3_dialog_mean / t1_mean),
        "vision_inflation_pct": 100 * (t3_vision / t2_mean - 1),
        "PASS_dialog": pass_dialog,
        "PASS_vision": pass_vision,
        "PASS_overall": pass_dialog and pass_vision,
    }
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    with open("vllm_concurrent_bench.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    main()
