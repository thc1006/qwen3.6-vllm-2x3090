#!/usr/bin/env python3
"""End-to-end voice dialog latency budget — embodied robot stack on 2x RTX 3090.

Measures each stage of the voice pipeline:
    Mic -> VAD -> STT -> Mem0 search -> LLM prefill -> LLM first-decode
        -> TTS first-byte -> WebRTC RTT -> total user-perceived TTFB

Comparable to:
- Smallest.ai 2026 voice agent budget (cloud): ~800 ms
- Trillet 2026 voice latency benchmarks (cloud): 400-1000 ms

Novel angle: this is an **embodied robot** stack with WebRTC + actual hardware,
which adds network + WebRTC latency that pure cloud benchmarks omit.

Components measured:
- STT: faster-whisper large-v3-turbo (CUDA int8_float16, locally on s1)
- LLM prefill + decode: vLLM 0.19.1 with QuantTrio/Qwen3.6-35B-A3B-AWQ TP=2
- Mem0: bge-m3 via Ollama + Qdrant
- TTS: edge-tts cloud
- WebRTC: Tailscale RTT to robot (100.85.191.3)

Output: results/voice_latency_budget.json
"""
import base64
import io
import json
import socket
import subprocess
import time
import urllib.request
from pathlib import Path

VLLM_URL = "http://127.0.0.1:8000/v1/chat/completions"
OLLAMA_URL = "http://127.0.0.1:11434"
ROBOT_IP = "100.85.191.3"

RESULTS = {"hardware": "2x RTX 3090 24GB, i7-11700, 64GB DDR4",
           "vllm": "0.19.1 + QuantTrio/Qwen3.6-35B-A3B-AWQ TP=2",
           "stages": {}}


def measure_ping_to_robot():
    """WebRTC RTT proxy: ICMP ping to robot's Tailscale IP."""
    print("[1/6] WebRTC RTT (ping to robot)...")
    try:
        out = subprocess.check_output(
            ["ping", "-c", "10", "-W", "2", ROBOT_IP], timeout=30
        ).decode()
    except Exception as e:
        return {"error": str(e)}
    # Parse "rtt min/avg/max/mdev = 7.7/28.2/48.7/20.4 ms"
    for line in out.splitlines():
        if "rtt" in line and "/" in line:
            stats = line.split("=")[1].strip().split()[0].split("/")
            return {
                "min_ms": float(stats[0]),
                "avg_ms": float(stats[1]),
                "max_ms": float(stats[2]),
                "mdev_ms": float(stats[3]),
            }
    return {"raw": out}


def measure_stt_latency():
    """Run Whisper STT inline via the same library robot_brain uses."""
    print("[2/6] STT (faster-whisper local CUDA)...")
    try:
        import numpy as np
        from faster_whisper import WhisperModel
    except ImportError:
        return {"error": "faster_whisper not in this venv; run from robot venv"}
    model = WhisperModel(
        "large-v3-turbo", device="cuda", compute_type="int8_float16", cpu_threads=8
    )
    # Warmup
    warm = np.zeros(16000 * 2, dtype=np.float32)
    list(model.transcribe(warm, language="en", beam_size=3, vad_filter=True)[0])

    # Three audio lengths: 2 s, 6 s, 12 s
    results = {}
    for sec in (2, 6, 12):
        audio = np.random.uniform(-0.05, 0.05, 16000 * sec).astype(np.float32)
        # spike to trigger VAD
        audio[::1600] = 0.5
        latencies = []
        for _ in range(3):
            t0 = time.perf_counter()
            list(
                model.transcribe(
                    audio, language="en", beam_size=3, vad_filter=True
                )[0]
            )
            latencies.append((time.perf_counter() - t0) * 1000)
        results[f"audio_{sec}s_ms"] = {
            "mean": sum(latencies) / len(latencies),
            "min": min(latencies),
            "max": max(latencies),
            "rtf": sum(latencies) / len(latencies) / (sec * 1000),
        }
    return results


def measure_mem0_search_latency():
    """Mem0 = bge-m3 embed + Qdrant retrieve."""
    print("[3/6] Mem0 (bge-m3 via Ollama + Qdrant)...")
    queries = [
        "What is my name?",
        "Tell me about Hideyoshi.",
        "What did I say earlier about coding?",
        "Recall my preferences.",
        "Random query that hasn't been embedded before.",
    ]
    results = []
    for q in queries:
        t0 = time.perf_counter()
        try:
            req = urllib.request.Request(
                f"{OLLAMA_URL}/api/embeddings",
                data=json.dumps({"model": "bge-m3", "prompt": q}).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=10) as r:
                _ = json.loads(r.read().decode())
            embed_ms = (time.perf_counter() - t0) * 1000
            results.append({"query": q[:30], "embed_only_ms": embed_ms})
        except Exception as e:
            results.append({"query": q[:30], "error": str(e)})
    return {
        "embed_only_first_call_ms": results[0].get("embed_only_ms"),
        "embed_only_subsequent_mean_ms": (
            sum(r["embed_only_ms"] for r in results[1:] if "embed_only_ms" in r)
            / max(1, len(results) - 1)
        ),
        "raw": results,
    }


def measure_llm_prefill():
    """vLLM TTFT for various prompt lengths."""
    print("[4/6] LLM prefill (vLLM AWQ-Marlin TP=2)...")
    sys_prompt = "You are a helpful assistant. Reply briefly."
    history = ""
    sizes = [(0, "no_history"), (10, "5_turns"), (30, "15_turns"), (60, "30_turns")]
    results = {}
    for n_msg, label in sizes:
        if n_msg:
            history = " ".join([f"M{i}." for i in range(n_msg)])
        body = {
            "model": "qwen36-awq",
            "messages": [
                {"role": "system", "content": sys_prompt + " " + history},
                {"role": "user", "content": "What is your name?"},
            ],
            "max_tokens": 1,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        latencies = []
        for _ in range(3):
            t0 = time.perf_counter()
            try:
                req = urllib.request.Request(
                    VLLM_URL,
                    data=json.dumps(body).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=60) as r:
                    _ = json.loads(r.read().decode())
                latencies.append((time.perf_counter() - t0) * 1000)
            except Exception as e:
                latencies.append({"error": str(e)})
                break
        results[label] = {
            "n_history_msgs": n_msg,
            "ttfb_max_tok_1_ms": (
                sum(latencies) / len(latencies)
                if all(isinstance(x, (int, float)) for x in latencies)
                else latencies
            ),
        }
    return results


def measure_llm_streaming_first_byte():
    """vLLM SSE first-byte latency = real TTFB for streaming dialog."""
    print("[5/6] LLM streaming first-byte (vLLM)...")
    body = {
        "model": "qwen36-awq",
        "messages": [
            {"role": "system", "content": "Reply briefly."},
            {"role": "user", "content": "Count to three."},
        ],
        "max_tokens": 60,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    runs = []
    for _ in range(3):
        t0 = time.perf_counter()
        ttfb_ms = None
        try:
            req = urllib.request.Request(
                VLLM_URL,
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=30) as r:
                for line in r:
                    s = line.strip()
                    if s.startswith(b"data: ") and s != b"data: [DONE]":
                        obj = json.loads(s[6:].decode())
                        delta = obj.get("choices", [{}])[0].get("delta", {}).get(
                            "content", ""
                        )
                        if delta and ttfb_ms is None:
                            ttfb_ms = (time.perf_counter() - t0) * 1000
                        if ttfb_ms is not None:
                            break
        except Exception as e:
            runs.append({"error": str(e)})
            continue
        runs.append({"ttfb_ms": ttfb_ms})
    valid = [r["ttfb_ms"] for r in runs if "ttfb_ms" in r and r["ttfb_ms"]]
    return {
        "mean_ttfb_ms": (sum(valid) / len(valid)) if valid else None,
        "runs": runs,
    }


def measure_tts_first_byte():
    """edge-tts call: time from request to first audio chunk."""
    print("[6/6] TTS edge-tts first-byte...")
    try:
        from edge_tts import Communicate
    except ImportError:
        return {"error": "edge_tts not installed; run from robot venv"}

    import asyncio

    async def one_call(text):
        comm = Communicate(text, "en-US-AnaNeural")
        t0 = time.perf_counter()
        async for chunk in comm.stream():
            if chunk.get("type") == "audio":
                return (time.perf_counter() - t0) * 1000
        return None

    runs = []
    for txt in ["Hello.", "How are you doing today?", "I am fine, thanks for asking."]:
        try:
            ms = asyncio.run(one_call(txt))
            runs.append({"chars": len(txt), "first_byte_ms": ms})
        except Exception as e:
            runs.append({"chars": len(txt), "error": str(e)})
    valid = [r["first_byte_ms"] for r in runs if "first_byte_ms" in r and r["first_byte_ms"]]
    return {
        "mean_first_byte_ms": (sum(valid) / len(valid)) if valid else None,
        "runs": runs,
    }


def main():
    RESULTS["stages"]["1_webrtc_rtt"] = measure_ping_to_robot()
    RESULTS["stages"]["2_stt"] = measure_stt_latency()
    RESULTS["stages"]["3_mem0_search"] = measure_mem0_search_latency()
    RESULTS["stages"]["4_llm_prefill"] = measure_llm_prefill()
    RESULTS["stages"]["5_llm_streaming_first_byte"] = (
        measure_llm_streaming_first_byte()
    )
    RESULTS["stages"]["6_tts_first_byte"] = measure_tts_first_byte()

    # Compute total budget estimate
    try:
        stt6 = RESULTS["stages"]["2_stt"].get("audio_6s_ms", {}).get("mean", 0)
        mem0 = RESULTS["stages"]["3_mem0_search"].get(
            "embed_only_subsequent_mean_ms", 0
        )
        llm_ttfb = RESULTS["stages"]["5_llm_streaming_first_byte"].get(
            "mean_ttfb_ms", 0
        )
        tts = RESULTS["stages"]["6_tts_first_byte"].get("mean_first_byte_ms", 0)
        rtt = RESULTS["stages"]["1_webrtc_rtt"].get("avg_ms", 0)
        RESULTS["budget_summary_ms"] = {
            "stt_6s_audio": stt6,
            "mem0_search_cached": mem0,
            "llm_first_byte": llm_ttfb,
            "tts_first_byte": tts,
            "webrtc_rtt": rtt,
            "estimated_total_ttfb": stt6 + mem0 + llm_ttfb + tts + rtt,
        }
    except Exception as e:
        RESULTS["budget_summary_error"] = str(e)

    out = Path("voice_latency_budget.json")
    out.write_text(json.dumps(RESULTS, indent=2))
    print(f"\n=== budget summary ===")
    print(json.dumps(RESULTS.get("budget_summary_ms", {}), indent=2))
    print(f"\nfull results -> {out.resolve()}")


if __name__ == "__main__":
    main()
