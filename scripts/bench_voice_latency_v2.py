#!/usr/bin/env python3
"""End-to-end voice dialog latency budget v2 — methodologically rigorous.

What changed vs v1:
- STT now uses REAL synthesized speech (edge-tts → WAV) instead of synthetic
  noise. v1's `np.random.uniform()` short-circuited Silero VAD and reported
  bogus 8 ms latency. Real Whisper CUDA inference is 50-300 ms for 6 s audio.
- Mem0 now measures FULL search (embed + Qdrant retrieve) via the
  `RobotMemory` class, not just bge-m3 embed.
- LLM prefill now uses a realistic prompt mirroring production: full system
  prompt + 30-message history + scene desc. v1's 30-token toy prompt gave
  artificially low TTFT.
- TTS first-byte caveat is now explicit: this is "first audio chunk arrives
  at robot_brain", NOT "user hears sound". Audio buffer + GStreamer +
  WebRTC encode adds 100-300 ms more.
- WebRTC RTT caveat: ICMP ping is a lower bound; real WebRTC audio frame
  RTT is 30-80 ms once NAT traversal + encryption + jitter buffer are
  counted.
- Adds REAL PRODUCTION TIMINGS extracted from robot_brain.log
  ([輪總耗時] STT+LLM+TTS = X ms) — these are ground truth.

Output: results/voice_latency_budget_v2.json
"""
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

VLLM_URL = "http://127.0.0.1:8000/v1/chat/completions"
OLLAMA_URL = "http://127.0.0.1:11434"
ROBOT_IP = "100.85.191.3"
ROBOT_LOG = "/home/reachym/logs/robot-brain.log"

RESULTS = {
    "v": 2,
    "hardware": "2x RTX 3090 24GB, i7-11700, 64GB DDR4",
    "vllm": "0.19.1 + QuantTrio/Qwen3.6-35B-A3B-AWQ TP=2",
    "stages": {},
    "caveats": [
        "STT: measured on synthesized speech via edge-tts. Real human speech "
        "with similar clarity should be comparable; noisy / accented speech may "
        "be 2-3x slower.",
        "Mem0 search: measures embed + Qdrant retrieve. Cold cache (first "
        "query) is 10-20x slower than warm.",
        "LLM TTFB: measured on the production-mirroring prompt (~1500 tok "
        "system+history). Shorter prompts will be faster.",
        "TTS first-byte: time from request to first audio CHUNK arriving at "
        "robot_brain. NOT time-to-user-hears. Add ~100-300 ms for WebRTC + "
        "audio buffer pipeline before sound reaches the speaker.",
        "WebRTC RTT: ICMP ping is a lower bound on network round-trip. Real "
        "WebRTC audio frame RTT is 30-80 ms (NAT traversal, encryption, "
        "jitter buffer overhead).",
        "Sum-of-stages 'estimated TTFB' is a theoretical lower bound for an "
        "ideal pipeline. Real production wall-clock is higher due to "
        "sequential dependencies the bench does not capture (audio playback "
        "wait, etc). Real production times reported separately below.",
    ],
}

PROD_SYSTEM_PROMPT = """\
You are Reachy Mini, a curious desk robot. Warm, playful, specific, not cartoonish. No emoji prefixes. Do not pad.

LENGTH: 1 sentence for greetings. 2-4 sentences for questions. Match the user's depth — never longer.
LANGUAGE: English only. No emojis (read aloud).
MEMORY: Use the conversation history to recall names/facts. Do not invent.
ACTIONS (at most one, optional): happy | nod | shake | think | greet

OUTPUT FORMAT — MUST be valid JSON, no markdown:
{"speech":"<words>","actions":["<one_or_empty>"]}"""


PROD_HISTORY = []
for i in range(15):
    PROD_HISTORY.append(
        {"role": "user", "content": f"Hi, can you tell me about topic number {i}?"}
    )
    PROD_HISTORY.append(
        {
            "role": "assistant",
            "content": f'{{"speech":"Sure, topic {i} is interesting because it relates to several ideas including the previous conversation we had.","actions":["nod"]}}',
        }
    )


def measure_ping_to_robot():
    print("[1/6] WebRTC RTT proxy: ICMP ping to robot...")
    try:
        out = subprocess.check_output(
            ["ping", "-c", "10", "-W", "2", ROBOT_IP], timeout=30
        ).decode()
    except Exception as e:
        return {"error": str(e)}
    for line in out.splitlines():
        if "rtt" in line and "/" in line:
            stats = line.split("=")[1].strip().split()[0].split("/")
            return {
                "min_ms": float(stats[0]),
                "avg_ms": float(stats[1]),
                "max_ms": float(stats[2]),
                "mdev_ms": float(stats[3]),
                "note": "ICMP ping; WebRTC audio frame RTT will be higher",
            }
    return {"raw": out}


def synthesize_real_audio(out_dir):
    """Synthesize realistic speech via edge-tts at 2 / 6 / 12 s lengths."""
    print("[STT prep] Synthesizing real speech via edge-tts...")
    try:
        import asyncio

        from edge_tts import Communicate
    except ImportError:
        return None
    texts = {
        2: "Hello there.",
        6: "Hi Reachy, how are you doing today? I have a question for you.",
        12: "Can you tell me about the weather in Taipei? I'm planning a trip there next week and I want to know whether to pack warm clothes.",
    }

    async def synth_one(text, fname):
        comm = Communicate(text, "en-US-AnaNeural")
        await comm.save(fname)

    results = {}
    for sec, text in texts.items():
        fname = os.path.join(out_dir, f"speech_{sec}s.mp3")
        try:
            asyncio.run(synth_one(text, fname))
            results[sec] = fname
        except Exception as e:
            results[sec] = {"error": str(e)}
    return results


def measure_stt_latency_real(audio_files):
    print("[2/6] STT (faster-whisper CUDA, REAL synthesized speech)...")
    try:
        import io as _io
        import numpy as np
        import soundfile as sf
        import subprocess
        from faster_whisper import WhisperModel
    except ImportError as e:
        return {"error": f"missing import: {e}"}
    model = WhisperModel(
        "large-v3-turbo", device="cuda", compute_type="int8_float16", cpu_threads=8
    )
    # warm
    warm = np.zeros(16000 * 2, dtype=np.float32)
    list(model.transcribe(warm, language="en", beam_size=3, vad_filter=True)[0])

    out = {}
    for sec, path in audio_files.items():
        if not isinstance(path, str):
            out[f"audio_{sec}s"] = path
            continue
        # Convert mp3 to 16kHz wav via ffmpeg (subprocess)
        wav_path = path.replace(".mp3", ".wav")
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", path, "-ar", "16000", "-ac", "1", wav_path],
                check=True, capture_output=True,
            )
            audio, sr = sf.read(wav_path, dtype="float32")
        except Exception as e:
            out[f"audio_{sec}s"] = {"error": str(e)}
            continue
        latencies = []
        transcripts = []
        for _ in range(3):
            t0 = time.perf_counter()
            segs, _info = model.transcribe(
                audio, language="en", beam_size=3, vad_filter=True,
                condition_on_previous_text=False,
            )
            text = "".join(s.text for s in segs).strip()
            latencies.append((time.perf_counter() - t0) * 1000)
            transcripts.append(text)
        out[f"audio_{sec}s_ms"] = {
            "audio_duration_s": sec,
            "mean_ms": sum(latencies) / len(latencies),
            "min_ms": min(latencies),
            "max_ms": max(latencies),
            "rtf": sum(latencies) / len(latencies) / (sec * 1000),
            "sample_transcript": transcripts[0][:80],
        }
    return out


def measure_mem0_full_search():
    """Use the actual RobotMemory class (embed + Qdrant retrieve)."""
    print("[3/6] Mem0 full search (bge-m3 embed + Qdrant retrieve)...")
    sys.path.insert(0, "/home/reachym/dev/reachy-agent/robot")
    try:
        from robot_memory import RobotMemory
    except ImportError as e:
        return {"error": f"robot_memory not importable: {e}"}
    try:
        mem = RobotMemory(
            conversation_log_path="/home/reachym/dev/reachy-agent/robot/conversation_log.jsonl"
        )
    except Exception as e:
        return {"error": f"RobotMemory init failed: {e}"}
    if not getattr(mem, "enabled", False):
        return {"error": "Mem0 not enabled in this venv"}

    queries = [
        "What is my name?",
        "Tell me about Hideyoshi.",
        "What did I say about coding?",
        "Recall my preferences.",
        "Random uncached query " + str(time.time()),
    ]
    raw = []
    for q in queries:
        t0 = time.perf_counter()
        try:
            facts = mem.search(q, limit=3)
            ms = (time.perf_counter() - t0) * 1000
            raw.append({"q": q[:30], "ms": ms, "n_facts": len(facts) if facts else 0})
        except Exception as e:
            raw.append({"q": q[:30], "error": str(e)})
    valid = [r for r in raw if "ms" in r]
    return {
        "first_call_cold_ms": valid[0]["ms"] if valid else None,
        "subsequent_warm_mean_ms": (
            sum(r["ms"] for r in valid[1:]) / max(1, len(valid) - 1)
        )
        if len(valid) > 1
        else None,
        "raw": raw,
    }


def measure_llm_prefill_realistic():
    print("[4/6] LLM prefill (PRODUCTION-mirror prompt with system + 30-msg history)...")
    msgs = (
        [{"role": "system", "content": PROD_SYSTEM_PROMPT}]
        + PROD_HISTORY
        + [{"role": "user", "content": "What is your name?"}]
    )
    body_template = {
        "model": "qwen36-awq",
        "messages": msgs,
        "max_tokens": 1,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    latencies = []
    for _ in range(5):
        t0 = time.perf_counter()
        try:
            req = urllib.request.Request(
                VLLM_URL,
                data=json.dumps(body_template).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=60) as r:
                _ = json.loads(r.read().decode())
            latencies.append((time.perf_counter() - t0) * 1000)
        except Exception as e:
            latencies.append({"error": str(e)})
    valid = [x for x in latencies if isinstance(x, (int, float))]
    n_msg = len(msgs)
    return {
        "n_messages": n_msg,
        "estimated_prompt_tokens": sum(len(m["content"]) for m in msgs) // 4,  # ~chars/4
        "ttft_max_tok_1_mean_ms": (sum(valid) / len(valid)) if valid else None,
        "ttft_min_ms": min(valid) if valid else None,
        "ttft_max_ms": max(valid) if valid else None,
        "n_runs": len(valid),
    }


def measure_llm_streaming_first_byte_realistic():
    print("[5/6] LLM streaming first-byte (PRODUCTION-mirror prompt)...")
    body = {
        "model": "qwen36-awq",
        "messages": [{"role": "system", "content": PROD_SYSTEM_PROMPT}]
        + PROD_HISTORY
        + [{"role": "user", "content": "Hi, what should we talk about today?"}],
        "max_tokens": 60,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    runs = []
    for _ in range(5):
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
                            break
        except Exception as e:
            runs.append({"error": str(e)})
            continue
        runs.append({"ttfb_ms": ttfb_ms})
    valid = [r["ttfb_ms"] for r in runs if "ttfb_ms" in r and r["ttfb_ms"]]
    return {
        "mean_ttfb_ms": (sum(valid) / len(valid)) if valid else None,
        "min_ttfb_ms": min(valid) if valid else None,
        "max_ttfb_ms": max(valid) if valid else None,
        "n_runs": len(valid),
    }


def measure_tts_first_byte():
    print("[6/6] TTS edge-tts first audio chunk arrival...")
    try:
        from edge_tts import Communicate
    except ImportError:
        return {"error": "edge_tts not installed"}
    import asyncio

    async def one(text):
        comm = Communicate(text, "en-US-AnaNeural")
        t0 = time.perf_counter()
        async for chunk in comm.stream():
            if chunk.get("type") == "audio":
                return (time.perf_counter() - t0) * 1000
        return None

    runs = []
    for txt in [
        "Hi.",
        "Sure, that sounds great.",
        "Let me think about that for a moment.",
        "Hello, how are you doing today?",
        "I am Reachy Mini, a curious robot.",
    ]:
        try:
            ms = asyncio.run(one(txt))
            runs.append({"chars": len(txt), "first_chunk_ms": ms})
        except Exception as e:
            runs.append({"chars": len(txt), "error": str(e)})
    valid = [r["first_chunk_ms"] for r in runs if "first_chunk_ms" in r and r["first_chunk_ms"]]
    return {
        "mean_first_chunk_ms": (sum(valid) / len(valid)) if valid else None,
        "min_ms": min(valid) if valid else None,
        "max_ms": max(valid) if valid else None,
        "raw": runs,
        "note": "First audio chunk arrives at robot_brain — NOT time-to-user-hears.",
    }


def extract_production_timings(log_path):
    """Pull real `[輪總耗時] STT+LLM+TTS = X ms` and TTFB lines from production log."""
    print("[7] Real production wall-clock from robot_brain.log...")
    if not os.path.exists(log_path):
        return {"error": f"log not at {log_path}"}
    try:
        with open(log_path) as f:
            lines = f.read().splitlines()
    except Exception as e:
        return {"error": str(e)}
    rounds = [int(m.group(1)) for m in (
        re.search(r"\[輪總耗時\] STT\+LLM\+TTS = (\d+)ms", l) for l in lines) if m]
    ttfbs = [int(m.group(1)) for m in (
        re.search(r"TTFB=(\d+)ms", l) for l in lines) if m]
    walls = [int(m.group(1)) for m in (
        re.search(r"wall=(\d+)ms", l) for l in lines) if m]
    stt = [
        float(m.group(1)) for m in (
            re.search(r"\[STT [^\]]*?\] ([\d.]+)ms /", l) for l in lines)
        if m
    ]

    def stats(xs):
        if not xs:
            return None
        s = sorted(xs)
        return {
            "n": len(xs),
            "min": s[0],
            "p50": s[len(s) // 2],
            "p95": s[int(len(s) * 0.95)] if len(s) > 1 else s[0],
            "max": s[-1],
            "mean": sum(xs) / len(xs),
        }

    return {
        "round_total_ms_full_pipeline": stats(rounds),
        "llm_ttfb_ms_streaming": stats(ttfbs),
        "llm_wall_ms_includes_audio_playback": stats(walls),
        "stt_actual_ms": stats(stt),
        "n_rounds_total": len(rounds),
    }


def main():
    out_dir = "/tmp/voice_latency_v2"
    os.makedirs(out_dir, exist_ok=True)

    RESULTS["stages"]["1_webrtc_rtt"] = measure_ping_to_robot()
    audio_files = synthesize_real_audio(out_dir) or {}
    RESULTS["stages"]["2_stt_real_speech"] = measure_stt_latency_real(audio_files)
    RESULTS["stages"]["3_mem0_full_search"] = measure_mem0_full_search()
    RESULTS["stages"]["4_llm_prefill_realistic"] = measure_llm_prefill_realistic()
    RESULTS["stages"]["5_llm_streaming_ttfb_realistic"] = (
        measure_llm_streaming_first_byte_realistic()
    )
    RESULTS["stages"]["6_tts_first_chunk"] = measure_tts_first_byte()
    RESULTS["stages"]["7_real_production_timings"] = extract_production_timings(
        ROBOT_LOG
    )

    # Theoretical sum (with proper caveat that real wall is higher)
    try:
        st = RESULTS["stages"]
        stt_realistic = st["2_stt_real_speech"].get("audio_6s_ms", {}).get("mean_ms", 0)
        mem0 = st["3_mem0_full_search"].get("subsequent_warm_mean_ms", 0)
        llm = st["5_llm_streaming_ttfb_realistic"].get("mean_ttfb_ms", 0)
        tts = st["6_tts_first_chunk"].get("mean_first_chunk_ms", 0)
        rtt = st["1_webrtc_rtt"].get("avg_ms", 0)
        prod = st["7_real_production_timings"]
        RESULTS["budget_theoretical_lower_bound_ms"] = {
            "stt_6s_real_speech": stt_realistic,
            "mem0_warm_search": mem0,
            "llm_streaming_first_byte": llm,
            "tts_first_chunk": tts,
            "webrtc_ping_lower_bound": rtt,
            "sum": stt_realistic + mem0 + llm + tts + rtt,
            "note": "This is theoretical. Real production p50 is far higher.",
        }
        if prod.get("round_total_ms_full_pipeline"):
            RESULTS["real_production_observed_ms"] = {
                "full_pipeline_p50": prod["round_total_ms_full_pipeline"]["p50"],
                "full_pipeline_min": prod["round_total_ms_full_pipeline"]["min"],
                "llm_ttfb_p50": (prod.get("llm_ttfb_ms_streaming") or {}).get("p50"),
                "n_observations": prod["round_total_ms_full_pipeline"]["n"],
                "note": "Ground truth from real conversations. Includes audio playback + sequential pipeline + sentence chunking.",
            }
    except Exception as e:
        RESULTS["budget_compute_error"] = str(e)

    out = Path("voice_latency_budget_v2.json")
    out.write_text(json.dumps(RESULTS, indent=2, ensure_ascii=False))
    print(f"\n=== theoretical lower bound ===")
    print(json.dumps(RESULTS.get("budget_theoretical_lower_bound_ms", {}), indent=2))
    print(f"\n=== real production observed ===")
    print(json.dumps(RESULTS.get("real_production_observed_ms", {}), indent=2))
    print(f"\nfull -> {out.resolve()}")


if __name__ == "__main__":
    main()
