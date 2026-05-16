"""vllm-2x3090 v5 bench — Dense Qwen3.6-27B vs MoE Qwen3.6-35B-A3B on voice workload.

Run with $V5_MODEL env var = "qwen36-awq" (production MoE+MTP) or "qwen36-27b" (test dense).
Reads vLLM at http://localhost:8000/v1/chat/completions.

Output: results_${model}.json with per-call telemetry.

Metrics captured per call (× 10 prompts × N=3 trials = 30 samples/model):
  ttft_ms       streaming TTFT
  e2e_ms        total wall time
  prompt_tok    input tokens
  out_tok       output tokens
  toks_per_sec  out_tok / (e2e/1000)
  text          full response
  tool_calls    list[{name, arguments}]

Post-analysis comparisons (separate script):
  - 繁/簡 ratio per response via OpenCC t2s identity
  - tool-call accuracy vs expected per prompt
  - TTFT / e2e / tok/s mean ± std per cell

PROMPTS = 10 — 4 chat-only + 6 tool-expected. Same per model so paired test is clean.
TOOLS = 4 production-shaped tools (move_head, play_emotion, stop_motion, get_current_time).
"""
import json
import os
import re
import sys
import time
import urllib.request

URL = "http://localhost:8000/v1/chat/completions"
MODEL = os.environ.get("V5_MODEL", "qwen36-awq")
TRIALS = int(os.environ.get("V5_TRIALS", "3"))
OUT = os.environ.get("V5_OUT", f"results_{MODEL}.json")

SYSTEM = """You are Reachy Mini, a small desktop robot. Reply briefly in the user's language.
You have 4 tools — use them when the user asks for an action; otherwise just chat.
When using a tool, output ONLY the tool call, no explanatory text."""

PROMPTS = [
    # (id, text, expected_tool_or_None)
    ("c1", "你好，我是瑞奇。",             None),
    ("c2", "你今天好嗎？",                 None),
    ("c3", "跟我講個短笑話。",             None),
    ("c4", "你是誰？簡短自我介紹。",       None),
    ("t1", "向左轉。",                     "move_head"),
    ("t2", "頭抬起來看上面。",             "move_head"),
    ("t3", "跳一下舞。",                   "play_emotion"),
    ("t4", "做個開心的動作。",             "play_emotion"),
    ("t5", "停下來。",                     "stop_motion"),
    ("t6", "現在幾點？",                   "get_current_time"),
]

TOOLS = [
    {"type": "function", "function": {
        "name": "move_head",
        "description": "Move the robot head. Angles in degrees, ±25° safe range. duration ≥ 1.8s.",
        "parameters": {
            "type": "object",
            "properties": {
                "pitch": {"type": "number", "description": "Pitch deg, +down -up"},
                "yaw": {"type": "number", "description": "Yaw deg, +right -left"},
                "roll": {"type": "number", "description": "Roll deg"},
                "duration": {"type": "number", "description": "Move duration seconds"},
            },
        },
    }},
    {"type": "function", "function": {
        "name": "play_emotion",
        "description": "Play a preset emotion or dance. name examples: happy / sad / curious / nod / shake / dance / yeah_nod / chicken_peck",
        "parameters": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    }},
    {"type": "function", "function": {
        "name": "stop_motion",
        "description": "Stop all current motion.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "get_current_time",
        "description": "Get current local time.",
        "parameters": {"type": "object", "properties": {}},
    }},
]


def call_once(prompt: str) -> dict:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "tools": TOOLS,
        "tool_choice": "auto",
        "temperature": 0.2,
        "max_tokens": 200,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"})

    t_start = time.perf_counter()
    ttft_ms = None
    text_parts = []
    tool_calls_acc = {}   # idx -> {name, arguments}
    usage = {}

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            for raw in resp:
                if not raw:
                    continue
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data: "):
                    continue
                body = line[6:]
                if body == "[DONE]":
                    break
                try:
                    chunk = json.loads(body)
                except Exception:
                    continue
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for ch in chunk.get("choices", []) or []:
                    delta = ch.get("delta") or {}
                    c = delta.get("content")
                    if c:
                        if ttft_ms is None:
                            ttft_ms = (time.perf_counter() - t_start) * 1000
                        text_parts.append(c)
                    for tc in delta.get("tool_calls") or []:
                        idx = tc.get("index", 0)
                        slot = tool_calls_acc.setdefault(idx, {"name": "", "arguments": ""})
                        fn = tc.get("function") or {}
                        if fn.get("name"):
                            slot["name"] = fn["name"]
                            if ttft_ms is None:
                                ttft_ms = (time.perf_counter() - t_start) * 1000
                        if fn.get("arguments") is not None:
                            slot["arguments"] += fn["arguments"]
    except Exception as e:
        return {"error": repr(e)[:200], "e2e_ms": (time.perf_counter() - t_start) * 1000}

    e2e_ms = (time.perf_counter() - t_start) * 1000
    text = "".join(text_parts)
    tcs = [tool_calls_acc[i] for i in sorted(tool_calls_acc)]
    out_tok = usage.get("completion_tokens", 0)
    return {
        "ttft_ms": round(ttft_ms, 1) if ttft_ms else None,
        "e2e_ms": round(e2e_ms, 1),
        "prompt_tok": usage.get("prompt_tokens", 0),
        "out_tok": out_tok,
        "toks_per_sec": round(out_tok / (e2e_ms/1000), 2) if e2e_ms > 0 else None,
        "text": text,
        "tool_calls": tcs,
    }


def main():
    print(f"V5 bench  model={MODEL}  trials={TRIALS}  prompts={len(PROMPTS)}")
    print(f"→ {OUT}\n")
    rows = []
    for pid, ptext, expected in PROMPTS:
        for trial in range(1, TRIALS + 1):
            t0 = time.perf_counter()
            res = call_once(ptext)
            wall = (time.perf_counter() - t0) * 1000
            row = {"pid": pid, "trial": trial, "prompt": ptext,
                   "expected_tool": expected, **res}
            rows.append(row)
            tc_names = [t["name"] for t in (res.get("tool_calls") or [])]
            text_preview = (res.get("text") or "")[:60].replace("\n", " ")
            ttft = res.get("ttft_ms")
            tok_s = res.get("toks_per_sec")
            print(f"  [{pid}/{trial}] ttft={ttft}  tok/s={tok_s}  tools={tc_names}  text='{text_preview}'")
    json.dump({"model": MODEL, "trials": TRIALS, "rows": rows},
              open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\nwrote {OUT}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
