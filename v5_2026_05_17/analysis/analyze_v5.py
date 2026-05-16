#!/usr/bin/env python3
"""v5 analysis: Dense Qwen3.6-27B vs MoE Qwen3.6-35B-A3B + MTP k=3 on voice workload.

Reads:
  data/results_moe_35b_mtp_tp2.json
  data/results_dense_27b_tp1.json

Reports per model:
  - tool-call accuracy (overall, chat false-fires, tool misses)
  - TTFT / e2e / tok/s (mean, median, stdev, range)
  - zh-TW vs zh-CN ratio on chat responses (TRAD markers vs SIMP markers,
    requires `opencc` for canonical t2s identity check; falls back to a
    char-marker heuristic if opencc not installed)

Outputs aggregate.json (machine-readable) and prints a human summary.

Caveats (do not generalize without reading the v5 README scope section):
  - N=3 trials × 10 prompts = 30 samples per model. Underpowered for
    deciding production-grade close calls; sufficient to falsify any claim
    of "Dense 27B is a free upgrade on this hardware".
  - Dense 27B forced TP=1 on a single 3090 (production was paused; we did
    not run Dense TP=2 because Qwen3.6-27B-AWQ (~13.5 GB) fits TP=1, and
    swapping it through TP=2 would have stolen GPU from the MoE arm during
    its bench window). The comparison is "what each model looks like at
    the *largest TP that fits its weights cleanly*", not a pure
    architecture isolation.
  - Dense ran without spec decoding (no MTP head exists for Qwen3.6-27B);
    MoE ran with MTP k=3 (production winner from v4.0). This is the
    fair production comparison; an isolated "Dense vs MoE both no-spec"
    arm is out of scope for this release.
"""
import json
import re
import statistics as st
import sys
from pathlib import Path

# ---- zh-TW / zh-CN detection ----
try:
    from opencc import OpenCC
    _cc_t2s = OpenCC("t2s")

    def trad_status(text):
        if not text:
            return "none"
        han = re.findall(r"[一-鿿]", text)
        if not han:
            return "none"
        s_han = "".join(han)
        return "TRAD" if _cc_t2s.convert(s_han) != s_han else "shared"
except Exception:
    _TRAD = set(
        "為麼樂書買賣應記車沒動來裡東問發現說話長樣這點國還學時間見點開關電腦進這對體當頭裡會學業講願氣個歲時錢愛網過內導語給東說語經當還動覺車員樣國連風機鳳權門總獨見覺識記實業課專塊試廠處員議週節電興奮樂業務識歲時點問題見錯誤節"
    )
    _SIMP = set(
        "为么乐书买卖应记车没动来里东问发现说话长样这点国还学时间见点开关电脑进这对体当头里会学业讲愿气个岁时钱爱网过内导语给东说语经当还动觉车员样国连风机凤权门总独见觉识记实业课专块试厂处员议周节电兴奋乐业务识岁时点问题见错误节"
    )

    def trad_status(text):
        if not text:
            return "none"
        han = re.findall(r"[一-鿿]", text)
        if not han:
            return "none"
        chars = set(han)
        if chars & _TRAD and not chars & _SIMP:
            return "TRAD"
        if chars & _SIMP and not chars & _TRAD:
            return "SIMP"
        if chars & _TRAD and chars & _SIMP:
            return "MIX"
        return "shared"


def summarize(path, label):
    data = json.load(open(path, encoding="utf-8"))
    rows = data["rows"]
    out = {"label": label, "path": str(path), "n": len(rows)}

    correct = chat_wrongfire = tool_miss = 0
    chat_total = tool_total = 0
    trad_counts = {"TRAD": 0, "SIMP": 0, "MIX": 0, "shared": 0, "none": 0}
    ttfts, e2es, tps = [], [], []
    bad_chat, missed_tools = [], []

    for r in rows:
        exp = r.get("expected_tool")
        actual = [t["name"] for t in (r.get("tool_calls") or [])]
        if exp is None:
            chat_total += 1
            if actual:
                chat_wrongfire += 1
                bad_chat.append({"pid": r["pid"], "trial": r["trial"], "fired": actual})
            else:
                correct += 1
        else:
            tool_total += 1
            if exp in actual:
                correct += 1
            else:
                tool_miss += 1
                missed_tools.append({"pid": r["pid"], "trial": r["trial"], "expected": exp, "actual": actual})

        if r.get("ttft_ms"):
            ttfts.append(r["ttft_ms"])
        if r.get("e2e_ms"):
            e2es.append(r["e2e_ms"])
        if r.get("toks_per_sec"):
            tps.append(r["toks_per_sec"])
        trad_counts[trad_status(r.get("text") or "")] += 1

    out["tool_accuracy"] = {
        "overall_correct": correct,
        "overall_total": len(rows),
        "overall_pct": round(correct / len(rows) * 100, 1),
        "chat_false_fires": chat_wrongfire,
        "chat_total": chat_total,
        "tool_misses": tool_miss,
        "tool_total": tool_total,
        "bad_chat_examples": bad_chat,
        "missed_tools": missed_tools,
    }
    out["latency"] = {
        "ttft_ms": _stats(ttfts),
        "e2e_ms": _stats(e2es),
        "toks_per_sec": _stats(tps),
    }
    out["zh_tw_status"] = trad_counts
    return out


def _stats(xs):
    if not xs:
        return None
    return {
        "mean": round(st.mean(xs), 2),
        "median": round(st.median(xs), 2),
        "stdev": round(st.stdev(xs) if len(xs) > 1 else 0, 2),
        "min": round(min(xs), 2),
        "max": round(max(xs), 2),
        "n": len(xs),
    }


def print_human(out):
    label = out["label"]
    print(f"=== {label}   (N={out['n']} rows) ===")
    ta = out["tool_accuracy"]
    print(f"  tool acc overall : {ta['overall_correct']}/{ta['overall_total']} = {ta['overall_pct']}%")
    print(f"  chat false-fires : {ta['chat_false_fires']}/{ta['chat_total']} chat prompts")
    print(f"  tool misses      : {ta['tool_misses']}/{ta['tool_total']} tool prompts")
    z = out["zh_tw_status"]
    print(f"  zh-TW status     : TRAD {z.get('TRAD', 0)}  SIMP {z.get('SIMP', 0)}  MIX {z.get('MIX', 0)}  shared {z.get('shared', 0)}  none {z.get('none', 0)}")
    L = out["latency"]
    if L["ttft_ms"]:
        s = L["ttft_ms"]
        print(f"  TTFT ms          : mean {s['mean']}  median {s['median']}  stdev {s['stdev']}  range [{s['min']}, {s['max']}]")
    if L["e2e_ms"]:
        s = L["e2e_ms"]
        print(f"  e2e  ms          : mean {s['mean']}  median {s['median']}  stdev {s['stdev']}")
    if L["toks_per_sec"]:
        s = L["toks_per_sec"]
        print(f"  tok/s            : mean {s['mean']}  median {s['median']}  stdev {s['stdev']}")
    print()


def main():
    here = Path(__file__).resolve().parent.parent
    moe = summarize(here / "data/results_moe_35b_mtp_tp2.json", "MoE Qwen3.6-35B-A3B (MTP k=3, TP=2, production)")
    dense = summarize(here / "data/results_dense_27b_tp1.json", "Dense Qwen3.6-27B (no MTP, TP=1 GPU1, enforce-eager)")
    print_human(moe)
    print_human(dense)

    # head-to-head
    mt, dt = moe["latency"]["ttft_ms"]["mean"], dense["latency"]["ttft_ms"]["mean"]
    mp, dp = moe["latency"]["toks_per_sec"]["mean"], dense["latency"]["toks_per_sec"]["mean"]
    me, de = moe["latency"]["e2e_ms"]["mean"], dense["latency"]["e2e_ms"]["mean"]
    ma, da = moe["tool_accuracy"]["overall_pct"], dense["tool_accuracy"]["overall_pct"]
    print("===== HEAD-TO-HEAD =====")
    print(f"  TTFT mean (ms)   : MoE {mt}  vs  Dense {dt}  →  MoE {dt/mt:.2f}× faster")
    print(f"  tok/s mean       : MoE {mp}  vs  Dense {dp}  →  MoE {mp/dp:.2f}× faster")
    print(f"  e2e  mean (ms)   : MoE {me}  vs  Dense {de}  →  MoE {de/me:.2f}× faster")
    print(f"  tool accuracy    : MoE {ma}%  vs  Dense {da}%  →  Δ {ma-da:+.1f} pp")

    json.dump({"moe": moe, "dense": dense}, open(here / "analysis/aggregate.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\nwrote analysis/aggregate.json")


if __name__ == "__main__":
    main()
