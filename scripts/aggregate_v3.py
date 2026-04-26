"""v3 deep verification + aggregation. Outputs summary JSON + console table."""
import json, os, statistics

HERE = os.path.dirname(os.path.abspath(__file__))
no_mtp = json.load(open(os.path.join(HERE, "results_v3_no_mtp.json"), encoding="utf-8"))
mtp = json.load(open(os.path.join(HERE, "results_v3_mtp.json"), encoding="utf-8"))


def collect_exp1(d):
    flat_decode_tpot = []
    flat_tok_s = []
    flat_ttft = []
    flat_ct = []
    per_prompt = {p: {"decode_tpot": [], "tok_s": [], "ttft": [], "ct": [], "sha1": []} for p in range(1, 6)}
    for trial in d["exp1_dialog"]["trials"]:
        for i, r in enumerate(trial, 1):
            flat_decode_tpot.append(r["decode_tpot_ms"])
            flat_tok_s.append(r["tok_s"])
            flat_ttft.append(r["ttft_s"] * 1000)
            flat_ct.append(r["ct"])
            per_prompt[i]["decode_tpot"].append(r["decode_tpot_ms"])
            per_prompt[i]["tok_s"].append(r["tok_s"])
            per_prompt[i]["ttft"].append(r["ttft_s"] * 1000)
            per_prompt[i]["ct"].append(r["ct"])
            per_prompt[i]["sha1"].append(r["text_sha1"])
    return {
        "flat_decode_tpot": flat_decode_tpot,
        "flat_tok_s": flat_tok_s,
        "flat_ttft": flat_ttft,
        "flat_ct": flat_ct,
        "per_prompt": per_prompt,
    }


def collect_exp3(d):
    out = {}
    for ckey, cdata in d["exp3_concurrent"].items():
        cn = int(ckey.replace("C", ""))
        agg = [t["aggregate_tok_s"] for t in cdata["trials"]]
        ttft = [t["mean_ttft_ms"] for t in cdata["trials"]]
        decode_tpot = [t["mean_decode_tpot_ms"] for t in cdata["trials"]]
        out[cn] = {"agg": agg, "ttft": ttft, "decode_tpot": decode_tpot}
    return out


e1_no = collect_exp1(no_mtp)
e1_mtp = collect_exp1(mtp)
e3_no = collect_exp3(no_mtp)
e3_mtp = collect_exp3(mtp)


def stats(xs):
    return statistics.mean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0)


print("=" * 86)
print("v3 CLEAN A/B — matched flags 0.90/8/hermes, prefix-cache DISABLED, streaming, N=5")
print("=" * 86)

print("\nEXP 1 — sequential dialog (5 prompts × 5 trials = 25 measurements per phase)")
print("-" * 86)
m_no, s_no = stats(e1_no["flat_decode_tpot"])
m_mtp, s_mtp = stats(e1_mtp["flat_decode_tpot"])
print(f"  decode_tpot (ms/output token)  no-MTP: {m_no:.3f} ± {s_no:.3f}   MTP: {m_mtp:.3f} ± {s_mtp:.3f}")
print(f"  Δ decode_tpot                  MTP {(m_mtp - m_no) / m_no * 100:+.2f}%   ⇒ MTP {1 / (m_mtp / m_no) * 100 - 100:+.2f}% faster decode rate")
print()

m_no, s_no = stats(e1_no["flat_tok_s"])
m_mtp, s_mtp = stats(e1_mtp["flat_tok_s"])
print(f"  total tok/s (incl prefill)     no-MTP: {m_no:.2f} ± {s_no:.2f}   MTP: {m_mtp:.2f} ± {s_mtp:.2f}")
print(f"  Δ                              MTP {(m_mtp - m_no) / m_no * 100:+.2f}%")
print()

m_no, s_no = stats(e1_no["flat_ttft"])
m_mtp, s_mtp = stats(e1_mtp["flat_ttft"])
print(f"  TTFT (ms)                      no-MTP: {m_no:.1f} ± {s_no:.1f}   MTP: {m_mtp:.1f} ± {s_mtp:.1f}")
print()

print("Per-prompt decode_tpot (mean over 5 trials):")
for i in range(1, 6):
    no = statistics.mean(e1_no["per_prompt"][i]["decode_tpot"])
    mtp_ = statistics.mean(e1_mtp["per_prompt"][i]["decode_tpot"])
    no_cts = e1_no["per_prompt"][i]["ct"]
    mtp_cts = e1_mtp["per_prompt"][i]["ct"]
    print(
        f"  p{i}: no-MTP {no:.3f}ms (ct={set(no_cts)})   MTP {mtp_:.3f}ms (ct={set(mtp_cts)})   "
        f"Δ {(mtp_ - no) / no * 100:+.2f}%"
    )
print()

print("\nEXP 3 — concurrent stress (20 reqs × 5 trials per concurrency)")
print("-" * 86)
print(f"{'C':>3} | {'no-MTP agg':>11} | {'MTP agg':>11} | {'Δ agg':>7} | {'no-MTP TPOT':>11} | {'MTP TPOT':>9} | {'Δ TPOT':>7}")
print("-" * 86)
for c in [1, 4, 8]:
    a_no = statistics.mean(e3_no[c]["agg"])
    a_mtp = statistics.mean(e3_mtp[c]["agg"])
    t_no = statistics.mean(e3_no[c]["decode_tpot"])
    t_mtp = statistics.mean(e3_mtp[c]["decode_tpot"])
    print(
        f"{c:>3} | {a_no:>9.2f} | {a_mtp:>9.2f} | {(a_mtp - a_no) / a_no * 100:+6.2f}% | "
        f"{t_no:>11.3f} | {t_mtp:>9.3f} | {(t_mtp - t_no) / t_no * 100:+6.2f}%"
    )
print()

print("\nDETERMINISM CHECK — SHA1 of response text per trial (vLLM has known intrinsic non-determinism with chunked-prefill)")
print("-" * 86)
for tag, e1 in [("no-MTP", e1_no), ("MTP", e1_mtp)]:
    print(f"  [{tag}]")
    for i in range(1, 6):
        sha1s = e1["per_prompt"][i]["sha1"]
        unique = list(set(sha1s))
        print(f"    p{i}: {len(unique)} unique response(s) across 5 trials  ({sha1s})")
print()

print("\nSANITY — sample response previews (first 200 chars)")
print("-" * 86)
for tag, d in [("no-MTP", no_mtp), ("MTP", mtp)]:
    print(f"  [{tag}]")
    trial1 = d["exp1_dialog"]["trials"][0]
    for i, r in enumerate(trial1, 1):
        preview = r["text_preview"].replace("\n", " ⏎ ")
        print(f"    p{i} (sha1={r['text_sha1']}, ct={r['ct']}, len={r['text_len']}): {preview[:150]}...")

# Save summary
summary = {
    "version": "v3",
    "host": "s1 (reachy-compute, 2× RTX 3090 PCIe Gen4 x8, 24GB each)",
    "vllm": "0.19.1",
    "model": "QuantTrio/Qwen3.6-35B-A3B-AWQ (AWQ-Marlin Q4)",
    "config": {
        "tensor_parallel_size": 2,
        "gpu_memory_utilization": 0.90,
        "max_num_seqs": 8,
        "tool_call_parser": "hermes",
        "prefix_caching": "DISABLED (--no-enable-prefix-caching)",
        "streaming": True,
        "seed": 42,
        "temperature": 0.5,
        "max_tokens": 200,
        "n_trials_exp1": 5,
        "n_trials_exp3": 5,
        "concurrencies_exp3": [1, 4, 8],
    },
    "exp1_dialog_summary": {
        "no_mtp": {
            "n": len(e1_no["flat_decode_tpot"]),
            "decode_tpot_ms_mean": statistics.mean(e1_no["flat_decode_tpot"]),
            "decode_tpot_ms_stdev": statistics.stdev(e1_no["flat_decode_tpot"]),
            "tok_s_mean": statistics.mean(e1_no["flat_tok_s"]),
            "tok_s_stdev": statistics.stdev(e1_no["flat_tok_s"]),
            "ttft_ms_mean": statistics.mean(e1_no["flat_ttft"]),
        },
        "mtp": {
            "n": len(e1_mtp["flat_decode_tpot"]),
            "decode_tpot_ms_mean": statistics.mean(e1_mtp["flat_decode_tpot"]),
            "decode_tpot_ms_stdev": statistics.stdev(e1_mtp["flat_decode_tpot"]),
            "tok_s_mean": statistics.mean(e1_mtp["flat_tok_s"]),
            "tok_s_stdev": statistics.stdev(e1_mtp["flat_tok_s"]),
            "ttft_ms_mean": statistics.mean(e1_mtp["flat_ttft"]),
        },
        "delta_decode_tpot_pct": (
            statistics.mean(e1_mtp["flat_decode_tpot"]) - statistics.mean(e1_no["flat_decode_tpot"])
        )
        / statistics.mean(e1_no["flat_decode_tpot"]) * 100,
    },
    "exp3_concurrent_summary": {
        f"C{c}": {
            "no_mtp_agg_tok_s_mean": statistics.mean(e3_no[c]["agg"]),
            "no_mtp_decode_tpot_ms_mean": statistics.mean(e3_no[c]["decode_tpot"]),
            "mtp_agg_tok_s_mean": statistics.mean(e3_mtp[c]["agg"]),
            "mtp_decode_tpot_ms_mean": statistics.mean(e3_mtp[c]["decode_tpot"]),
            "delta_agg_pct": (statistics.mean(e3_mtp[c]["agg"]) - statistics.mean(e3_no[c]["agg"]))
            / statistics.mean(e3_no[c]["agg"]) * 100,
            "delta_decode_tpot_pct": (
                statistics.mean(e3_mtp[c]["decode_tpot"]) - statistics.mean(e3_no[c]["decode_tpot"])
            )
            / statistics.mean(e3_no[c]["decode_tpot"]) * 100,
        }
        for c in [1, 4, 8]
    },
    "interpretation": (
        "Under matched flags (0.90/8/hermes) AND prefix-caching DISABLED, MTP k=1 is "
        "decisively faster on 2× RTX 3090 PCIe across all measured operating points: "
        "single-stream sequential dialog (decode_tpot -22%, ~+27% faster decode rate), "
        "concurrent C=1 (-21%), C=4 (-13%), C=8 (-11%). This refutes the v2 published "
        "-12% NET LOSS framing on this hardware. The v2 result was a flag confound "
        "(0.80/2 vs 0.90/8). The Modal A100 v2 -11.4% finding was with prefix-caching "
        "ENABLED, which is known per vllm Issue #38182 to drop hit rate 92%→71% under "
        "MTP — that's an MTP-prefix-cache interaction artifact, not a property of MTP "
        "itself. With prefix-caching disabled (this v3), MTP's compute speedup is "
        "isolated and consistently positive."
    ),
}
with open(os.path.join(HERE, "v3_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print("\n=== written: v3_summary.json ===")
