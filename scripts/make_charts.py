"""Make the README charts from the results files.  python scripts/make_charts.py

Each chart reads its own inputs and is skipped (with a message) if they're missing, so this also
works on a partial checkout. Output: results/charts/*.png

Style: one y-axis per panel, thin marks, a fixed color per entity (never per rank), a legend for
2+ series, recessive grid, text in ink colors (never series colors). Palette: validated reference
categorical slots 1-3 (blue, orange, aqua), light mode.
"""
import json
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

R = Path("results")
OUT = R / "charts"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"          # categorical slots 1, 2, 3
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e7e6e2", "#fcfcfb"
SLO_S = 15

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "font.size": 10, "axes.titlesize": 11,
    "axes.titlelocation": "left", "legend.frameon": False, "legend.labelcolor": INK,
})


def style(ax, grid_axis="y"):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(True, axis=grid_axis, color=GRID, lw=0.8, zorder=0)
    ax.tick_params(length=0)


def finish(fig, name, title, subtitle=None):
    h_pt = fig.get_figheight() * 72
    top = 0.985
    fig.text(0.01, top, title, ha="left", va="top", fontsize=13, color=INK)
    if subtitle:
        fig.text(0.01, top - 22 / h_pt, subtitle, ha="left", va="top", fontsize=9.5, color=INK2)
    header_pt = 48 if subtitle else 28
    fig.tight_layout(rect=(0, 0, 1, top - header_pt / h_pt))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)
    print(f"  saved {OUT / name}")


def need(*paths) -> bool:
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        print(f"  skipped: missing {', '.join(missing)}")
    return not missing


def load_summary() -> pd.DataFrame:
    d = pd.read_csv(R / "loadtest_summary.csv")
    return d.drop_duplicates(["label", "users"], keep="last")        # latest run per (label, users)


def log2_x(ax, levels):
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(levels))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.xaxis.set_minor_locator(NullLocator())


# 1 ------------------------------------------------------------------------------------------
def chart_admission():
    print("01 admission control")
    if not need(R / "loadtest_summary.csv"):
        return
    d = load_summary()
    series = [("version_a_guarded", "Version A (Python, guarded)", BLUE),
              ("llamacpp_fp16_np8", "llama.cpp alone", ORANGE),
              ("gw_llamacpp_admit", "llama.cpp + gateway (admission control)", AQUA)]
    users = [10, 100, 1000]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for label, name, color in series:
        s = d[(d["label"] == label) & d["users"].isin(users)].sort_values("users")
        if s.empty:
            continue
        ax.plot(s["users"], s["goodput_rps"], color=color, lw=2, marker="o", ms=8,
                markeredgecolor=SURFACE, markeredgewidth=2, label=name, zorder=3)
        last = s.iloc[-1]
        ax.annotate(f"{last['goodput_rps']:.2f}", (last["users"], last["goodput_rps"]), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=9, color=INK2)
    log_x = [10, 100, 1000]
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(log_x))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(7, 1800)
    ax.set_ylim(0, 1.85)
    ax.set_xlabel("concurrent users")
    ax.set_ylabel(f"goodput (answers within {SLO_S} s per second)")
    ax.legend(loc="upper left", fontsize=9, ncol=1)
    style(ax)
    finish(fig, "01_admission_control.png", "Admission control keeps goodput flat under overload",
           "Same 2 CPU cores. Without it, llama.cpp queues everyone until all requests miss the SLO (goodput 0).")


# 2 ------------------------------------------------------------------------------------------
def chart_vllm_batching():
    print("02 vLLM continuous batching")
    path = R / "vllm_t4_fp16_batch_sweep.csv"
    if not need(path):
        return
    s = pd.read_csv(path)
    levels = s["concurrency"].tolist()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, col, title, fmt in [(axes[0], "tokens_per_s", "Throughput (output tokens/s)", "{:,.0f}"),
                                (axes[1], "mean_latency_s", "Latency per request (s)", "{:.2f} s")]:
        ax.plot(s["concurrency"], s[col], color=BLUE, lw=2, marker="o", ms=8,
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
        for i in sorted({0, len(s) - 1, int(s["tokens_per_s"].idxmax())}):
            ax.annotate(fmt.format(s[col].iloc[i]), (s["concurrency"].iloc[i], s[col].iloc[i]),
                        xytext=(0, 10), textcoords="offset points", ha="center", fontsize=9, color=INK2)
        log2_x(ax, levels)
        ax.set_ylim(0, s[col].max() * 1.18)
        ax.set_title(title)
        ax.set_xlabel("concurrent requests (= batch size)")
        style(ax)
    best = s.loc[s["tokens_per_s"].idxmax()]
    gain = best["tokens_per_s"] / s["tokens_per_s"].iloc[0]
    finish(fig, "02_vllm_batching.png",
           f"One T4, vLLM: {gain:.0f}x the throughput of batch 1 at batch {int(best['concurrency'])}",
           f"Latency stays under {best['mean_latency_s']:.1f} s up to there; past it, requests mostly wait "
           "(32-token answers, one session).")


# 3 ------------------------------------------------------------------------------------------
def chart_cache():
    print("03 Redis cache")
    if not need(R / "loadtest_summary.csv"):
        return
    d = load_summary()
    rows = []
    for rr, name in [("0", "0%"), ("0.2", "20%"), ("0.5", "50%")]:
        m = d[(d["label"] == f"gw_llamacpp_cache_r{rr}") & (d["users"] == 100)]
        if m.empty:
            continue
        calls = None
        st = R / "raw" / f"gw_llamacpp_cache_r{rr}_u100_server_stats.json"
        if st.exists():
            calls = json.loads(st.read_text() or "{}").get("backend_calls")
        rows.append((name, float(m.iloc[-1]["goodput_rps"]), calls))
    if not rows:
        print("  skipped: no gw_llamacpp_cache_* rows")
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    x = range(len(rows))
    ax.bar(x, [r[1] for r in rows], width=0.55, color=BLUE, zorder=3)
    for i, (name, g, calls) in enumerate(rows):
        ax.annotate(f"{g:.1f} req/s", (i, g), xytext=(0, 4), textcoords="offset points", ha="center",
                    fontsize=10, color=INK)
    ax.set_xticks(list(x), [f"{r[0]} repeated prompts" + (f"\n{r[2]} backend calls" if r[2] is not None else "")
                            for r in rows])
    ax.set_ylabel(f"goodput (answers within {SLO_S} s per second)")
    ax.set_ylim(0, max(r[1] for r in rows) * 1.2)
    style(ax)
    finish(fig, "03_cache.png", "A response cache multiplies capacity when prompts repeat",
           "llama.cpp behind the gateway, 100 users. The backend does the same work; every hit is ~free.")


# 4 ------------------------------------------------------------------------------------------
def chart_quantization():
    print("04 quantization")
    if not need(R / "quality_summary.csv"):
        return
    q = pd.read_csv(R / "quality_summary.csv").drop_duplicates("label", keep="last").set_index("label")
    variants = [("llamacpp_fp16", "CPU fp16", BLUE), ("llamacpp_q8_0", "CPU Q8_0", BLUE),
                ("llamacpp_q4_k_m", "CPU Q4_K_M", BLUE), ("vllm_t4_fp16", "GPU fp16", ORANGE),
                ("vllm_t4_awq", "GPU AWQ 4-bit", ORANGE)]
    variants = [v for v in variants if v[0] in q.index]
    speed = {}
    cpu_p, gpu_p = R / "quant_cpu_summary.csv", R / "quant_gpu_summary.csv"
    if cpu_p.exists():
        c = pd.read_csv(cpu_p).set_index("quant")
        for k, lab in [("fp16", "llamacpp_fp16"), ("q8_0", "llamacpp_q8_0"), ("q4_k_m", "llamacpp_q4_k_m")]:
            if k in c.index:
                speed[lab] = float(c.loc[k, "tok_s_batch1_x_fp16"])
    if gpu_p.exists():
        g = pd.read_csv(gpu_p).set_index("variant")
        if {"vllm_t4_fp16", "vllm_t4_awq"} <= set(g.index):
            speed["vllm_t4_fp16"] = 1.0
            speed["vllm_t4_awq"] = float(g.loc["vllm_t4_awq", "tok_s_batch1"] / g.loc["vllm_t4_fp16", "tok_s_batch1"])
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), gridspec_kw={"width_ratios": [1.25, 1]})
    names = [v[1] for v in variants][::-1]
    ys = range(len(variants))
    ax = axes[0]
    for y, (lab, name, color) in zip(ys, variants[::-1]):
        r = q.loc[lab]
        acc, lo, hi = 100 * r["accuracy"], 100 * r["ci95_low"], 100 * r["ci95_high"]
        ax.barh(y, acc, height=0.55, color=color, zorder=3)
        ax.plot([lo, hi], [y, y], color=INK, lw=1.2, zorder=4)
        ax.annotate(f"{acc:.1f}%", (hi, y), xytext=(5, 0), textcoords="offset points", va="center",
                    fontsize=9, color=INK2)
    ax.set_yticks(list(ys), names)
    ax.set_xlim(0, 100)
    ax.axvline(25, color=MUTED, lw=1, ls=(0, (3, 3)), zorder=2)
    ax.annotate("chance (4 options)", (25, len(variants) - 0.4), xytext=(4, 0), textcoords="offset points",
                fontsize=8.5, color=MUTED)
    ax.set_ylim(-0.6, len(variants) - 0.1)
    ax.set_title("ARC-Easy accuracy, 1,000 questions (line = 95% CI)")
    style(ax, "x")
    ax = axes[1]
    for y, (lab, name, color) in zip(ys, variants[::-1]):
        if lab in speed:
            ax.barh(y, speed[lab], height=0.55, color=color, zorder=3)
            ax.annotate(f"{speed[lab]:.2f}x", (speed[lab], y), xytext=(5, 0), textcoords="offset points",
                        va="center", fontsize=9, color=INK2)
    ax.axvline(1, color=MUTED, lw=1, ls=(0, (3, 3)), zorder=2)
    ax.set_yticks(list(ys), [""] * len(variants))
    ax.set_ylim(-0.6, len(variants) - 0.1)
    ax.set_xlim(0, max(speed.values() or [2]) * 1.25)
    ax.set_title("Single-request speed vs its own fp16")
    style(ax, "x")
    handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=ORANGE)]
    axes[0].legend(handles, ["CPU (llama.cpp)", "GPU T4 (vLLM)"], loc="lower right", fontsize=9)
    finish(fig, "04_quantization.png", "Quantization: smaller files, mixed speed, quality must be checked",
           "AWQ's low score is mostly ignoring the answer format (32% followed it vs 91% for fp16).")


# 5 ------------------------------------------------------------------------------------------
def chart_hpa():
    print("05 Kubernetes autoscaling")
    watch, raw = R / "k8s_hpa_watch.txt", R / "raw" / "k8s_hpa_u30.csv"
    if not need(watch, raw):
        return
    w = []
    for line in watch.read_text().splitlines():
        m = re.match(r"(\d+)s desired=(\d+) ready=(\d+) pending=(\d+)", line)
        if m:
            w.append([int(g) for g in m.groups()])
    w = pd.DataFrame(w, columns=["t", "desired", "ready", "pending"])
    df = pd.read_csv(raw, keep_default_na=False)
    t0 = df["start_ts"].min()
    ok = df[df["status"].astype(str) == "200"]
    win = 30
    per = ((ok["end_ts"] - t0) // win * win).value_counts().sort_index()
    # the watcher started a few seconds before Locust; align both to Locust's start
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.6), sharex=True, gridspec_kw={"height_ratios": [1, 1.2]})
    ax = axes[0]
    ax.step(w["t"], w["ready"], where="post", color=BLUE, lw=2, label="ready replicas", zorder=3)
    ax.step(w["t"], w["desired"], where="post", color=ORANGE, lw=2, ls=(0, (4, 2)),
            label="replicas the autoscaler wants", zorder=3)
    ax.set_ylim(0, max(w["desired"].max(), w["ready"].max()) + 0.8)
    ax.set_yticks(range(0, int(ax.get_ylim()[1]) + 1))
    ax.set_title("Replicas")
    ax.legend(loc="lower center", fontsize=9, ncol=2)
    style(ax)
    ax = axes[1]
    ax.bar(per.index + win / 2, per.values / win, width=win * 0.85, color=BLUE, zorder=3)
    ax.set_title(f"Successful answers per second ({win} s windows)")
    ax.set_xlabel("seconds since the load started (30 users)")
    style(ax)
    end = df["end_ts"].max() - t0
    for a in axes:
        a.axvline(end, color=MUTED, lw=1, ls=(0, (3, 3)))
    axes[0].annotate("load stops", (end, axes[0].get_ylim()[1] * 0.92), xytext=(4, 0),
                     textcoords="offset points", fontsize=8.5, color=MUTED)
    finish(fig, "05_autoscaling.png", "Kubernetes HPA: 1 -> 3 replicas in ~1 minute, 4th stays Pending",
           "Node out of reservable CPU ('Insufficient cpu'). 3 one-core replicas on 2 physical cores ~= one 2-core replica.")


# 6 ------------------------------------------------------------------------------------------
def chart_capacity():
    print("06 capacity plan")
    path = R / "capacity_plan.csv"
    if not need(path):
        return
    p = pd.read_csv(path)
    p = p[p["cache_hit_rate"] == 0]
    tiers = [100, 10_000, 1_000_000]
    tier_name = {100: "100", 10_000: "10K", 1_000_000: "1M"}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    for ax, (wl, title) in zip(axes, [("benchmark", "Short answers (~50 tokens, measured)"),
                                      ("chat", "Chat answers (~300 tokens, extrapolated)")]):
        for j, (hw, name, color) in enumerate([("cpu_llamacpp", "CPU: c7i.xlarge (llama.cpp)", BLUE),
                                               ("gpu_t4_vllm", "GPU: g4dn.xlarge, T4 (vLLM)", ORANGE)]):
            vals, reps = [], []
            for t in tiers:
                r = p[(p["workload"] == wl) & (p["hardware"] == hw) & (p["dau"] == t)].iloc[0]
                vals.append(r["total_usd_month"]); reps.append(int(r["replicas_planned"]))
            xs = [i + (j - 0.5) * 0.36 for i in range(len(tiers))]
            ax.bar(xs, vals, width=0.33, color=color, label=name, zorder=3)
            for x, v, n in zip(xs, vals, reps):
                ax.annotate(f"${v / 1000:,.1f}K\n{n} {'VMs' if hw.startswith('cpu') else 'GPUs'}" if v >= 1000
                            else f"${v:,.0f}\n{n} {'VMs' if hw.startswith('cpu') else 'GPUs'}",
                            (x, v), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8,
                            color=INK2)
        ax.set_yscale("log")
        ax.set_ylim(100, p["total_usd_month"].max() * 8)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:,.0f}"))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_xticks(range(len(tiers)), [f"{tier_name[t]} daily users" for t in tiers])
        ax.set_title(title)
        style(ax)
    axes[0].set_ylabel("cost per month (log scale)")
    axes[0].legend(loc="upper left", fontsize=9)
    finish(fig, "06_capacity_cost.png", "Monthly cost by scale: GPUs win as traffic and answer length grow",
           "AWS us-east-1 on-demand, Nov 2026. 70% target utilization, N+1, min 2 replicas. Model: Qwen2.5-0.5B.")


if __name__ == "__main__":
    for f in (chart_admission, chart_vllm_batching, chart_cache, chart_quantization, chart_hpa, chart_capacity):
        try:
            f()
        except Exception as e:                     # one broken input must not stop the rest
            print(f"  FAILED: {type(e).__name__}: {e}")
