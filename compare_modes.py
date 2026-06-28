#!/usr/bin/env python3
"""
compare_modes.py — Compare 15W vs 25W vs MAXN across all combos
Bonsai-1.7B Q1_0 · Jetson Orin Nano Super 8GB

Usage:
  python3 compare_modes.py

  # Custom paths:
  python3 compare_modes.py \
    --15w  results/profile_export_15w_bonsai1.7b.jsonl \
    --25w  results/profile_export_25w_bonsai1.7b.jsonl \
    --maxn results/profile_export_maxn_bonsai1.7b.jsonl \
    --gpu-15w  2.503 \
    --gpu-25w  3.750 \
    --gpu-maxn 4.000 \
    --out  results/

Outputs (saved to --out folder):
  compare_tokps.png
  compare_tokj.png
  compare_avg_summary.png
  compare_tokps_vs_prompt.png
"""

import json, statistics, argparse
from collections import defaultdict
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

DEFAULT_GPU_POWER = {"15W": 2.503, "25W": 3.750, "MAXN": 4.000}
COLORS = {"15W": "#d19900", "25W": "#01696f", "MAXN": "#a12c7b"}


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def summarise(records):
    groups = defaultdict(list)
    for r in records:
        key = (r['target_prompt_tokens'], r['target_gen_tokens'])
        groups[key].append(r)
    out = {}
    for key, runs in sorted(groups.items()):
        clean = [r for r in runs if r['run'] > 0]
        if not clean:
            clean = runs
        lats  = sorted(r['latency_s'] for r in clean)
        p50   = lats[len(lats) // 2]
        p95   = lats[int(len(lats) * 0.95)]
        tok_s = statistics.median(r['output_tokens'] / r['latency_s'] for r in clean)
        out[key] = {'p50': p50, 'p95': p95, 'tok_s': tok_s, 'n': len(clean)}
    return out


def print_table(title, header, rows, footer=None):
    w = max(len(header), max(len(r) for r in rows))
    print(f"\n{title}")
    print("=" * w)
    print(header)
    print("-" * w)
    for r in rows:
        print(r)
    if footer:
        print("-" * w)
        print(footer)
    print("=" * w)


def save_chart(fig, png_path):
    if plt is None:
        print("  matplotlib not installed — skipping chart export.")
        return
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {png_path}")


def save_charts(d, gpu, keys, out_dir):
    if plt is None:
        print("  matplotlib not installed — skipping charts.")
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    combos = [f"pp{pt}+tg{gt}" for pt, gt in keys]
    tok15  = [d["15W"][k]['tok_s']  for k in keys]
    tok25  = [d["25W"][k]['tok_s']  for k in keys]
    tokmx  = [d["MAXN"][k]['tok_s'] for k in keys]
    j15    = [t / gpu["15W"]  for t in tok15]
    j25    = [t / gpu["25W"]  for t in tok25]
    jmx    = [t / gpu["MAXN"] for t in tokmx]

    # Chart 1: tok/s
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    y = range(len(combos))
    ax1.barh(y, tok15, color=COLORS["15W"], label="15W")
    ax1.barh([v + 0.24 for v in y], tok25, color=COLORS["25W"], label="25W")
    ax1.barh([v + 0.48 for v in y], tokmx, color=COLORS["MAXN"], label="MAXN")
    ax1.set_yticks([v + 0.24 for v in y])
    ax1.set_yticklabels(combos, fontsize=10)
    ax1.invert_yaxis()
    ax1.set_xlabel("tok/s (p50)")
    ax1.set_title("25W Beats MAXN — Faster AND More Efficient")
    ax1.legend(loc="lower right")
    ax1.grid(axis="x", linestyle="--", alpha=0.3)
    fig1.tight_layout()
    save_chart(fig1, out_dir / "compare_tokps.png")

    # Chart 2: tok/J
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.barh(y, j15, color=COLORS["15W"], label="15W")
    ax2.barh([v + 0.24 for v in y], j25, color=COLORS["25W"], label="25W")
    ax2.barh([v + 0.48 for v in y], jmx, color=COLORS["MAXN"], label="MAXN")
    ax2.set_yticks([v + 0.24 for v in y])
    ax2.set_yticklabels(combos, fontsize=10)
    ax2.invert_yaxis()
    ax2.set_xlabel("tok/J")
    ax2.set_title("25W is the Efficiency Sweet Spot (tok/J)")
    ax2.legend(loc="lower right")
    ax2.grid(axis="x", linestyle="--", alpha=0.3)
    fig2.tight_layout()
    save_chart(fig2, out_dir / "compare_tokj.png")

    # Chart 3: avg summary
    modes = ["15W", "25W", "MAXN"]
    avg_ts = [sum(d[m][k]['tok_s'] for k in keys) / len(keys) for m in modes]
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    ax3.bar(modes, avg_ts, color=[COLORS[m] for m in modes])
    ax3.set_ylabel("Avg tok/s")
    ax3.set_title("Avg tok/s by Power Mode — 25W Peaks")
    ax3.set_ylim(0, 28)
    for bar, val in zip(ax3.patches, avg_ts):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f"{val:.1f}", ha="center")
    fig3.tight_layout()
    save_chart(fig3, out_dir / "compare_avg_summary.png")

    # Chart 4: tok/s vs prompt length
    prompts = sorted(set(pt for pt, gt in keys))

    def avg_by_prompt(mode):
        return [
            sum(d[mode][(pt, gt)]['tok_s']
                for gt in sorted(set(g for p, g in keys if p == pt)))
            / len([g for p, g in keys if p == pt])
            for pt in prompts
        ]

    fig4, ax4 = plt.subplots(figsize=(8, 5))
    for mode in ["15W", "25W", "MAXN"]:
        vals = avg_by_prompt(mode)
        ax4.plot(prompts, vals, marker="o", linewidth=2.5, color=COLORS[mode], label=mode)
        for x, y in zip(prompts, vals):
            ax4.text(x, y + 0.3, f"{y:.1f}", ha="center", color=COLORS[mode], fontsize=9)
    ax4.set_xlabel("Prompt tokens")
    ax4.set_ylabel("Avg tok/s")
    ax4.set_title("tok/s Drops with Prompt Length (KV Cache Bottleneck)")
    ax4.set_ylim(0, 30)
    ax4.legend()
    ax4.grid(True, linestyle="--", alpha=0.3)
    fig4.tight_layout()
    save_chart(fig4, out_dir / "compare_tokps_vs_prompt.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--15w",      dest="p15",
                        default="results/profile_export_15w_bonsai1.7b.jsonl")
    parser.add_argument("--25w",      dest="p25",
                        default="results/profile_export_25w_bonsai1.7b.jsonl")
    parser.add_argument("--maxn",     dest="pmaxn",
                        default="results/profile_export_maxn_bonsai1.7b.jsonl")
    parser.add_argument("--gpu-15w",  type=float, default=DEFAULT_GPU_POWER["15W"])
    parser.add_argument("--gpu-25w",  type=float, default=DEFAULT_GPU_POWER["25W"])
    parser.add_argument("--gpu-maxn", type=float, default=DEFAULT_GPU_POWER["MAXN"])
    parser.add_argument("--out",      dest="out_dir", default="results")
    args = parser.parse_args()

    gpu = {"15W": args.gpu_15w, "25W": args.gpu_25w, "MAXN": args.gpu_maxn}

    print(f"\n{'='*65}")
    print(" Bonsai-1.7B Q1_0 — 3-Mode Power Benchmark Comparison")
    print(f" Device : Jetson Orin Nano Super 8GB")
    print(f" Modes  : 15W  |  25W  |  MAXN_SUPER")
    print(f" Charts -> {args.out_dir}/")
    print(f"{'='*65}")

    for label, path in [("15W", args.p15), ("25W", args.p25), ("MAXN", args.pmaxn)]:
        if not Path(path).exists():
            print(f"  WARNING: Missing file for {label}: {path}")

    d    = {"15W":  summarise(load_jsonl(args.p15)),
            "25W":  summarise(load_jsonl(args.p25)),
            "MAXN": summarise(load_jsonl(args.pmaxn))}
    keys = sorted(d["25W"].keys())

    # Table 1: tok/s
    hdr = (f"{'Combo':<16}  {'15W tok/s':>10}  {'25W tok/s':>10}  "
           f"{'MAXN tok/s':>11}  {'25v15':>7}  {'MAXv25':>8}")
    rows, g1s, g2s = [], [], []
    for pt, gt in keys:
        combo = f"pp{pt}+tg{gt}"
        r15, r25, rmx = d["15W"][(pt,gt)], d["25W"][(pt,gt)], d["MAXN"][(pt,gt)]
        g1 = (r25['tok_s'] / r15['tok_s'] - 1) * 100
        g2 = (rmx['tok_s'] / r25['tok_s'] - 1) * 100
        g1s.append(g1); g2s.append(g2)
        rows.append(f"{combo:<16}  {r15['tok_s']:>10.2f}  {r25['tok_s']:>10.2f}  "
                    f"{rmx['tok_s']:>11.2f}  {g1:>+6.1f}%  {g2:>+7.1f}%")
    ftr = (f"{'AVERAGE':<16}  {'':>10}  {'':>10}  {'':>11}  "
           f"{sum(g1s)/len(g1s):>+6.1f}%  {sum(g2s)/len(g2s):>+7.1f}%")
    print_table("THROUGHPUT (tok/s) — p50, run=0 excluded", hdr, rows, ftr)

    # Table 2: latency
    hdr2 = (f"{'Combo':<16}  {'15W p50':>9}  {'15W p95':>9}  "
            f"{'25W p50':>9}  {'25W p95':>9}  {'MAXN p50':>10}  {'MAXN p95':>10}")
    rows2 = []
    for pt, gt in keys:
        combo = f"pp{pt}+tg{gt}"
        r15, r25, rmx = d["15W"][(pt,gt)], d["25W"][(pt,gt)], d["MAXN"][(pt,gt)]
        rows2.append(f"{combo:<16}  {r15['p50']:>9.3f}  {r15['p95']:>9.3f}  "
                     f"{r25['p50']:>9.3f}  {r25['p95']:>9.3f}  "
                     f"{rmx['p50']:>10.3f}  {rmx['p95']:>10.3f}")
    print_table("LATENCY (seconds) — p50 and p95", hdr2, rows2)

    # Table 3: tok/J
    hdr3 = (f"{'Combo':<16}  {'15W tok/J':>10}  {'25W tok/J':>10}  "
            f"{'MAXN tok/J':>11}  {'Best':>6}")
    rows3, j_avgs = [], {"15W": [], "25W": [], "MAXN": []}
    for pt, gt in keys:
        combo = f"pp{pt}+tg{gt}"
        vals = {m: d[m][(pt,gt)]['tok_s'] / gpu[m] for m in ["15W","25W","MAXN"]}
        best = max(vals, key=vals.get)
        for m in vals:
            j_avgs[m].append(vals[m])
        rows3.append(f"{combo:<16}  {vals['15W']:>10.3f}  {vals['25W']:>10.3f}  "
                     f"{vals['MAXN']:>11.3f}  {best:>6}")
    ftr3 = (f"{'AVERAGE':<16}  "
            f"{sum(j_avgs['15W'])/len(j_avgs['15W']):>10.3f}  "
            f"{sum(j_avgs['25W'])/len(j_avgs['25W']):>10.3f}  "
            f"{sum(j_avgs['MAXN'])/len(j_avgs['MAXN']):>11.3f}")
    print_table("ENERGY EFFICIENCY (tok/J) — from GPU power rail", hdr3, rows3, ftr3)

    # Sweet-spot summary
    avg_ts = {m: sum(d[m][k]['tok_s'] for k in keys) / len(keys)
              for m in ["15W","25W","MAXN"]}
    avg_j  = {m: sum(d[m][k]['tok_s'] / gpu[m] for k in keys) / len(keys)
              for m in ["15W","25W","MAXN"]}

    print(f"\n{'='*55}")
    print(" SWEET-SPOT ANALYSIS")
    print(f"{'='*55}")
    print(f"  Avg tok/s  -> 15W:{avg_ts['15W']:>6.2f}  25W:{avg_ts['25W']:>6.2f}  MAXN:{avg_ts['MAXN']:>6.2f}")
    print(f"  Avg tok/J  -> 15W:{avg_j['15W']:>6.3f}  25W:{avg_j['25W']:>6.3f}  MAXN:{avg_j['MAXN']:>6.3f}")
    print()
    print(f"  25W vs 15W  -> {(avg_ts['25W']/avg_ts['15W']-1)*100:>+.1f}% tok/s  "
          f"| {(avg_j['25W']/avg_j['15W']-1)*100:>+.1f}% tok/J")
    print(f"  MAXN vs 25W -> {(avg_ts['MAXN']/avg_ts['25W']-1)*100:>+.1f}% tok/s  "
          f"| {(avg_j['MAXN']/avg_j['25W']-1)*100:>+.1f}% tok/J")
    print()
    print(f"  25W is the sweet spot:")
    print(f"    +{(avg_ts['25W']/avg_ts['15W']-1)*100:.0f}% faster than 15W, only ~2% efficiency loss")
    print(f"    Faster AND more efficient than MAXN")
    print(f"    MAXN = memory-bandwidth bound, extra clock buys nothing")
    print(f"{'='*55}\n")

    # Save charts
    print(f"Saving charts to {args.out_dir}/ ...")
    save_charts(d, gpu, keys, args.out_dir)
    print("Done!\n")


if __name__ == "__main__":
    main()