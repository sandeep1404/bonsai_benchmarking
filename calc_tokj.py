#!/usr/bin/env python3
"""
calc_tokj.py — Parse tegrastats + profile jsonl → compute real tok/J per combo

Usage:
  python3 calc_tokj.py \
    --tegra  results/tegrastats_25w_bonsai1.7b.txt \
    --jsonl  results/profile_export_25w_bonsai1.7b.jsonl \
    --mode   25W \
    --out    results/tokj_25w.csv

  # For all 3 modes at once:
  python3 calc_tokj.py --all

Outputs:
  results/tokj_<mode>.csv          per-combo CSV with tok/s, avg_W, tok/J
  results/tokj_all_modes.csv       merged CSV (if --all)
  results/tokj_comparison.png      bar chart (if --all)
"""

import re, json, csv, argparse, statistics
from collections import defaultdict
from pathlib import Path

# ── Defaults ─────────────────────────────────────────────────────────────────
FILES = {
    "15W":  ("results/tegrastats_15w_bonsai1.7b.txt",
             "results/profile_export_15w_bonsai1.7b.jsonl"),
    "25W":  ("results/tegrastats_25w_bonsai1.7b.txt",
             "results/profile_export_25w_bonsai1.7b.jsonl"),
    "MAXN": ("results/tegrastats_maxn_bonsai1.7b.txt",
             "results/profile_export_maxn_bonsai1.7b.jsonl"),
}
COLORS = {"15W": "#d19900", "25W": "#01696f", "MAXN": "#a12c7b"}

# ── Tegrastats parser ─────────────────────────────────────────────────────────
# Line format (unix-ts variant):
#   <unix_ts> <date> <time> ... VDD_CPU_GPU_CV <instant>mW/<avg>mW ...
# OR (no unix-ts):
#   <date> <time> ... VDD_CPU_GPU_CV <instant>mW/<avg>mW ...

TS_RE    = re.compile(r'^(\d+\.\d+)\s+\S+\s+(\d{2}:\d{2}:\d{2})')
NOTSRE   = re.compile(r'^\S+\s+(\d{2}:\d{2}:\d{2})')
POWER_RE = re.compile(r'VDD_CPU_GPU_CV\s+(\d+)mW/(\d+)mW')

def parse_tegrastats(path):
    """Return list of (unix_ts_float, instant_mW, avg_mW). unix_ts is 0 if absent."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m_ts  = TS_RE.match(line)
            m_pow = POWER_RE.search(line)
            if not m_pow:
                continue
            ts  = float(m_ts.group(1)) if m_ts else 0.0
            rows.append((ts, int(m_pow.group(1)), int(m_pow.group(2))))
    return rows

# ── Profile jsonl loader ──────────────────────────────────────────────────────
def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]

# ── Core: match inference windows to tegrastats ───────────────────────────────
def _get_run_window(record):
    """Return (start_ts, end_ts) for a profile record using either legacy or current field names."""
    latency_s = float(record.get('latency_s', 0) or 0)

    start_keys = ('start_ts', 'start_time', 'request_start_ts')
    end_keys = ('end_ts', 'end_time', 'request_end_ts')

    t0 = None
    for key in start_keys:
        if key in record:
            t0 = float(record[key])
            break
    if t0 is None:
        for key in ('request_start_ns', 'start_ns'):
            if key in record:
                t0 = float(record[key]) / 1e9
                break
    if t0 is None:
        t0 = 0.0

    t1 = None
    for key in end_keys:
        if key in record:
            t1 = float(record[key])
            break
    if t1 is None:
        for key in ('request_end_ns', 'end_ns'):
            if key in record:
                t1 = float(record[key]) / 1e9
                break
    if t1 is None:
        t1 = t0 + latency_s

    return t0, t1


def calc_tokj(tegra_rows, profile_records):
    """
    For each warm inference run (run > 0):
      1. Find tegrastats samples that overlap the run window
         (start_ts, end_ts = start_ts + latency_s)
      2. Compute avg VDD_CPU_GPU_CV (mW) for that window
      3. Compute energy_J = avg_W * latency_s
      4. tok/J = output_tokens / energy_J

    Returns dict keyed by (prompt_tokens, gen_tokens) →
      { tok_s, avg_W, energy_J, tokj, n }
    """
    # Build tegrastats index: list of (ts, instant_mW)
    teg = [(ts, inst) for ts, inst, _ in tegra_rows if ts > 0]

    if not teg:
        print("  ⚠️  No unix timestamps in tegrastats — falling back to rolling avg per combo.")
        return _fallback_avg(tegra_rows, profile_records)

    results = defaultdict(list)
    for r in profile_records:
        if r['run'] == 0:
            continue
        key = (r['target_prompt_tokens'], r['target_gen_tokens'])
        t0, t1 = _get_run_window(r)

        if t0 <= 0 and t1 <= 0:
            continue

        # Collect tegrastats samples inside the window
        samples = [mw for ts, mw in teg if t0 <= ts <= t1]

        if not samples:
            # Widen search ±0.5s to catch edge samples
            samples = [mw for ts, mw in teg if (t0 - 0.5) <= ts <= (t1 + 0.5)]

        if not samples:
            continue  # skip if no overlap found

        avg_w    = statistics.mean(samples) / 1000.0   # mW → W
        energy_j = avg_w * r['latency_s']
        tok_s    = r['output_tokens'] / r['latency_s']
        tokj     = r['output_tokens'] / energy_j if energy_j > 0 else 0

        results[key].append({
            'tok_s':    tok_s,
            'avg_W':    avg_w,
            'energy_J': energy_j,
            'tokj':     tokj,
            'latency_s': r['latency_s'],
            'output_tokens': r['output_tokens'],
        })

    if not results:
        print("  ⚠️  No overlapping tegrastats samples found for the profile windows — falling back to global average power.")
        return _fallback_avg(tegra_rows, profile_records)

    # Aggregate per combo
    out = {}
    for key, runs in sorted(results.items()):
        out[key] = {
            'tok_s':    statistics.median(r['tok_s']    for r in runs),
            'avg_W':    statistics.mean  (r['avg_W']    for r in runs),
            'energy_J': statistics.mean  (r['energy_J'] for r in runs),
            'tokj':     statistics.median(r['tokj']     for r in runs),
            'n':        len(runs),
        }
    return out


def _fallback_avg(tegra_rows, profile_records):
    """
    No timestamps in tegrastats: use global mean VDD_CPU_GPU_CV as constant power.
    Rougher estimate but still useful.
    """
    global_avg_mw = statistics.mean(inst for _, inst, _ in tegra_rows)
    avg_w = global_avg_mw / 1000.0
    print(f"  Fallback: global avg GPU power = {avg_w:.3f} W")

    results = defaultdict(list)
    for r in profile_records:
        if r['run'] == 0:
            continue
        key      = (r['target_prompt_tokens'], r['target_gen_tokens'])
        tok_s    = r['output_tokens'] / r['latency_s']
        energy_j = avg_w * r['latency_s']
        tokj     = r['output_tokens'] / energy_j
        results[key].append({'tok_s': tok_s, 'avg_W': avg_w,
                              'energy_J': energy_j, 'tokj': tokj})

    out = {}
    for key, runs in sorted(results.items()):
        out[key] = {
            'tok_s':    statistics.median(r['tok_s']    for r in runs),
            'avg_W':    statistics.mean  (r['avg_W']    for r in runs),
            'energy_J': statistics.mean  (r['energy_J'] for r in runs),
            'tokj':     statistics.median(r['tokj']     for r in runs),
            'n':        len(runs),
        }
    return out

# ── CSV writer ────────────────────────────────────────────────────────────────
def write_csv(path, mode, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['mode','combo','prompt_tokens','gen_tokens',
                    'tok_s','avg_W','energy_J','tok_J','n'])
        for (pt, gt), v in sorted(data.items()):
            combo = f"pp{pt}+tg{gt}"
            w.writerow([mode, combo, pt, gt,
                        f"{v['tok_s']:.3f}", f"{v['avg_W']:.3f}",
                        f"{v['energy_J']:.3f}", f"{v['tokj']:.3f}", v['n']])
    print(f"  CSV: {path}")

# ── Print table ───────────────────────────────────────────────────────────────
def print_results(mode, data):
    hdr = f"{'Combo':<16}  {'tok/s':>7}  {'avg_W':>7}  {'energy_J':>10}  {'tok/J':>8}  {'n':>4}"
    sep = "=" * len(hdr)
    print(f"\n{sep}")
    print(f" {mode} — Real tok/J from tegrastats")
    print(sep)
    print(hdr)
    print("-" * len(hdr))
    for (pt, gt), v in sorted(data.items()):
        combo = f"pp{pt}+tg{gt}"
        print(f"{combo:<16}  {v['tok_s']:>7.2f}  {v['avg_W']:>7.3f}  "
              f"{v['energy_J']:>10.2f}  {v['tokj']:>8.3f}  {v['n']:>4}")
    print(sep)
    avgs = {k: statistics.mean(v[k] for v in data.values())
            for k in ['tok_s','avg_W','tokj']}
    print(f"  Avg tok/s : {avgs['tok_s']:.2f}")
    print(f"  Avg GPU W : {avgs['avg_W']:.3f}")
    print(f"  Avg tok/J : {avgs['tokj']:.3f}")
    print(sep)

# ── Chart (all 3 modes) ───────────────────────────────────────────────────────
def _save_matplotlib_bar_chart(path, title, labels, values, colors, xlabel, ylabel, horizontal=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    if horizontal:
        y_pos = list(range(len(labels)))
        bars = ax.barh(y_pos, values, color=colors)
        ax.set_yticks(y_pos, labels=labels)
        ax.invert_yaxis()
        for bar, value in zip(bars, values):
            ax.text(value + max(values) * 0.01 if values else 0, bar.get_y() + bar.get_height() / 2,
                    f"{value:.2f}", va="center", ha="left", fontsize=9)
    else:
        x_pos = list(range(len(labels)))
        bars = ax.bar(x_pos, values, color=colors)
        ax.set_xticks(x_pos, labels=labels)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.01 if values else 0,
                    f"{value:.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_chart(all_data, keys, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    combos = [f"pp{pt}+tg{gt}" for pt, gt in keys]
    modes = [m for m in ["15W", "25W", "MAXN"] if m in all_data]
    if not modes:
        return

    # Chart 1: tok/J measured (grouped bars)
    p1 = out_dir / "tokj_comparison.png"
    x_positions = list(range(len(combos)))
    bar_width = 0.8 / max(1, len(modes))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5.4))
    for idx, mode in enumerate(modes):
        vals = [all_data[mode].get(k, {}).get('tokj', 0) for k in keys]
        offsets = [pos + (idx - (len(modes) - 1) / 2) * bar_width for pos in x_positions]
        bars = ax.bar(offsets, vals, width=bar_width, label=mode, color=COLORS[mode], alpha=0.9)
        for bar, value in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{value:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_title("Measured tok/J by Mode")
    ax.set_xlabel("Prompt/Generation Combo")
    ax.set_ylabel("tok/J")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(combos, rotation=20, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(p1, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Chart: {p1}")

    # Chart 2: avg GPU power per mode
    p2 = out_dir / "tokj_avg_power.png"
    avg_watt = [statistics.mean(v['avg_W'] for v in all_data[m].values()) for m in modes]
    _save_matplotlib_bar_chart(p2, "Avg GPU Power Draw by Mode", modes, avg_watt,
                               [COLORS[m] for m in modes], "Power Mode", "Avg W", horizontal=False)
    print(f"  Chart: {p2}")

# ── Single mode run ───────────────────────────────────────────────────────────
def run_single(mode, tegra_path, jsonl_path, out_dir):
    print(f"\nProcessing {mode} ...")
    if not Path(tegra_path).exists():
        print(f"  WARNING: {tegra_path} not found; skipping"); return None
    if not Path(jsonl_path).exists():
        print(f"  WARNING: {jsonl_path} not found; skipping"); return None

    tegra   = parse_tegrastats(tegra_path)
    records = load_jsonl(jsonl_path)
    print(f"  tegrastats rows : {len(tegra)}")
    print(f"  profile runs    : {len(records)}")

    data = calc_tokj(tegra, records)
    print_results(mode, data)
    write_csv(f"{out_dir}/tokj_{mode.lower()}.csv", mode, data)
    return data

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Compute real tok/J from tegrastats + profile jsonl")
    parser.add_argument("--tegra",  help="tegrastats log file")
    parser.add_argument("--jsonl",  help="profile export jsonl")
    parser.add_argument("--mode",   help="mode label (e.g. 25W)", default="25W")
    parser.add_argument("--out",    dest="out_dir", default="results",
                        help="Output folder for CSV/PNG (default: results/)")
    parser.add_argument("--all",    action="store_true",
                        help="Process all 3 modes using default file paths")
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    if args.all:
        all_data = {}
        for mode, (tp, jp) in FILES.items():
            d = run_single(mode, tp, jp, args.out_dir)
            if d:
                all_data[mode] = d

        if len(all_data) >= 2:
            # Merged CSV
            merged_path = f"{args.out_dir}/tokj_all_modes.csv"
            Path(merged_path).parent.mkdir(parents=True, exist_ok=True)
            all_keys = sorted(next(iter(all_data.values())).keys())
            with open(merged_path, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['combo','prompt_tokens','gen_tokens'] +
                           [f'{m}_{k}' for m in ["15W","25W","MAXN"]
                            for k in ['tok_s','avg_W','tok_J']])
                for (pt, gt) in all_keys:
                    combo = f"pp{pt}+tg{gt}"
                    row = [combo, pt, gt]
                    for m in ["15W","25W","MAXN"]:
                        v = all_data.get(m, {}).get((pt,gt), {})
                        row += [f"{v.get('tok_s',0):.3f}",
                                f"{v.get('avg_W',0):.3f}",
                                f"{v.get('tokj',0):.3f}"]
                    w.writerow(row)
            print(f"\n  Merged CSV: {merged_path}")

            # Charts
            print("\nSaving charts ...")
            save_chart(all_data, all_keys, args.out_dir)

        # Sweet-spot summary
        if all_data:
            print(f"\n{'='*55}")
            print(" REAL tok/J SWEET-SPOT SUMMARY")
            print(f"{'='*55}")
            for m in ["15W","25W","MAXN"]:
                if m not in all_data:
                    continue
                d = all_data[m]
                avg_tokj = statistics.mean(v['tokj'] for v in d.values())
                avg_ts   = statistics.mean(v['tok_s'] for v in d.values())
                avg_w    = statistics.mean(v['avg_W'] for v in d.values())
                print(f"  {m:<6}  tok/s={avg_ts:.2f}  GPU_W={avg_w:.3f}  tok/J={avg_tokj:.3f}")
            print(f"{'='*55}\n")

    else:
        # Single mode
        tegra_path = args.tegra or FILES.get(args.mode, (None,None))[0]
        jsonl_path = args.jsonl or FILES.get(args.mode, (None,None))[1]
        if not tegra_path or not jsonl_path:
            print(f"ERROR: Unknown mode '{args.mode}'. Use --tegra and --jsonl explicitly.")
            return
        run_single(args.mode, tegra_path, jsonl_path, args.out_dir)

    print("Done!\n")

if __name__ == "__main__":
    main()