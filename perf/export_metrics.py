#!/usr/bin/env python3
"""Export OM1 perf-validation metrics from Prometheus into AI-friendly files.

For a given time window (one benchmark run) it writes, under <out>/<label>/:
  - summary.csv / summary.md  : per-series min/mean/p50/p95/max + unit  (PRIMARY — feed this to the AI)
  - series/<metric>.csv       : downsampled raw time series (long format)  (optional, for spotting spikes)
  - manifest.json             : run metadata (label, window, step, prom url, notes)

Run it ONCE PER RUN (once for the Python window, once for the Go window),
labeling each, then hand the two summaries to the AI to compare.

Usage:
  # export the last 30 minutes, label the run "go"
  python3 export_metrics.py --label go --last 30m

  # or an explicit window (RFC3339 or unix seconds)
  python3 export_metrics.py --label python --start 2026-06-05T17:00:00Z --end 2026-06-05T17:30:00Z

Env/flags:
  --prom   Prometheus base URL (default http://localhost:9091)
  --step   sample step for raw series (default 5s)
"""
import argparse
import json
import math
import os
import statistics
import time
import urllib.parse
import urllib.request

# Curated metrics: name -> (promql, unit). Both Go and Python variants are
# included; the ones that don't exist on a given run just come back empty.
METRICS = {
    "cpu_cores_container":   ('rate(container_cpu_usage_seconds_total{name="om1"}[1m])', "cores"),
    "cpu_cores_process":     ('rate(process_cpu_seconds_total{job="om1"}[1m])', "cores"),
    "rss_container_bytes":   ('container_memory_rss{name="om1"}', "bytes"),
    "workingset_bytes":      ('container_memory_working_set_bytes{name="om1"}', "bytes"),
    "rss_process_bytes":     ('process_resident_memory_bytes{job="om1"}', "bytes"),
    "heap_alloc_bytes_per_s_go": ('rate(go_memstats_alloc_bytes_total{job="om1"}[1m])', "bytes/s"),
    "alloc_blocks_per_s_py": ('rate(om1_python_allocated_blocks{job="om1"}[1m])', "blocks/s"),
    "traced_heap_bytes_py":  ('om1_python_traced_heap_bytes', "bytes"),
    "goroutines_go":         ('go_goroutines{job="om1"}', "count"),
    "gc_per_s_go":           ('rate(go_gc_duration_seconds_count{job="om1"}[1m])', "1/s"),
    "power_watts":           ('om1_jetson_power_watts', "watt"),
    "temp_celsius":          ('om1_jetson_temp_celsius', "celsius"),
    "throttled":             ('om1_jetson_cpu_throttled', "0/1"),
    "throttle_events":       ('om1_jetson_throttle_events_total', "count"),
    "startup_seconds":       ('om1_startup_duration_seconds', "s"),
    "llm_p50_s":             ('histogram_quantile(0.50, sum by (le) (rate(om1_llm_latency_seconds_bucket[1m])))', "s"),
    "llm_p95_s":             ('histogram_quantile(0.95, sum by (le) (rate(om1_llm_latency_seconds_bucket[1m])))', "s"),
    "asr_p95_s":             ('histogram_quantile(0.95, sum by (le) (rate(om1_asr_latency_seconds_bucket[1m])))', "s"),
    "http_upstream_ttfb_p95_s": ('histogram_quantile(0.95, sum by (le) (rate(om1_http_upstream_ttfb_seconds_bucket[1m])))', "s"),
    "http_proxy_total_p95_s":   ('histogram_quantile(0.95, sum by (le) (rate(om1_http_proxy_total_seconds_bucket[1m])))', "s"),
    "llm_throughput_per_s":  ('rate(om1_llm_latency_seconds_count[1m])', "1/s"),
}


def parse_t(s):
    """RFC3339 or unix-seconds string -> unix seconds (float)."""
    try:
        return float(s)
    except ValueError:
        from datetime import datetime
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def parse_dur(s):
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    return float(s[:-1]) * units[s[-1]] if s[-1] in units else float(s)


def query_range(prom, q, start, end, step):
    qs = urllib.parse.urlencode({"query": q, "start": start, "end": end, "step": step})
    with urllib.request.urlopen(f"{prom}/api/v1/query_range?{qs}", timeout=30) as r:
        return json.load(r)["data"]["result"]


def series_label(metric_labels):
    drop = {"__name__", "job", "instance"}
    items = {k: v for k, v in metric_labels.items() if k not in drop}
    return ",".join(f"{k}={v}" for k, v in sorted(items.items())) or "-"


def stats(values):
    vals = [v for v in values if not math.isnan(v)]
    if not vals:
        return None
    s = sorted(vals)
    def pct(p):
        return s[max(0, math.ceil(p / 100 * len(s)) - 1)]
    return {"count": len(s), "min": min(s), "mean": statistics.fmean(s),
            "p50": pct(50), "p95": pct(95), "max": max(s)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="run label, e.g. python or go")
    ap.add_argument("--prom", default="http://localhost:9091")
    ap.add_argument("--start"); ap.add_argument("--end")
    ap.add_argument("--last", help="export the last N (e.g. 30m) instead of start/end")
    ap.add_argument("--step", default="5s")
    ap.add_argument("--out", default="perf/export")
    ap.add_argument("--notes", default="", help="freeform: workload, hardware, anything the AI should know")
    args = ap.parse_args()

    now = time.time()
    if args.last:
        start, end = now - parse_dur(args.last), now
    else:
        start, end = parse_t(args.start), parse_t(args.end)

    outdir = os.path.join(args.out, args.label)
    os.makedirs(os.path.join(outdir, "series"), exist_ok=True)

    summary_rows = []
    for name, (q, unit) in METRICS.items():
        try:
            result = query_range(args.prom, q, start, end, args.step)
        except Exception as e:
            print(f"  ! {name}: query failed: {e}")
            continue
        # write raw long-format CSV
        with open(os.path.join(outdir, "series", f"{name}.csv"), "w") as f:
            f.write("timestamp,series,value\n")
            for s in result:
                lab = series_label(s.get("metric", {}))
                for ts, val in s.get("values", []):
                    f.write(f"{ts},{lab},{val}\n")
        # per-series summary
        for s in result:
            lab = series_label(s.get("metric", {}))
            vals = [float(v) for _, v in s.get("values", []) if v not in ("NaN", "+Inf", "-Inf")]
            st = stats(vals)
            if st:
                summary_rows.append((name, lab, unit, st))
        print(f"  - {name}: {len(result)} series")

    # summary.csv
    with open(os.path.join(outdir, "summary.csv"), "w") as f:
        f.write("metric,series,unit,count,min,mean,p50,p95,max\n")
        for name, lab, unit, st in summary_rows:
            f.write(f'{name},"{lab}",{unit},{st["count"]},{st["min"]:.6g},{st["mean"]:.6g},'
                    f'{st["p50"]:.6g},{st["p95"]:.6g},{st["max"]:.6g}\n')
    # summary.md (the AI-friendly one)
    with open(os.path.join(outdir, "summary.md"), "w") as f:
        f.write(f"# OM1 perf metrics — run `{args.label}`\n\n")
        f.write(f"- window: {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(start))} → "
                f"{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(end))} UTC ({(end-start):.0f}s), step {args.step}\n")
        f.write(f"- prometheus: {args.prom}\n- notes: {args.notes or '(none)'}\n\n")
        f.write("| metric | series | unit | mean | p50 | p95 | min | max |\n|---|---|---|---|---|---|---|---|\n")
        for name, lab, unit, st in summary_rows:
            f.write(f'| {name} | {lab} | {unit} | {st["mean"]:.4g} | {st["p50"]:.4g} | '
                    f'{st["p95"]:.4g} | {st["min"]:.4g} | {st["max"]:.4g} |\n')
    # manifest
    json.dump({"label": args.label, "start": start, "end": end, "step": args.step,
               "prom": args.prom, "notes": args.notes, "metrics": list(METRICS)},
              open(os.path.join(outdir, "manifest.json"), "w"), indent=2)

    print(f"\nwrote {outdir}/  (summary.md, summary.csv, series/*.csv, manifest.json)")


if __name__ == "__main__":
    main()
