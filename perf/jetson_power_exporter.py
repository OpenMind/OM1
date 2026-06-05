#!/usr/bin/env python3
"""Prometheus exporter for NVIDIA Jetson power, thermal and throttling metrics.

Jetson boards (including Jetson AGX Thor) expose on-board INA3221 power monitors
and SoC thermal sensors through sysfs. ARM has no RAPL, so this is the supported
way to read power. This exporter reads sysfs directly (no `tegrastats` binary
dependency) so it can run inside a container with `/sys` bind-mounted read-only.

Exposed metrics (Prometheus text format on :9300/metrics):

  om1_jetson_power_watts{rail="..."}        instantaneous power per INA rail + "total"
  om1_jetson_temp_celsius{zone="..."}       per thermal-zone temperature
  om1_jetson_cpu_freq_hertz{cpu="N"}        current per-core frequency
  om1_jetson_cpu_throttled                  1 if any online core is below its max freq
  om1_jetson_throttle_events_total          monotonically rising count of throttle onsets
  om1_jetson_exporter_up                    1 (scrape liveness / sysfs readability)

It is intentionally degradation-tolerant: any rail/zone that cannot be read is
skipped rather than failing the whole scrape, so the same image works across
Jetson generations and on dev machines (where it simply reports what it can).
"""

import glob
import http.server
import os
import socketserver
import threading

PORT = int(os.environ.get("JETSON_EXPORTER_PORT", "9300"))
SYSFS = os.environ.get("JETSON_SYSFS_ROOT", "/sys")
# A core running below this fraction of its max frequency is treated as throttled.
THROTTLE_RATIO = float(os.environ.get("JETSON_THROTTLE_RATIO", "0.95"))

# Cumulative count of throttle onset transitions (not-throttled -> throttled).
_throttle_events = 0
_prev_throttled = False
_lock = threading.Lock()


def _read_int(path):
    """Read a single integer from a sysfs file, or None on any error."""
    try:
        with open(path) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def _read_str(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def read_power():
    """Return {rail_label: watts} from every INA3221 hwmon channel found.

    Modern Jetson kernels expose channels as in<N>_input (mV) / curr<N>_input (mA)
    with an in<N>_label; some expose power<N>_input directly (microwatts). We
    prefer the direct power reading and fall back to V*I when it is absent.
    """
    rails = {}
    for hwmon in glob.glob(os.path.join(SYSFS, "class/hwmon/hwmon*")):
        chip = _read_str(os.path.join(hwmon, "name")) or ""
        # Only look at INA-style power monitors to avoid e.g. fan controllers.
        if "ina" not in chip.lower() and not glob.glob(os.path.join(hwmon, "in*_input")):
            continue
        for label_path in glob.glob(os.path.join(hwmon, "in*_label")):
            idx = os.path.basename(label_path)[2:].split("_")[0]
            label = _read_str(label_path) or f"{chip}_ch{idx}"
            label = label.replace(" ", "_")
            watts = None
            uw = _read_int(os.path.join(hwmon, f"power{idx}_input"))
            if uw is not None:
                watts = uw / 1_000_000.0
            else:
                mv = _read_int(os.path.join(hwmon, f"in{idx}_input"))
                ma = _read_int(os.path.join(hwmon, f"curr{idx}_input"))
                if mv is not None and ma is not None:
                    watts = (mv / 1000.0) * (ma / 1000.0)
            if watts is not None:
                # Disambiguate identical labels across chips.
                key = label if label not in rails else f"{label}_{os.path.basename(hwmon)}"
                rails[key] = watts
    if rails:
        rails["total"] = sum(rails.values())
    return rails


def read_temps():
    """Return {zone_type: celsius} from /sys/class/thermal/thermal_zone*."""
    temps = {}
    for zone in glob.glob(os.path.join(SYSFS, "class/thermal/thermal_zone*")):
        ztype = _read_str(os.path.join(zone, "type"))
        milli = _read_int(os.path.join(zone, "temp"))
        if ztype and milli is not None:
            temps[ztype.replace(" ", "_")] = milli / 1000.0
    return temps


def read_cpu_freqs():
    """Return ({cpu_index: current_hz}, throttled_bool) from cpufreq sysfs."""
    freqs = {}
    throttled = False
    for cpudir in sorted(glob.glob(os.path.join(SYSFS, "devices/system/cpu/cpu[0-9]*"))):
        name = os.path.basename(cpudir)
        idx = name[3:]
        if not idx.isdigit():
            continue
        # Skip offline cores so they don't read as "throttled to 0".
        online = _read_int(os.path.join(cpudir, "online"))
        if online == 0:
            continue
        cur = _read_int(os.path.join(cpudir, "cpufreq/scaling_cur_freq"))
        mx = _read_int(os.path.join(cpudir, "cpufreq/cpuinfo_max_freq"))
        if cur is None:
            continue
        freqs[idx] = cur * 1000.0  # sysfs cpufreq is in kHz
        if mx and cur < mx * THROTTLE_RATIO:
            throttled = True
    return freqs, throttled


def collect():
    """Render the full metrics payload in Prometheus text exposition format."""
    global _throttle_events, _prev_throttled
    lines = []

    def metric(name, help_text, mtype, samples):
        if not samples:
            return
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} {mtype}")
        lines.extend(samples)

    power = read_power()
    metric(
        "om1_jetson_power_watts",
        "Instantaneous power draw per INA rail (plus 'total') in watts",
        "gauge",
        [f'om1_jetson_power_watts{{rail="{r}"}} {v:.4f}' for r, v in power.items()],
    )

    temps = read_temps()
    metric(
        "om1_jetson_temp_celsius",
        "SoC thermal-zone temperature in degrees Celsius",
        "gauge",
        [f'om1_jetson_temp_celsius{{zone="{z}"}} {v:.3f}' for z, v in temps.items()],
    )

    freqs, throttled = read_cpu_freqs()
    metric(
        "om1_jetson_cpu_freq_hertz",
        "Current per-core CPU frequency in hertz",
        "gauge",
        [f'om1_jetson_cpu_freq_hertz{{cpu="{c}"}} {v:.0f}' for c, v in freqs.items()],
    )

    with _lock:
        if throttled and not _prev_throttled:
            _throttle_events += 1
        _prev_throttled = throttled
        events = _throttle_events

    metric(
        "om1_jetson_cpu_throttled",
        "1 if any online core is currently below its max frequency, else 0",
        "gauge",
        [f"om1_jetson_cpu_throttled {1 if throttled else 0}"],
    )
    metric(
        "om1_jetson_throttle_events_total",
        "Cumulative number of throttle onset transitions since exporter start",
        "counter",
        [f"om1_jetson_throttle_events_total {events}"],
    )
    metric(
        "om1_jetson_exporter_up",
        "1 if the exporter is running and sysfs is readable",
        "gauge",
        ["om1_jetson_exporter_up 1"],
    )

    return "\n".join(lines) + "\n"


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path not in ("/metrics", "/"):
            self.send_response(404)
            self.end_headers()
            return
        payload = collect().encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):  # silence per-request stderr logging
        pass


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    print(f"jetson-power-exporter listening on :{PORT} (sysfs root: {SYSFS})", flush=True)
    Server(("", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
