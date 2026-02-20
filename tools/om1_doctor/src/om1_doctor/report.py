from __future__ import annotations

import platform
import shutil
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import psutil

from .checks.signatures import match_signatures


def _run(cmd: list[str]) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, check=False)
        out = (p.stdout or "") + (p.stderr or "")
        return p.returncode, out.strip()
    except Exception as e:
        return 1, str(e)


def _port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.2)
        return s.connect_ex(("127.0.0.1", port)) == 0


def build_report(service_name: str, log_path: Path | None, ports: list[int] | None = None) -> dict:
    checks: list[dict] = []
    now = datetime.now(timezone.utc).isoformat()
    ports = ports or [8000, 8545]

    # Disk usage
    try:
        du = shutil.disk_usage(Path.cwd().anchor)
        checks.append(
            {
                "name": "Disk free",
                "status": "OK" if du.free > 5 * 1024**3 else "WARN",
                "details": f"{du.free/1024**3:.1f} GB free on drive",
            }
        )
    except Exception as e:
        checks.append(
            {
                "name": "Disk free",
                "status": "INFO",
                "details": f"Unable to read disk usage: {e}",
            }
        )

    # RAM
    try:
        ram = psutil.virtual_memory()
        checks.append(
            {
                "name": "RAM available",
                "status": "OK" if ram.available > 1 * 1024**3 else "WARN",
                "details": f"{ram.available/1024**3:.1f} GB available",
            }
        )
    except Exception as e:
        checks.append(
            {
                "name": "RAM available",
                "status": "INFO",
                "details": f"Unable to read memory stats: {e}",
            }
        )

    # Python version
    code, out = _run(["python", "--version"])
    checks.append(
        {
            "name": "Python",
            "status": "OK" if code == 0 else "FAIL",
            "details": out,
        }
    )

    # uv version
    code, out = _run(["uv", "--version"])
    checks.append(
        {
            "name": "uv",
            "status": "OK" if code == 0 else "WARN",
            "details": out or "uv not found",
        }
    )

    # systemd (Linux only)
    os_name = platform.system().lower()
    if os_name == "linux":
        code, out = _run(["systemctl", "is-active", service_name])
        if code == 0 and out.strip() == "active":
            status = "OK"
            details = f"{service_name} is active"
        else:
            status = "WARN"
            details = f"{service_name} not active: {out or 'unknown'}"
        checks.append(
            {
                "name": f"systemd:{service_name}",
                "status": status,
                "details": details,
            }
        )
    else:
        checks.append(
            {
                "name": f"systemd:{service_name}",
                "status": "INFO",
                "details": f"systemd check not available on {platform.system()} (expected)",
            }
        )

    # Ports
    port_details = ", ".join(
        [f"{p}:{'open' if _port_open(p) else 'closed'}" for p in ports]
    )
    checks.append(
        {
            "name": "Local ports",
            "status": "OK",
            "details": port_details,
        }
    )

    # Log signatures
    findings = []
    if log_path and log_path.exists():
        try:
            tail = log_path.read_text(errors="ignore")[-20000:]
            findings = match_signatures(tail)
        except Exception as e:
            findings = [
                {
                    "id": "LOG_READ_ERROR",
                    "title": "Log read error",
                    "hint": str(e),
                }
            ]

    return {
        "generated_at": now,
        "host": socket.gethostname(),
        "service_name": service_name,
        "log_path": str(log_path) if log_path else None,
        "ports": ports,
        "checks": checks,
        "findings": findings,
    }


def report_to_markdown(rep: dict) -> str:
    lines: list[str] = []
    lines.append("# OM1 Doctor Report")
    lines.append(f"- Generated: `{rep['generated_at']}`")
    lines.append(f"- Host: `{rep['host']}`")
    lines.append(f"- Service: `{rep['service_name']}`")

    if rep.get("log_path"):
        lines.append(f"- Log: `{rep['log_path']}`")

    if rep.get("ports"):
        lines.append(f"- Ports: `{','.join(str(p) for p in rep['ports'])}`")

    lines.append("")
    lines.append("## Checks")

    for c in rep["checks"]:
        lines.append(
            f"- **{c['name']}**: `{c['status']}` — {c.get('details','')}"
        )

    lines.append("")
    lines.append("## Findings (log signatures)")

    if rep["findings"]:
        for f in rep["findings"]:
            lines.append(
                f"- `{f['id']}`: {f['title']} — {f['hint']}"
            )
    else:
        lines.append("- (none)")

    return "\n".join(lines)
