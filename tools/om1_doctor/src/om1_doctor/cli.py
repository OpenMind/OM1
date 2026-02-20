from __future__ import annotations

import json
from pathlib import Path
import socket

import typer
from rich.console import Console
from rich.table import Table

from .report import build_report, report_to_markdown

app = typer.Typer(add_completion=False)
console = Console()


def _parse_ports(ports: str) -> list[int]:
    # Accepts "8000,8545" or "8000 8545"
    parts = [p.strip() for p in ports.replace(" ", ",").split(",") if p.strip()]
    out: list[int] = []
    for p in parts:
        try:
            out.append(int(p))
        except ValueError:
            raise typer.BadParameter(f"Invalid port: {p}")
    if not out:
        raise typer.BadParameter("Ports list cannot be empty")
    return out


@app.command()
def doctor(
    service: str = typer.Option("om1", help="systemd service name to check (if exists)"),
    log_path: Path | None = typer.Option(None, help="Optional log file to scan"),
    ports: str = typer.Option("8000,8545", help="Comma/space separated ports to scan (e.g. 8000,8545)"),
    json_out: Path | None = typer.Option(None, help="Write raw JSON report to file"),
):
    """Run diagnostics and print a summary."""
    rep = build_report(service_name=service, log_path=log_path, ports=_parse_ports(ports))

    table = Table(title="OM1 Doctor Summary")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Details")

    for item in rep["checks"]:
        table.add_row(item["name"], item["status"], item.get("details", ""))

    console.print(table)

    if json_out:
        json_out.write_text(json.dumps(rep, indent=2), encoding="utf-8")
        console.print(f"[green]Wrote JSON report:[/green] {json_out}")


@app.command()
def report(
    service: str = typer.Option("om1", help="systemd service name to check (if exists)"),
    log_path: Path | None = typer.Option(None, help="Optional log file to scan"),
    ports: str = typer.Option("8000,8545", help="Comma/space separated ports to scan (e.g. 8000,8545)"),
    md_out: Path | None = typer.Option(None, help="Write Markdown report to file"),
):
    """Generate a Markdown report for GitHub/Discord."""
    rep = build_report(service_name=service, log_path=log_path, ports=_parse_ports(ports))
    md = report_to_markdown(rep)

    if md_out:
        md_out.write_text(md, encoding="utf-8")
        console.print(f"[green]Wrote Markdown report:[/green] {md_out}")
    else:
        console.print(md)


@app.command()
def service_template(
    workdir: Path = typer.Option(Path("C:\\OM1"), help="Working directory"),
    user: str = typer.Option("om1", help="Windows username or Linux user (template only)"),
):
    """Print a systemd unit template (mainly for Linux VPS)."""
    unit = f"""[Unit]
Description=OM1 Node
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User={user}
WorkingDirectory={workdir}
ExecStart={workdir}/.venv/bin/python {workdir}/src/run.py
Restart=on-failure
RestartSec=3
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
"""
    console.print(unit)


if __name__ == "__main__":
    app()
