from pathlib import Path
from om1_doctor.report import build_report

def test_build_report_has_host_and_checks():
    rep = build_report(service_name="om1", log_path=None, ports=[8000])
    assert rep["host"]
    assert isinstance(rep["checks"], list)
    assert any(c["name"] == "Local ports" for c in rep["checks"])

def test_build_report_includes_ports():
    rep = build_report(service_name="om1", log_path=None, ports=[1234, 5678])
    assert rep["ports"] == [1234, 5678]
