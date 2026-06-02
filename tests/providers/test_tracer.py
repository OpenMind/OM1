import json
from unittest.mock import patch

import pytest

from providers.tracer import Tracer


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the Tracer singleton before each test."""
    Tracer.reset()  # type: ignore
    yield
    Tracer.reset()  # type: ignore


def test_tracer_disabled_by_default():
    tracer = Tracer()
    assert tracer.enabled is False


def test_enable_sets_enabled_flag(tmp_path):
    tracer = Tracer(output_dir=str(tmp_path / "traces"))
    tracer.enable()
    assert tracer.enabled is True


def test_disable_sets_enabled_flag_false(tmp_path):
    tracer = Tracer(output_dir=str(tmp_path / "traces"))
    tracer.enable()
    tracer.disable()
    assert tracer.enabled is False


def test_enable_creates_output_directory(tmp_path):
    output_dir = tmp_path / "traces"
    tracer = Tracer(output_dir=str(output_dir))
    assert not output_dir.exists()
    tracer.enable()
    assert output_dir.exists()


def test_record_does_nothing_when_disabled(tmp_path):
    tracer = Tracer(output_dir=str(tmp_path / "traces"))
    tracer.gauge("input", [{"role": "assistant", "content": "hello"}])
    assert list((tmp_path / "traces").glob("*.jsonl")) == []


def test_record_writes_jsonl_when_enabled(tmp_path):
    output_dir = tmp_path / "traces"
    tracer = Tracer(output_dir=str(output_dir))
    tracer.enable()

    tracer.gauge("my prompt", [{"role": "assistant", "content": "reply"}])

    files = list(output_dir.glob("*.jsonl"))
    assert len(files) == 1
    lines = files[0].read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["llm_input"] == "my prompt"
    assert data["llm_output"] == [{"role": "assistant", "content": "reply"}]
    assert "ts" in data
    assert "generation" in data


def test_record_writes_generation_number(tmp_path):
    output_dir = tmp_path / "traces"
    tracer = Tracer(output_dir=str(output_dir))
    tracer.enable()
    tracer.set_generation(42)

    tracer.gauge("prompt", [])

    files = list(output_dir.glob("*.jsonl"))
    data = json.loads(files[0].read_text(encoding="utf-8").strip())
    assert data["generation"] == 42


def test_record_appends_multiple_lines(tmp_path):
    output_dir = tmp_path / "traces"
    tracer = Tracer(output_dir=str(output_dir))
    tracer.enable()

    tracer.gauge("prompt1", [])
    tracer.gauge("prompt2", [])

    files = list(output_dir.glob("*.jsonl"))
    assert len(files) == 1
    lines = files[0].read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2


def test_stop_closes_file(tmp_path):
    tracer = Tracer(output_dir=str(tmp_path / "traces"))
    tracer.enable()
    tracer.gauge("prompt", [])

    assert tracer._file is not None
    tracer.stop()
    assert tracer._file is None
    assert tracer._current_date is None


def test_stop_is_idempotent(tmp_path):
    tracer = Tracer(output_dir=str(tmp_path / "traces"))
    tracer.enable()
    tracer.stop()
    tracer.stop()  # should not raise


def test_set_generation(tmp_path):
    tracer = Tracer(output_dir=str(tmp_path / "traces"))
    tracer.set_generation(7)
    assert tracer.generation == 7


def test_record_logs_exception_on_write_failure(tmp_path):
    tracer = Tracer(output_dir=str(tmp_path / "traces"))
    tracer.enable()

    with (
        patch.object(tracer, "_write", side_effect=RuntimeError("disk full")),
        patch("providers.tracer.logging.exception") as mock_log,
    ):
        tracer.gauge("prompt", [])
        mock_log.assert_called_once()


def test_daily_rotation(tmp_path):
    output_dir = tmp_path / "traces"
    tracer = Tracer(output_dir=str(output_dir))
    tracer.enable()

    dates = ["2025-01-01", "2025-01-02"]
    with patch("providers.tracer.datetime") as mock_dt:
        for date in dates:
            mock_dt.now.return_value.strftime.return_value = date
            mock_dt.now.return_value.isoformat.return_value = f"{date}T00:00:00+00:00"
            tracer.gauge(f"prompt on {date}", [])

    files = sorted(output_dir.glob("*.jsonl"))
    assert len(files) == 2
    assert files[0].name == "2025-01-01.jsonl"
    assert files[1].name == "2025-01-02.jsonl"


def test_singleton_returns_same_instance(tmp_path):
    t1 = Tracer(output_dir=str(tmp_path / "traces"))
    t2 = Tracer()
    assert t1 is t2
