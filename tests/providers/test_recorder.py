import json
from unittest.mock import patch

import pytest

from providers.recorder import Recorder


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the Recorder singleton before each test."""
    Recorder.reset()  # type: ignore
    yield
    Recorder.reset()  # type: ignore


def test_recorder_disabled_by_default():
    recorder = Recorder()
    assert recorder.enabled is False


def test_enable_sets_enabled_flag(tmp_path):
    recorder = Recorder(output_dir=str(tmp_path / "recordings"))
    recorder.enable()
    assert recorder.enabled is True


def test_disable_sets_enabled_flag_false(tmp_path):
    recorder = Recorder(output_dir=str(tmp_path / "recordings"))
    recorder.enable()
    recorder.disable()
    assert recorder.enabled is False


def test_enable_creates_output_directory(tmp_path):
    output_dir = tmp_path / "recordings"
    recorder = Recorder(output_dir=str(output_dir))
    assert not output_dir.exists()
    recorder.enable()
    assert output_dir.exists()


def test_record_does_nothing_when_disabled(tmp_path):
    recorder = Recorder(output_dir=str(tmp_path / "recordings"))
    recorder.record("input", [{"role": "assistant", "content": "hello"}])
    assert list((tmp_path / "recordings").glob("*.jsonl")) == []


def test_record_writes_jsonl_when_enabled(tmp_path):
    output_dir = tmp_path / "recordings"
    recorder = Recorder(output_dir=str(output_dir))
    recorder.enable()

    recorder.record("my prompt", [{"role": "assistant", "content": "reply"}])

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
    output_dir = tmp_path / "recordings"
    recorder = Recorder(output_dir=str(output_dir))
    recorder.enable()
    recorder.set_generation(42)

    recorder.record("prompt", [])

    files = list(output_dir.glob("*.jsonl"))
    data = json.loads(files[0].read_text(encoding="utf-8").strip())
    assert data["generation"] == 42


def test_record_appends_multiple_lines(tmp_path):
    output_dir = tmp_path / "recordings"
    recorder = Recorder(output_dir=str(output_dir))
    recorder.enable()

    recorder.record("prompt1", [])
    recorder.record("prompt2", [])

    files = list(output_dir.glob("*.jsonl"))
    assert len(files) == 1
    lines = files[0].read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2


def test_stop_closes_file(tmp_path):
    recorder = Recorder(output_dir=str(tmp_path / "recordings"))
    recorder.enable()
    recorder.record("prompt", [])

    assert recorder._file is not None
    recorder.stop()
    assert recorder._file is None
    assert recorder._current_date is None


def test_stop_is_idempotent(tmp_path):
    recorder = Recorder(output_dir=str(tmp_path / "recordings"))
    recorder.enable()
    recorder.stop()
    recorder.stop()  # should not raise


def test_set_generation(tmp_path):
    recorder = Recorder(output_dir=str(tmp_path / "recordings"))
    recorder.set_generation(7)
    assert recorder.generation == 7


def test_record_logs_exception_on_write_failure(tmp_path):
    recorder = Recorder(output_dir=str(tmp_path / "recordings"))
    recorder.enable()

    with (
        patch.object(recorder, "_write", side_effect=RuntimeError("disk full")),
        patch("providers.recorder.logging.exception") as mock_log,
    ):
        recorder.record("prompt", [])
        mock_log.assert_called_once()


def test_daily_rotation(tmp_path):
    output_dir = tmp_path / "recordings"
    recorder = Recorder(output_dir=str(output_dir))
    recorder.enable()

    dates = ["2025-01-01", "2025-01-02"]
    with patch("providers.recorder.datetime") as mock_dt:
        for date in dates:
            mock_dt.now.return_value.strftime.return_value = date
            mock_dt.now.return_value.isoformat.return_value = f"{date}T00:00:00+00:00"
            recorder.record(f"prompt on {date}", [])

    files = sorted(output_dir.glob("*.jsonl"))
    assert len(files) == 2
    assert files[0].name == "2025-01-01.jsonl"
    assert files[1].name == "2025-01-02.jsonl"


def test_singleton_returns_same_instance(tmp_path):
    r1 = Recorder(output_dir=str(tmp_path / "recordings"))
    r2 = Recorder()
    assert r1 is r2
