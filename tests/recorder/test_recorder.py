import json
import os
import tempfile
from unittest.mock import patch

import recorder


def _reset_recorder(tmpdir):
    """Reset module state and point to a temp directory."""
    recorder.stop()
    recorder.enable_recording = True
    recorder.output_dir = tmpdir
    recorder.current_date = None
    recorder.generation = 0


class TestRecord:
    def test_writes_valid_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _reset_recorder(tmpdir)
            recorder.record(
                llm_input="prompt text",
                llm_output=[{"type": "speak", "value": "hi"}],
            )
            recorder.stop()

            files = [f for f in os.listdir(tmpdir) if f.endswith(".jsonl")]
            assert len(files) == 1

            with open(os.path.join(tmpdir, files[0])) as f:
                lines = f.readlines()
            assert len(lines) == 1

            data = json.loads(lines[0])
            assert data["llm_input"] == "prompt text"
            assert data["llm_output"] == [{"type": "speak", "value": "hi"}]

    def test_multiple_records_append(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _reset_recorder(tmpdir)
            for _ in range(3):
                recorder.record(llm_input="p", llm_output=[])
            recorder.stop()

            files = [f for f in os.listdir(tmpdir) if f.endswith(".jsonl")]
            with open(os.path.join(tmpdir, files[0])) as f:
                lines = f.readlines()
            assert len(lines) == 3

    def test_record_has_utc_timestamp(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _reset_recorder(tmpdir)
            recorder.record(llm_input="p", llm_output=[])
            recorder.stop()

            files = [f for f in os.listdir(tmpdir) if f.endswith(".jsonl")]
            with open(os.path.join(tmpdir, files[0])) as f:
                data = json.loads(f.readline())
            assert "+00:00" in data["ts"]

    def test_exception_does_not_crash(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _reset_recorder(tmpdir)
            with patch("recorder._write", side_effect=OSError("disk full")):
                recorder.record(llm_input="p", llm_output=[])
            recorder.stop()


class TestDailyRotation:
    def test_date_change_creates_new_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _reset_recorder(tmpdir)

            with patch("recorder.datetime") as mock_dt:
                mock_dt.now.return_value.isoformat.return_value = "2026-01-01T00:00:00+00:00"
                mock_dt.now.return_value.strftime.return_value = "2026-01-01"
                recorder.record(llm_input="p", llm_output=[])

            with patch("recorder.datetime") as mock_dt:
                mock_dt.now.return_value.isoformat.return_value = "2026-01-02T00:00:00+00:00"
                mock_dt.now.return_value.strftime.return_value = "2026-01-02"
                recorder.record(llm_input="p", llm_output=[])

            recorder.stop()

            jsonl_files = sorted(f for f in os.listdir(tmpdir) if f.endswith(".jsonl"))
            assert jsonl_files == ["2026-01-01.jsonl", "2026-01-02.jsonl"]


class TestStop:
    def test_closes_file_handle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _reset_recorder(tmpdir)
            recorder.record(llm_input="p", llm_output=[])
            assert recorder.file is not None

            recorder.stop()
            assert recorder.file is None
            assert recorder.current_date is None

    def test_stop_idempotent(self):
        recorder.stop()
        recorder.stop()

    def test_data_persists_after_stop(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _reset_recorder(tmpdir)
            recorder.record(llm_input="p", llm_output=[])
            recorder.stop()

            files = [f for f in os.listdir(tmpdir) if f.endswith(".jsonl")]
            with open(os.path.join(tmpdir, files[0])) as f:
                assert len(f.readlines()) == 1
