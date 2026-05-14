import json
import os
import tempfile
from unittest.mock import patch

from runtime.recorder import RuntimeRecorder


class TestInit:
    def test_creates_output_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "sub", "recordings")
            RuntimeRecorder(output_dir=out)
            assert os.path.isdir(out)

    def test_default_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rec = RuntimeRecorder(output_dir=tmpdir)
            assert rec._current_date is None
            assert rec._file is None


class TestRecordTick:
    def _make_recorder(self, tmpdir):
        return RuntimeRecorder(output_dir=tmpdir)

    def test_writes_valid_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rec = self._make_recorder(tmpdir)
            rec.record_tick(
                tick_num=0,
                mode="patrol",
                asr_input="hello",
                llm_input="prompt text",
                llm_output=[{"type": "speak", "value": "hi"}],
                actions_executed=[{"type": "speak", "value": "hi"}],
            )
            rec.stop()

            files = [f for f in os.listdir(tmpdir) if f.endswith(".jsonl")]
            assert len(files) == 1

            with open(os.path.join(tmpdir, files[0])) as f:
                lines = f.readlines()
            assert len(lines) == 1

            record = json.loads(lines[0])
            assert record["tick"] == 0
            assert record["mode"] == "patrol"
            assert record["asr_input"] == "hello"
            assert record["llm_input"] == "prompt text"
            assert record["llm_output"] == [{"type": "speak", "value": "hi"}]

    def test_multiple_ticks_append(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rec = self._make_recorder(tmpdir)
            for i in range(3):
                rec.record_tick(
                    tick_num=i,
                    mode="chat",
                    asr_input=None,
                    llm_input="p",
                    llm_output=[],
                    actions_executed=[],
                )
            rec.stop()

            files = [f for f in os.listdir(tmpdir) if f.endswith(".jsonl")]
            with open(os.path.join(tmpdir, files[0])) as f:
                lines = f.readlines()
            assert len(lines) == 3

            for i, line in enumerate(lines):
                assert json.loads(line)["tick"] == i

    def test_asr_input_none_allowed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rec = self._make_recorder(tmpdir)
            rec.record_tick(
                tick_num=0,
                mode="idle",
                asr_input=None,
                llm_input="p",
                llm_output=[],
                actions_executed=[],
            )
            rec.stop()

            files = [f for f in os.listdir(tmpdir) if f.endswith(".jsonl")]
            with open(os.path.join(tmpdir, files[0])) as f:
                record = json.loads(f.readline())
            assert record["asr_input"] is None

    def test_record_has_utc_timestamp(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rec = self._make_recorder(tmpdir)
            rec.record_tick(
                tick_num=0,
                mode="test",
                asr_input=None,
                llm_input="p",
                llm_output=[],
                actions_executed=[],
            )
            rec.stop()

            files = [f for f in os.listdir(tmpdir) if f.endswith(".jsonl")]
            with open(os.path.join(tmpdir, files[0])) as f:
                record = json.loads(f.readline())
            assert "+00:00" in record["ts"]

    def test_exception_does_not_crash(self):
        """Recording failure must not raise."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rec = self._make_recorder(tmpdir)
            with patch.object(rec, "_write_record", side_effect=OSError("disk full")):
                rec.record_tick(
                    tick_num=0,
                    mode="test",
                    asr_input=None,
                    llm_input="p",
                    llm_output=[],
                    actions_executed=[],
                )


class TestDailyRotation:
    def test_date_change_creates_new_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rec = RuntimeRecorder(output_dir=tmpdir)

            # Write first record with mocked date
            with patch("runtime.recorder.datetime") as mock_dt:
                mock_dt.now.return_value.isoformat.return_value = "2026-01-01T00:00:00+00:00"
                mock_dt.now.return_value.strftime.return_value = "2026-01-01"
                rec.record_tick(0, "m", None, "p", [], [])

            # Write second record with different date
            with patch("runtime.recorder.datetime") as mock_dt:
                mock_dt.now.return_value.isoformat.return_value = "2026-01-02T00:00:00+00:00"
                mock_dt.now.return_value.strftime.return_value = "2026-01-02"
                rec.record_tick(1, "m", None, "p", [], [])

            rec.stop()

            jsonl_files = sorted(f for f in os.listdir(tmpdir) if f.endswith(".jsonl"))
            assert jsonl_files == ["2026-01-01.jsonl", "2026-01-02.jsonl"]


class TestStop:
    def test_closes_file_handle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rec = RuntimeRecorder(output_dir=tmpdir)
            rec.record_tick(0, "m", None, "p", [], [])
            assert rec._file is not None

            rec.stop()
            assert rec._file is None
            assert rec._current_date is None

    def test_stop_idempotent(self):
        """Calling stop() multiple times should not raise."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rec = RuntimeRecorder(output_dir=tmpdir)
            rec.stop()
            rec.stop()

    def test_data_persists_after_stop(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rec = RuntimeRecorder(output_dir=tmpdir)
            rec.record_tick(0, "m", None, "p", [], [])
            rec.stop()

            files = [f for f in os.listdir(tmpdir) if f.endswith(".jsonl")]
            with open(os.path.join(tmpdir, files[0])) as f:
                assert len(f.readlines()) == 1


class TestAppendAfterRestart:
    def test_restart_appends_to_same_file(self):
        """Simulates stopping and restarting the recorder within the same day."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # First session
            rec1 = RuntimeRecorder(output_dir=tmpdir)
            rec1.record_tick(0, "m", None, "p", [], [])
            rec1.stop()

            # Second session
            rec2 = RuntimeRecorder(output_dir=tmpdir)
            rec2.record_tick(0, "m", None, "p", [], [])
            rec2.stop()

            files = [f for f in os.listdir(tmpdir) if f.endswith(".jsonl")]
            assert len(files) == 1

            with open(os.path.join(tmpdir, files[0])) as f:
                lines = f.readlines()
            assert len(lines) == 2
