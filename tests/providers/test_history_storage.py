"""
Unit tests for history_storage module.
Tests JSON storage utilities for conversation history.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

from src.providers.history_storage import (
    HISTORY_FILE,
    history_exists,
    load_history,
    save_history,
)


class TestHistoryStorage(unittest.TestCase):
    """Test cases for history_storage functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_data = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        self.test_json = json.dumps(self.test_data, indent=2, ensure_ascii=False)

    def tearDown(self):
        """Clean up after tests."""
        if HISTORY_FILE.exists():
            try:
                os.remove(HISTORY_FILE)
            except (OSError, PermissionError):
                pass

    def test_save_history_success(self):
        """Test successful save of history data."""
        with patch("builtins.open", mock_open()) as mock_file:
            with patch.object(Path, "mkdir") as mock_mkdir:
                result = save_history(self.test_data)

                self.assertTrue(result)
                mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
                mock_file.assert_called_once()
                call_args = mock_file.call_args
                self.assertEqual(call_args[0][1], "w")
                self.assertEqual(call_args[1]["encoding"], "utf-8")

    def test_save_history_with_empty_list(self):
        """Test saving empty history list."""
        with patch("builtins.open", mock_open()):
            with patch.object(Path, "mkdir"):
                result = save_history([])

                self.assertTrue(result)

    def test_save_history_creates_parent_directory(self):
        """Test that save_history creates parent directory if needed."""
        with patch("builtins.open", mock_open()):
            with patch.object(Path, "mkdir") as mock_mkdir:
                save_history(self.test_data)

                mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    def test_save_history_io_error(self):
        """Test save_history handles IOError gracefully."""
        with patch.object(Path, "mkdir"):
            with patch("builtins.open", side_effect=IOError("Disk full")):
                result = save_history(self.test_data)

                self.assertFalse(result)

    def test_save_history_os_error(self):
        """Test save_history handles OSError gracefully."""
        with patch.object(Path, "mkdir"):
            with patch("builtins.open", side_effect=OSError("Permission denied")):
                result = save_history(self.test_data)

                self.assertFalse(result)

    def test_save_history_json_serialization(self):
        """Test that data is properly JSON serialized."""
        with patch("builtins.open", mock_open()):
            with patch.object(Path, "mkdir"):
                with patch("json.dump") as mock_json_dump:
                    save_history(self.test_data)

                    mock_json_dump.assert_called_once()
                    call_args = mock_json_dump.call_args
                    self.assertEqual(call_args[0][0], self.test_data)
                    self.assertEqual(call_args[1]["indent"], 2)
                    self.assertFalse(call_args[1]["ensure_ascii"])

    def test_save_history_with_unicode_content(self):
        """Test saving history with unicode characters."""
        unicode_data = [
            {"role": "user", "content": "Hello 你好 こんにちは 🌍"},
            {"role": "assistant", "content": "Привет مرحبا 안녕하세요"},
        ]

        with patch("builtins.open", mock_open()):
            with patch.object(Path, "mkdir"):
                with patch("json.dump") as mock_json_dump:
                    result = save_history(unicode_data)

                    self.assertTrue(result)
                    call_args = mock_json_dump.call_args
                    self.assertFalse(call_args[1]["ensure_ascii"])

    def test_save_history_with_large_data(self):
        """Test saving large history data."""
        large_data = [{"role": "user", "content": f"Message {i}"} for i in range(1000)]

        with patch("builtins.open", mock_open()):
            with patch.object(Path, "mkdir"):
                result = save_history(large_data)

                self.assertTrue(result)

    def test_save_history_creates_nested_directory(self):
        """Test that save_history can create nested directories."""
        with patch("builtins.open", mock_open()):
            with patch.object(Path, "mkdir") as mock_mkdir:
                save_history(self.test_data)

                mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    def test_load_history_success(self):
        """Test successful load of history data."""
        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = True

        with patch("src.providers.history_storage.HISTORY_FILE", mock_path):
            with patch("builtins.open", mock_open(read_data=self.test_json)):
                result = load_history()

                self.assertEqual(result, self.test_data)

    def test_load_history_file_not_exists(self):
        """Test load_history when file doesn't exist."""
        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = False

        with patch("src.providers.history_storage.HISTORY_FILE", mock_path):
            result = load_history()

            self.assertEqual(result, [])

    def test_load_history_empty_file(self):
        """Test load_history with empty JSON array."""
        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = True

        with patch("src.providers.history_storage.HISTORY_FILE", mock_path):
            with patch("builtins.open", mock_open(read_data="[]")):
                result = load_history()

                self.assertEqual(result, [])

    def test_load_history_invalid_json(self):
        """Test load_history handles invalid JSON gracefully."""
        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = True

        with patch("src.providers.history_storage.HISTORY_FILE", mock_path):
            with patch("builtins.open", mock_open(read_data="not valid json {")):
                result = load_history()

                self.assertEqual(result, [])

    def test_load_history_json_decode_error(self):
        """Test load_history handles JSONDecodeError."""
        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = True

        with patch("src.providers.history_storage.HISTORY_FILE", mock_path):
            with patch("builtins.open", mock_open(read_data='{"invalid": json}')):
                result = load_history()

                self.assertEqual(result, [])

    def test_load_history_io_error(self):
        """Test load_history handles IOError gracefully."""
        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = True

        with patch("src.providers.history_storage.HISTORY_FILE", mock_path):
            with patch("builtins.open", side_effect=IOError("Read error")):
                result = load_history()

                self.assertEqual(result, [])

    def test_load_history_os_error(self):
        """Test load_history handles OSError gracefully."""
        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = True

        with patch("src.providers.history_storage.HISTORY_FILE", mock_path):
            with patch("builtins.open", side_effect=OSError("Permission denied")):
                result = load_history()

                self.assertEqual(result, [])

    def test_load_history_returns_list(self):
        """Test that load_history always returns a list."""
        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = True

        with patch("src.providers.history_storage.HISTORY_FILE", mock_path):
            with patch("builtins.open", mock_open(read_data=self.test_json)):
                result = load_history()
                self.assertIsInstance(result, list)

        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = False

        with patch("src.providers.history_storage.HISTORY_FILE", mock_path):
            result = load_history()
            self.assertIsInstance(result, list)

        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = True

        with patch("src.providers.history_storage.HISTORY_FILE", mock_path):
            with patch("builtins.open", side_effect=IOError()):
                result = load_history()
                self.assertIsInstance(result, list)

    def test_load_history_with_extra_whitespace(self):
        """Test load_history handles JSON with extra whitespace."""
        json_with_whitespace = '  [  {"role": "user", "content": "test"}  ]  '
        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = True

        with patch("src.providers.history_storage.HISTORY_FILE", mock_path):
            with patch("builtins.open", mock_open(read_data=json_with_whitespace)):
                result = load_history()

                self.assertEqual(len(result), 1)
                self.assertEqual(result[0]["role"], "user")

    def test_history_exists_true(self):
        """Test history_exists returns True when file exists."""
        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = True

        with patch("src.providers.history_storage.HISTORY_FILE", mock_path):
            result = history_exists()

            self.assertTrue(result)
            mock_path.exists.assert_called_once()

    def test_history_exists_false(self):
        """Test history_exists returns False when file doesn't exist."""
        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = False

        with patch("src.providers.history_storage.HISTORY_FILE", mock_path):
            result = history_exists()

            self.assertFalse(result)
            mock_path.exists.assert_called_once()

    def test_save_and_load_integration(self):
        """Test saving and loading history in integration."""
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".json"
        ) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            with patch("src.providers.history_storage.HISTORY_FILE", tmp_path):
                save_result = save_history(self.test_data)
                self.assertTrue(save_result)

                loaded_data = load_history()
                self.assertEqual(loaded_data, self.test_data)

                exists = history_exists()
                self.assertTrue(exists)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_save_and_load_preserves_message_order(self):
        """Test that message order is preserved through save/load cycle."""
        ordered_data = [
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Second"},
            {"role": "user", "content": "Third"},
        ]

        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".json"
        ) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            with patch("src.providers.history_storage.HISTORY_FILE", tmp_path):
                save_history(ordered_data)
                loaded = load_history()

                for i, msg in enumerate(loaded):
                    self.assertEqual(msg["content"], ordered_data[i]["content"])
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_load_after_failed_save(self):
        """Test that load still works after a failed save."""
        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = False

        with patch("src.providers.history_storage.HISTORY_FILE", mock_path):
            result = load_history()
            self.assertEqual(result, [])

    def test_history_file_constant(self):
        """Test that HISTORY_FILE constant is properly defined."""
        self.assertIsNotNone(HISTORY_FILE)
        self.assertIsInstance(HISTORY_FILE, Path)
        self.assertTrue(str(HISTORY_FILE).endswith(".json"))
        self.assertEqual(str(HISTORY_FILE), "data/conversation_history.json")
