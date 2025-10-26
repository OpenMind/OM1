from unittest.mock import Mock, mock_open, patch
import pytest
from inputs import find_module_with_class, load_input
from inputs.base import Sensor


class MockInput(Sensor):
    async def raw_to_text(self, raw_input):
        pass

    def formatted_latest_buffer(self):
        return None


def test_load_input_success():
    with (
        patch("inputs.find_module_with_class") as mock_find_module,
        patch("importlib.import_module") as mock_import,
    ):
        mock_find_module.return_value = "mock_input"
        mock_module = Mock()
        mock_module.MockInput = MockInput
        mock_import.return_value = mock_module

        result = load_input("MockInput")

        mock_find_module.assert_called_once_with("MockInput")
        mock_import.assert_called_once_with("inputs.plugins.mock_input")
        assert result == MockInput


def test_load_input_not_found():
    with patch("inputs.find_module_with_class") as mock_find_module:
        mock_find_module.return_value = None

        with pytest.raises(
            ValueError,
            match="Class 'NonexistentInput' not found in any input plugin module",
        ):
            load_input("NonexistentInput")


def test_load_input_multiple_plugins():
    with (
        patch("inputs.find_module_with_class") as mock_find_module,
        patch("importlib.import_module") as mock_import,
    ):
        mock_find_module.return_value = "input2"
        mock_module2 = Mock()
        mock_module2.Input2 = type("Input2", (Sensor,), {})
        mock_import.return_value = mock_module2

        result = load_input("Input2")

        mock_find_module.assert_called_once_with("Input2")
        mock_import.assert_called_once_with("inputs.plugins.input2")
        assert result == mock_module2.Input2


def test_load_input_invalid_type():
    with (
        patch("inputs.find_module_with_class") as mock_find_module,
        patch("importlib.import_module") as mock_import,
    ):
        mock_find_module.return_value = "invalid_input"

        class InvalidInput:
            pass

        mock_module = Mock()
        mock_module.InvalidInput = InvalidInput
        mock_import.return_value = mock_module

        with pytest.raises(
            ValueError, match="'InvalidInput' is not a valid input subclass"
        ):
            load_input("InvalidInput")


def test_load_input_import_error_creates_stub():
    """Test that ImportError results in a stub sensor being created."""
    with (
        patch("inputs.find_module_with_class") as mock_find_module,
        patch("importlib.import_module") as mock_import,
        patch("logging.warning") as mock_warning,
    ):
        # Setup: module found but import fails due to missing dependencies
        mock_find_module.return_value = "vlm_coco_local"
        mock_import.side_effect = ImportError("No module named 'torch'")

        # Call load_input
        result = load_input("VLM_COCO_Local")

        # Verify warning was logged
        assert mock_warning.called
        warning_call = mock_warning.call_args[0][0]
        assert "Optional input module" in warning_call
        assert "vlm_coco_local" in warning_call
        assert "missing dependencies" in warning_call

        # Verify stub sensor was returned
        assert result is not None
        assert issubclass(result, Sensor)
        assert "Stub_" in result.__name__

        # Verify stub sensor can be instantiated and returns None on read()
        stub_instance = result()
        assert stub_instance.read() is None


def test_stub_sensor_logs_warning_on_instantiation():
    """Test that stub sensor logs a warning when instantiated."""
    with (
        patch("inputs.find_module_with_class") as mock_find_module,
        patch("importlib.import_module") as mock_import,
        patch("logging.warning") as mock_warning,
    ):
        mock_find_module.return_value = "vlm_vila"
        mock_import.side_effect = ImportError("No module named 'transformers'")

        # Load the stub sensor
        stub_class = load_input("VLM_Vila")

        # Reset mock to check for instantiation warning
        mock_warning.reset_mock()

        # Instantiate the stub sensor
        stub_instance = stub_class()

        # Verify warning was logged during instantiation
        assert mock_warning.called
        instantiation_warning = mock_warning.call_args[0][0]
        assert "Stub sensor" in instantiation_warning
        assert "VLM_Vila" in instantiation_warning
        assert "failed to load" in instantiation_warning
        assert "Install missing dependencies" in instantiation_warning


def test_find_module_with_class_success():
    with (
        patch("os.path.join") as mock_join,
        patch("os.path.exists") as mock_exists,
        patch("os.listdir") as mock_listdir,
        patch(
            "builtins.open",
            mock_open(read_data="class TestInput(FuserInput):\n    pass\n"),
        ),
    ):
        mock_join.side_effect = lambda *args: "/".join(args)
        mock_exists.return_value = True
        mock_listdir.return_value = ["test_input.py"]

        result = find_module_with_class("TestInput")

        assert result == "test_input"


def test_find_module_with_class_not_found():
    with (
        patch("os.path.join") as mock_join,
        patch("os.path.exists") as mock_exists,
        patch("os.listdir") as mock_listdir,
        patch("builtins.open", mock_open(read_data="class OtherClass:\n    pass\n")),
    ):
        mock_join.side_effect = lambda *args: "/".join(args)
        mock_exists.return_value = True
        mock_listdir.return_value = ["other_file.py"]

        result = find_module_with_class("TestInput")

        assert result is None


def test_find_module_with_class_no_plugins_dir():
    with patch("os.path.exists") as mock_exists:
        mock_exists.return_value = False

        result = find_module_with_class("TestInput")

        assert result is None
