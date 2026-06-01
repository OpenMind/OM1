import types
from unittest.mock import Mock, mock_open, patch

import pytest

from inputs import find_module_with_class, load_input
from inputs.base import Sensor, SensorConfig


class MockInput(Sensor):
    def __init__(self, config=None):
        pass

    async def raw_to_text(self, raw_input):
        pass

    def formatted_latest_buffer(self):
        return None


class MockConfig(SensorConfig):
    pass


def test_load_input_success():
    with (
        patch("inputs.find_module_with_class") as mock_find_module,
        patch("importlib.import_module") as mock_import,
    ):
        mock_find_module.return_value = "mock_input"
        mock_module = Mock()
        mock_module.MockInput = MockInput
        mock_import.return_value = mock_module

        mock_module.MockConfig = MockConfig
        result = load_input({"type": "MockInput"})

        mock_find_module.assert_called_once_with("MockInput")
        mock_import.assert_called_once_with("inputs.plugins.mock_input")
        assert isinstance(result, Sensor)


def test_load_input_not_found():
    with patch("inputs.find_module_with_class") as mock_find_module:
        mock_find_module.return_value = None

        with pytest.raises(
            ValueError,
            match="Class 'NonexistentInput' not found in any input plugin module",
        ):
            load_input({"type": "NonexistentInput"})


def test_load_input_multiple_plugins():
    with (
        patch("inputs.find_module_with_class") as mock_find_module,
        patch("importlib.import_module") as mock_import,
    ):
        mock_find_module.return_value = "input2"
        mock_module2 = Mock()
        Input2 = type(
            "Input2",
            (Sensor,),
            {
                "__init__": lambda self, config: None,
                "raw_to_text": lambda self, r: None,
            },
        )
        mock_module2.Input2 = Input2
        mock_import.return_value = mock_module2

        result = load_input({"type": "Input2"})

        mock_find_module.assert_called_once_with("Input2")
        mock_import.assert_called_once_with("inputs.plugins.input2")
        assert isinstance(result, Sensor)


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

        with pytest.raises(ValueError, match="'InvalidInput' is not a valid input subclass"):
            load_input({"type": "InvalidInput"})


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


def test_find_module_with_class_in_subpackage():
    """Cover subpackage __init__.py scanning — found."""
    with (
        patch("os.path.join", side_effect=lambda *args: "/".join(args)),
        patch("os.path.exists") as mock_exists,
        patch("os.listdir") as mock_listdir,
        patch("os.path.isdir") as mock_isdir,
        patch(
            "builtins.open",
            mock_open(read_data="class AirQualityInput(FuserInput):\n    pass\n"),
        ),
    ):
        mock_exists.return_value = True
        mock_listdir.return_value = ["air_quality"]
        mock_isdir.return_value = True

        result = find_module_with_class("AirQualityInput")
        assert result == "air_quality"


def test_find_module_with_class_subpackage_no_init():
    """Cover subpackage without __init__.py — skipped."""
    with (
        patch("os.path.join", side_effect=lambda *args: "/".join(args)),
        patch("os.path.exists") as mock_exists,
        patch("os.listdir") as mock_listdir,
        patch("os.path.isdir") as mock_isdir,
    ):

        def exists_side_effect(path):
            if path.endswith("__init__.py"):
                return False
            return True

        mock_exists.side_effect = exists_side_effect
        mock_listdir.return_value = ["air_quality"]
        mock_isdir.return_value = True

        result = find_module_with_class("AirQualityInput")
        assert result is None


def test_find_module_with_class_subpackage_read_error():
    """Cover subpackage __init__.py read exception."""
    with (
        patch("os.path.join", side_effect=lambda *args: "/".join(args)),
        patch("os.path.exists", return_value=True),
        patch("os.listdir") as mock_listdir,
        patch("os.path.isdir") as mock_isdir,
        patch("builtins.open", side_effect=OSError("permission denied")),
    ):
        mock_listdir.return_value = ["air_quality"]
        mock_isdir.return_value = True

        result = find_module_with_class("AirQualityInput")
        assert result is None


def test_find_module_with_class_skips_underscore_dirs():
    """Cover that dirs starting with _ are skipped."""
    with (
        patch("os.path.join", side_effect=lambda *args: "/".join(args)),
        patch("os.path.exists", return_value=True),
        patch("os.listdir") as mock_listdir,
        patch("os.path.isdir", return_value=True),
    ):
        mock_listdir.return_value = ["__pycache__", "_internal"]

        result = find_module_with_class("AirQualityInput")
        assert result is None


def test_find_module_with_class_direct_file_read_error():
    """Cover except Exception when reading direct .py plugin file."""
    with (
        patch("os.path.join", side_effect=lambda *args: "/".join(args)),
        patch("os.path.exists", return_value=True),
        patch("os.listdir", return_value=["broken_input.py"]),
        patch("os.path.isdir", return_value=False),
        patch("builtins.open", side_effect=OSError("permission denied")),
    ):
        result = find_module_with_class("BrokenInput")
        assert result is None


def test_load_input_with_config_class():
    """Cover config_class is not None branch (line 104)."""
    with (
        patch("inputs.find_module_with_class") as mock_find,
        patch("importlib.import_module") as mock_import,
    ):
        mock_find.return_value = "mock_input"
        mock_module = Mock()
        mock_module.MockInput = MockInput
        mock_module.MockConfig = MockConfig
        mock_import.return_value = mock_module

        result = load_input({"type": "MockInput", "config": {"key": "value"}})
        assert isinstance(result, Sensor)


def test_load_input_without_config_class():
    """Cover config_class is None branch (line 108) — no SensorConfig in module."""
    with (
        patch("inputs.find_module_with_class") as mock_find,
        patch("importlib.import_module") as mock_import,
    ):
        mock_find.return_value = "mock_input"

        mock_module = types.ModuleType("mock_input")
        setattr(mock_module, "MockInput", MockInput)
        mock_import.return_value = mock_module

        result = load_input({"type": "MockInput"})
        assert isinstance(result, Sensor)


def test_load_input_import_error():
    """Cover ImportError → ValueError (line 120)."""
    with (
        patch("inputs.find_module_with_class") as mock_find,
        patch("importlib.import_module", side_effect=ImportError("no module")),
    ):
        mock_find.return_value = "missing_input"
        with pytest.raises(ValueError, match="Could not import input module"):
            load_input({"type": "MissingInput"})


def test_load_input_attribute_error():
    """Cover AttributeError → ValueError (line 122)."""
    with (
        patch("inputs.find_module_with_class") as mock_find,
        patch("importlib.import_module") as mock_import,
    ):
        mock_find.return_value = "mock_input"
        mock_module = Mock(spec=[])  # no attributes
        mock_import.return_value = mock_module

        with pytest.raises(ValueError, match="not found in input module"):
            load_input({"type": "MockInput"})
