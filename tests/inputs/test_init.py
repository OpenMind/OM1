from unittest.mock import Mock, mock_open, patch

import pytest

from inputs import find_module_with_class, load_input
from inputs.base import Sensor


class MockInput(Sensor):
    async def raw_to_text(self, raw_input):
        pass

    def formatted_latest_buffer(self):
        return None


class MockConfig:
    def __init__(self, **kwargs):
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

        with pytest.raises(
            ValueError, match="'InvalidInput' is not a valid input subclass"
        ):
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
