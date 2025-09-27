from unittest.mock import Mock, mock_open, patch

import pytest

from simulators import find_module_with_class, load_simulator
from simulators.base import Simulator


class MockSimulator(Simulator):
    def process_data(self):
        pass


def test_load_simulator_success():
    with (
        patch("simulators.find_module_with_class") as mock_find_module,
        patch("importlib.import_module") as mock_import,
    ):
        mock_find_module.return_value = "mock_simulator"
        mock_module = Mock()
        mock_module.MockSimulator = MockSimulator
        mock_import.return_value = mock_module

        result = load_simulator("MockSimulator")

        mock_find_module.assert_called_once_with("MockSimulator")
        mock_import.assert_called_once_with("simulators.plugins.mock_simulator")
        assert result == MockSimulator


def test_load_simulator_not_found():
    with patch("simulators.find_module_with_class") as mock_find_module:
        mock_find_module.return_value = None

        with pytest.raises(
            ValueError,
            match="Class 'NonexistentSimulator' not found in any simulator plugin module",
        ):
            load_simulator("NonexistentSimulator")


def test_load_simulator_multiple_plugins():
    with (
        patch("simulators.find_module_with_class") as mock_find_module,
        patch("importlib.import_module") as mock_import,
    ):
        mock_find_module.return_value = "sim2"
        mock_module2 = Mock()
        mock_module2.Simulator2 = type("Simulator2", (Simulator,), {})
        mock_import.return_value = mock_module2

        result = load_simulator("Simulator2")

        mock_find_module.assert_called_once_with("Simulator2")
        mock_import.assert_called_once_with("simulators.plugins.sim2")
        assert result == mock_module2.Simulator2


def test_load_simulator_invalid_type():
    with (
        patch("simulators.find_module_with_class") as mock_find_module,
        patch("importlib.import_module") as mock_import,
    ):
        mock_find_module.return_value = "invalid_simulator"

        class InvalidSimulator:
            pass

        mock_module = Mock()
        mock_module.InvalidSimulator = InvalidSimulator
        mock_import.return_value = mock_module

        with pytest.raises(
            ValueError, match="'InvalidSimulator' is not a valid simulator subclass"
        ):
            load_simulator("InvalidSimulator")


def test_find_module_with_class_success():
    with (
        patch("os.path.join") as mock_join,
        patch("os.path.exists") as mock_exists,
        patch("os.listdir") as mock_listdir,
        patch(
            "builtins.open",
            mock_open(read_data="class TestSimulator(Simulator):\n    pass\n"),
        ),
    ):
        mock_join.side_effect = lambda *args: "/".join(args)
        mock_exists.return_value = True
        mock_listdir.return_value = ["test_simulator.py"]

        result = find_module_with_class("TestSimulator")

        assert result == "test_simulator"


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

        result = find_module_with_class("TestSimulator")

        assert result is None


def test_find_module_with_class_no_plugins_dir():
    with patch("os.path.exists") as mock_exists:
        mock_exists.return_value = False

        result = find_module_with_class("TestSimulator")

        assert result is None
