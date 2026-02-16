import importlib
import logging
import os
import types
from typing import Optional, Type
from unittest.mock import patch

import json5

from actions.base import ActionConnector, Interface
from inputs import find_module_with_class
from inputs.base import Sensor
from llm import get_llm_class
from simulators import get_simulator_class


def test_configs():
    """Test that all config files can be loaded."""
    config_folder_path = os.path.join(os.path.dirname(__file__), "../../config")
    files_names = [
        entry.name for entry in os.scandir(config_folder_path) if entry.is_file()
    ]

    for file_name in files_names:
        if file_name.endswith(".DS_Store"):
            continue
        assert file_name.endswith(".json5")
        with open(os.path.join(config_folder_path, file_name), "r") as f:
            raw_config = json5.load(f)

        agent_inputs = raw_config.get("agent_inputs", [])
        assert isinstance(agent_inputs, list)

        cortex_llm = raw_config.get("cortex_llm", {})
        assert isinstance(cortex_llm, dict)
        assert "type" in cortex_llm, f"'type' key missing in cortex_llm of {file_name}"
        assert get_llm_class(cortex_llm["type"]) is not None

        simulators = raw_config.get("simulators", [])
        assert isinstance(simulators, list)

        agent_actions = raw_config.get("agent_actions", [])
        assert isinstance(agent_actions, list)

        for input_config in agent_inputs:
            assert_input_class_exists(input_config)

        for simulator in simulators:
            assert get_simulator_class(simulator["type"]) is not None

        for action in agent_actions:
            assert_action_classes_exist(action)


def assert_input_class_exists(input_config):
    """Assert that the input class exists without instantiating it."""
    class_name = input_config["type"]
    module_name = find_module_with_class(class_name)
    assert module_name is not None, f"Input class '{class_name}' not found"

    module = importlib.import_module(f"inputs.plugins.{module_name}")
    input_class = find_subclass_in_module(module, Sensor)
    assert input_class is not None, f"No Sensor subclass found for '{class_name}'"


def assert_action_classes_exist(action_config):
    """Assert that all required classes for an action exist without instantiating them."""
    # Check interface exists
    action_module = importlib.import_module(
        f"actions.{action_config['name']}.interface"
    )
    interface = find_subclass_in_module(action_module, Interface)
    assert (
        interface is not None
    ), f"No interface found for action {action_config['name']}"

    # Check connector exists
    try:
        connector_module = importlib.import_module(
            f"actions.{action_config['name']}.connector.{action_config['connector']}"
        )
        connector = find_subclass_in_module(connector_module, ActionConnector)
        assert (
            connector is not None
        ), f"No connector found for action {action_config['name']}"
    except (ImportError, ModuleNotFoundError) as exc:
        missing_module = getattr(exc, "name", "") or ""
        error_text = str(exc)
        is_optional_dependency = any(
            (
                missing_module.startswith(prefix),
                f"No module named '{prefix}" in error_text,
                f"No module named '{prefix}." in error_text,
            )
            for prefix in ("unitree", "cyclonedds")
        )

        if is_optional_dependency:
            logging.warning(
                "Skipping connector import for action '%s' due to optional dependency: %s",
                action_config["name"],
                exc,
            )
            return

        assert False, f"Connector module not found for action {action_config['name']}"


def find_subclass_in_module(module, parent_class: Type) -> Optional[Type]:
    """Find a subclass of parent_class in the given module."""
    for _, obj in module.__dict__.items():
        if (
            isinstance(obj, type)
            and issubclass(obj, parent_class)
            and obj != parent_class
        ):
            return obj
    return None


def test_assert_action_classes_exist_skips_optional_dependency_import_error(caplog):
    """Optional connector imports should warn instead of failing config tests."""

    class DummyInterface(Interface):
        pass

    def _fake_import(module_name: str):
        if module_name == "actions.fake_action.interface":
            return types.SimpleNamespace(DummyInterface=DummyInterface)
        if module_name == "actions.fake_action.connector.fake_connector":
            raise ModuleNotFoundError("No module named 'unitree.unitree_sdk2py'")
        raise AssertionError(f"Unexpected import path: {module_name}")

    with patch("importlib.import_module", side_effect=_fake_import):
        with caplog.at_level(logging.WARNING):
            assert_action_classes_exist(
                {"name": "fake_action", "connector": "fake_connector"}
            )

    assert "Skipping connector import for action 'fake_action'" in caplog.text


def test_assert_action_classes_exist_still_fails_for_non_optional_import_error():
    """Non-optional import errors should still fail the config test."""

    class DummyInterface(Interface):
        pass

    def _fake_import(module_name: str):
        if module_name == "actions.fake_action.interface":
            return types.SimpleNamespace(DummyInterface=DummyInterface)
        if module_name == "actions.fake_action.connector.fake_connector":
            raise ModuleNotFoundError("No module named 'actions.fake_action.connector'")
        raise AssertionError(f"Unexpected import path: {module_name}")

    with patch("importlib.import_module", side_effect=_fake_import):
        try:
            assert_action_classes_exist(
                {"name": "fake_action", "connector": "fake_connector"}
            )
            assert False, "Expected non-optional import error to fail"
        except AssertionError as exc:
            assert "Connector module not found for action fake_action" in str(exc)
