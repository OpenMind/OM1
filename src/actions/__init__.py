import importlib
import re
import typing as T
from enum import Enum
from pathlib import Path
from typing import Optional, Set, Tuple

from actions.base import ActionConfig, ActionConnector, AgentAction, Interface

_VALID_ACTION_CONNECTOR_PAIRS: Optional[Set[Tuple[str, str]]] = None


def _enumerate_valid_actions_and_connectors(
    actions_dir: str = "src/actions",
) -> Set[Tuple[str, str]]:
    """Enumerates valid (action_name, connector_name) pairs from the actions directory."""
    valid_pairs = set()
    actions_path = Path(actions_dir)

    if not actions_path.is_dir():
        print(f"Warning: Actions directory '{actions_dir}' not found.")
        return valid_pairs

    for action_path in actions_path.iterdir():
        if not action_path.is_dir() or action_path.name.startswith("__"):
            continue

        action_name = action_path.name
        connector_dir = action_path / "connector"

        if connector_dir.is_dir():
            for connector_file in connector_dir.iterdir():
                if (
                    connector_file.suffix == ".py"
                    and connector_file.name != "__init__.py"
                ):
                    connector_name = connector_file.stem
                    valid_pairs.add((action_name, connector_name))

    return valid_pairs


def _get_valid_action_connector_pairs() -> Set[Tuple[str, str]]:
    """Gets the cached set of valid (action_name, connector_name) pairs."""
    global _VALID_ACTION_CONNECTOR_PAIRS
    if _VALID_ACTION_CONNECTOR_PAIRS is None:
        _VALID_ACTION_CONNECTOR_PAIRS = _enumerate_valid_actions_and_connectors()
    return _VALID_ACTION_CONNECTOR_PAIRS


def _validate_action_and_connector(
    action_name: str, connector_name: Optional[str] = None
) -> bool:
    """Validates action_name and optionally connector_name against the whitelist."""
    valid_pairs = _get_valid_action_connector_pairs()

    if connector_name is not None:
        return (action_name, connector_name) in valid_pairs

    return any(pair[0] == action_name for pair in valid_pairs)


def describe_action(
    action_name: str, llm_label: str, exclude_from_prompt: bool
) -> Optional[str]:
    """
    Generate a description of the action for use in prompts.

    Parameters
    ----------
    action_name : str
        The name of the action.
    llm_label : str
        The label used by the LLM for this action.
    exclude_from_prompt : bool
        Whether to exclude this action from the prompt. If True, returns None.

    Returns
    -------
    Optional[str]
        A formatted description of the action, or None if excluded.
    """
    if not _validate_action_and_connector(action_name):
        raise ValueError(f"Invalid action name '{action_name}' for description.")

    if not re.match(r"^[a-zA-Z0-9_-]+$", action_name):
        raise ValueError(
            f"Invalid characters in action name '{action_name}'. Only alphanumeric, underscore, and hyphen are allowed."
        )

    if exclude_from_prompt:
        return None

    action = importlib.import_module(f"actions.{action_name}.interface")

    interface = None
    for _, obj in action.__dict__.items():
        if isinstance(obj, type) and issubclass(obj, Interface) and obj != Interface:
            interface = obj

    if interface is None:
        raise ValueError(f"No interface found for action {action_name}")

    doc = interface.__doc__ or ""
    doc = doc.replace("\n", "")

    hints = {}
    input_interface = T.get_type_hints(interface)["input"]
    for field_name, field_type in T.get_type_hints(input_interface).items():
        if hasattr(field_type, "__origin__") and isinstance(
            field_type.__origin__, type
        ):
            hints[field_name] = str(field_type)
        elif isinstance(field_type, type) and issubclass(field_type, Enum):
            values = [f"'{v.value}'" for v in field_type]
            hints[field_name] = "value={" + f"{', '.join(values)}" + "}"
        else:
            hints[field_name] = f"value={str(field_type)}"

    type_hints = "\n".join(f"{desc}" for name, desc in hints.items())
    final_description = f"{llm_label.upper()}: {doc}\ntype={llm_label}\n{type_hints}"
    final_description = final_description.replace("    ", "")

    return final_description


def load_action(
    action_config: T.Dict[str, T.Any],
) -> AgentAction:
    """
    Load an action based on the provided configuration.

    Parameters
    ----------
    action_config : Dict[str, Union[str, Dict[str, str]]]
        Configuration dictionary for the action, including 'name', 'llm_label',
        'connector', and optional 'config' and 'exclude_from_prompt' keys.

    Returns
    -------
    AgentAction
        An instance of AgentAction with the specified interface and connector.
    """
    action_name_raw = action_config["name"]
    connector_name_raw = action_config["connector"]

    if not isinstance(action_name_raw, str):
        raise TypeError(f"Expected 'name' to be str, got {type(action_name_raw)}")
    if not isinstance(connector_name_raw, str):
        raise TypeError(
            f"Expected 'connector' to be str, got {type(connector_name_raw)}"
        )

    action_name = action_name_raw
    connector_name = connector_name_raw

    if not _validate_action_and_connector(action_name, connector_name):
        raise ValueError(
            f"Invalid action name '{action_name}' or connector name '{connector_name}' for loading."
        )

    if not re.match(r"^[a-zA-Z0-9_-]+$", action_name):
        raise ValueError(
            f"Invalid characters in action name '{action_name}'. Only alphanumeric, underscore, and hyphen are allowed."
        )
    if not re.match(r"^[a-zA-Z0-9_-]+$", connector_name):
        raise ValueError(
            f"Invalid characters in connector name '{connector_name}'. Only alphanumeric, underscore, and hyphen are allowed."
        )

    action = importlib.import_module(f"actions.{action_name}.interface")

    interface = None
    for _, obj in action.__dict__.items():
        if isinstance(obj, type) and issubclass(obj, Interface) and obj != Interface:
            interface = obj

    if interface is None:
        raise ValueError(f"No interface found for action {action_config['name']}")

    connector = importlib.import_module(
        f"actions.{action_config['name']}.connector.{action_config['connector']}"
    )

    connector_class = None
    config_class = None
    for _, obj in connector.__dict__.items():
        if isinstance(obj, type) and issubclass(obj, ActionConnector):
            connector_class = obj
        if (
            isinstance(obj, type)
            and issubclass(obj, ActionConfig)
            and obj != ActionConfig
        ):
            config_class = obj

    if connector_class is None:
        raise ValueError(
            f"No connector found for action {action_config['name']} connector {action_config['connector']}"
        )

    if config_class is not None:
        config_dict = action_config.get("config", {})
        config = config_class(**(config_dict if isinstance(config_dict, dict) else {}))
    else:
        config_dict = action_config.get("config", {})
        config = ActionConfig(**(config_dict if isinstance(config_dict, dict) else {}))

    exclude_from_prompt = False
    if "exclude_from_prompt" in action_config:
        exclude_from_prompt = bool(action_config["exclude_from_prompt"])

    return AgentAction(
        name=action_config["name"],
        llm_label=action_config["llm_label"],
        interface=interface,
        connector=connector_class(config),
        exclude_from_prompt=exclude_from_prompt,
    )
