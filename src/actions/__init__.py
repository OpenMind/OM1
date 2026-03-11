import importlib
import typing as T

from actions.base import ActionConfig, ActionConnector, AgentAction, Interface


def load_action(
    action_config: T.Dict[str, T.Union[str, T.Dict[str, str]]],
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
    interface = None
    action = importlib.import_module(f"actions.{action_config['name']}.interface")

    for obj in action.__dict__.values():
        if isinstance(obj, type) and issubclass(obj, Interface) and obj != Interface:
            interface = obj

    if interface is None:
        raise ValueError(f"No interface found for action {action_config['name']}")

    connector = importlib.import_module(
        f"actions.{action_config['name']}.connector.{action_config['connector']}"
    )

    connector_class = None
    config_class = None
    for obj in connector.__dict__.values():
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
        name=action_config["name"],  # type: ignore
        llm_label=action_config["llm_label"],  # type: ignore
        interface=interface,
        connector=connector_class(config),
        exclude_from_prompt=exclude_from_prompt,
    )
