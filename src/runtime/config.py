import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from jsonschema import ValidationError, validate

from actions.base import AgentAction
from backgrounds.base import Background
from inputs.base import Sensor
from llm import LLM
from simulators.base import Simulator


def _load_schema(schema_file: str) -> dict:
    """
    Load and cache schema files.

    Parameters
    ----------
    schema_file : str
        Name of the schema file to load.

    Returns
    -------
    dict
        The loaded schema dictionary.

    Raises
    ------
    FileNotFoundError
        If the schema file does not exist.
    """
    schema_path = Path(__file__).parent / "../../config/schema" / schema_file

    if not schema_path.exists():
        raise FileNotFoundError(
            f"Schema file not found: {schema_path}. Cannot validate configuration."
        )

    with open(schema_path, "r") as f:
        return json.load(f)


def validate_config_schema(raw_config: dict) -> None:
    """
    Validate the configuration against the appropriate schema.

    Parameters
    ----------
    raw_config : dict
        The raw configuration dictionary to validate.
    """
    schema_file = "multi_mode_schema.json"

    try:
        schema = _load_schema(schema_file)
        validate(instance=raw_config, schema=schema)

    except FileNotFoundError as e:
        logging.error(str(e))
        raise
    except ValidationError as e:
        field_path = ".".join(str(p) for p in e.path) if e.path else "root"
        logging.error(f"Schema validation failed at field '{field_path}': {e.message}")
        raise


@dataclass
class RuntimeConfig:
    """Runtime configuration for the agent."""

    version: str
    hertz: float
    name: str
    system_prompt_base: str
    system_governance: str
    system_prompt_examples: str

    agent_inputs: List[Sensor]
    cortex_llm: LLM
    simulators: List[Simulator]
    agent_actions: List[AgentAction]
    backgrounds: List[Background]

    mode: Optional[str] = None
    api_key: Optional[str] = None
    robot_ip: Optional[str] = None
    URID: Optional[str] = None
    unitree_ethernet: Optional[str] = None
    action_execution_mode: Optional[str] = None
    action_dependencies: Optional[Dict[str, List[str]]] = None


def add_meta(
    config: Dict,
    g_api_key: Optional[str],
    g_ut_eth: Optional[str],
    g_URID: Optional[str],
    g_robot_ip: Optional[str],
    g_mode: Optional[str] = None,
) -> dict[str, str]:
    """Add API key and robot configuration to a component's config dict."""
    if "api_key" not in config and g_api_key is not None:
        config["api_key"] = g_api_key
    if "unitree_ethernet" not in config and g_ut_eth is not None:
        config["unitree_ethernet"] = g_ut_eth
    if "URID" not in config and g_URID is not None:
        config["URID"] = g_URID
    if "robot_ip" not in config and g_robot_ip is not None:
        config["robot_ip"] = g_robot_ip
    if "mode" not in config and g_mode is not None:
        config["mode"] = g_mode
    return config


def build_runtime_config_from_test_case(config: dict) -> RuntimeConfig:
    """Build a RuntimeConfig from a test case dictionary."""
    from actions import load_action
    from backgrounds import load_background
    from inputs import load_input
    from llm import load_llm
    from simulators import load_simulator

    api_key = config.get("api_key")
    g_ut_eth = config.get("unitree_ethernet")
    g_URID = config.get("URID")
    g_robot_ip = config.get("robot_ip")

    backgrounds = [
        load_background(
            {
                **bg,
                "config": add_meta(
                    bg.get("config", {}), api_key, g_ut_eth, g_URID, g_robot_ip
                ),
            }
        )
        for bg in config.get("backgrounds", [])
    ]
    agent_inputs = [
        load_input(
            {
                **inp,
                "config": add_meta(
                    inp.get("config", {}), api_key, g_ut_eth, g_URID, g_robot_ip
                ),
            }
        )
        for inp in config.get("agent_inputs", [])
    ]
    simulators = [
        load_simulator(
            {
                **sim,
                "config": add_meta(
                    sim.get("config", {}), api_key, g_ut_eth, g_URID, g_robot_ip
                ),
            }
        )
        for sim in config.get("simulators", [])
    ]
    agent_actions = [
        load_action(
            {
                **action,
                "config": add_meta(
                    action.get("config", {}), api_key, g_ut_eth, g_URID, g_robot_ip
                ),
            }
        )
        for action in config.get("agent_actions", [])
    ]
    cortex_llm = load_llm(
        {
            **config["cortex_llm"],
            "config": add_meta(
                config["cortex_llm"].get("config", {}),
                api_key,
                g_ut_eth,
                g_URID,
                g_robot_ip,
            ),
        },
        available_actions=agent_actions,
    )
    return RuntimeConfig(
        version=config.get("version", "v1.0.2"),
        hertz=config.get("hertz", 1),
        name=config.get("name", "TestAgent"),
        system_prompt_base=config.get("system_prompt_base", ""),
        system_governance=config.get("system_governance", ""),
        system_prompt_examples=config.get("system_prompt_examples", ""),
        agent_inputs=agent_inputs,
        cortex_llm=cortex_llm,
        simulators=simulators,
        agent_actions=agent_actions,
        backgrounds=backgrounds,
    )
