import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import json5

from actions import load_action
from actions.base import AgentAction
from backgrounds import load_background
from backgrounds.base import Background
from inputs import load_input
from inputs.base import Sensor
from llm import LLM, load_llm
from runtime.robotics import load_unitree
from runtime.version import verify_runtime_version
from simulators import load_simulator
from simulators.base import Simulator


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

    robot_ip: Optional[str] = None
    api_key: Optional[str] = None
    URID: Optional[str] = None
    unitree_ethernet: Optional[str] = None
    mode: Optional[str] = None

    @classmethod
    def load(cls, config_name: str) -> "RuntimeConfig":
        return load_config(config_name)


def load_config(
    config_name: str, config_source_path: Optional[str] = None
) -> RuntimeConfig:
    config_path = (
        os.path.join(
            os.path.dirname(__file__), "../../../config", config_name + ".json5"
        )
        if config_source_path is None
        else config_source_path
    )

    with open(config_path, "r+") as f:
        raw_config = json5.load(f)

    config_version = raw_config.get("version")
    verify_runtime_version(config_version, config_name)

    # ------------------------------------------------------------------
    # ENV FLAGS (HEADLESS / VPS MODE)
    # ------------------------------------------------------------------
    TEXT_ONLY_MODE = (
        os.getenv("OM_FORCE_TEXT_MODE") == "1"
        or os.getenv("OM_DISABLE_ASR") == "1"
    )

    if TEXT_ONLY_MODE:
        logging.warning(
            "Text-only mode enabled (OM_FORCE_TEXT_MODE / OM_DISABLE_ASR). "
            "All audio/ASR inputs will be skipped."
        )

    # ------------------------------------------------------------------
    # ROBOT / API / URID
    # ------------------------------------------------------------------
    g_robot_ip = raw_config.get("robot_ip")
    if not g_robot_ip or g_robot_ip == "192.168.0.241":
        logging.warning(
            "No robot ip found in the configuration file. Checking .env."
        )
        g_robot_ip = os.environ.get("ROBOT_IP")
        if g_robot_ip:
            raw_config["robot_ip"] = g_robot_ip
            logging.info("Found ROBOT_IP in .env.")
        else:
            logging.warning("Robot IP not found.")

    g_api_key = raw_config.get("api_key")
    if not g_api_key or g_api_key == "openmind_free":
        logging.warning(
            "No API key found in config. Checking OM_API_KEY in .env."
        )
        g_api_key = os.environ.get("OM_API_KEY")
        if g_api_key:
            raw_config["api_key"] = g_api_key
            logging.info("Found OM_API_KEY in .env.")
        else:
            logging.warning("No API key found.")

    g_URID = raw_config.get("URID")
    if not g_URID:
        logging.warning("No URID found. Multirobot deployments may conflict.")

    if g_URID == "default":
        backup_URID = os.environ.get("URID")
        if backup_URID:
            g_URID = backup_URID
            logging.info("Found URID in .env.")

    g_ut_eth = raw_config.get("unitree_ethernet")
    if g_ut_eth:
        load_unitree(g_ut_eth)
    else:
        logging.info("No robot hardware ethernet port provided.")

    # ------------------------------------------------------------------
    # BACKGROUNDS
    # ------------------------------------------------------------------
    backgrounds = [
        load_background(
            {
                **bg,
                "config": add_meta(
                    bg.get("config", {}),
                    g_api_key,
                    g_ut_eth,
                    g_URID,
                    g_robot_ip,
                ),
            }
        )
        for bg in raw_config.get("backgrounds", [])
    ]

    # ------------------------------------------------------------------
    # AGENT INPUTS (FIX: SKIP AUDIO / ASR IN TEXT-ONLY MODE)
    # ------------------------------------------------------------------
    if TEXT_ONLY_MODE:
        agent_inputs: List[Sensor] = []
    else:
        agent_inputs = [
            load_input(
                {
                    **inp,
                    "config": add_meta(
                        inp.get("config", {}),
                        g_api_key,
                        g_ut_eth,
                        g_URID,
                        g_robot_ip,
                    ),
                }
            )
            for inp in raw_config.get("agent_inputs", [])
        ]

    # ------------------------------------------------------------------
    # SIMULATORS
    # ------------------------------------------------------------------
    simulators = [
        load_simulator(
            {
                **sim,
                "config": add_meta(
                    sim.get("config", {}),
                    g_api_key,
                    g_ut_eth,
                    g_URID,
                    g_robot_ip,
                ),
            }
        )
        for sim in raw_config.get("simulators", [])
    ]

    # ------------------------------------------------------------------
    # ACTIONS
    # ------------------------------------------------------------------
    agent_actions = [
        load_action(
            {
                **action,
                "config": add_meta(
                    action.get("config", {}),
                    g_api_key,
                    g_ut_eth,
                    g_URID,
                    g_robot_ip,
                ),
            }
        )
        for action in raw_config.get("agent_actions", [])
    ]

    # ------------------------------------------------------------------
    # LLM
    # ------------------------------------------------------------------
    cortex_llm = load_llm(
        {
            **raw_config["cortex_llm"],
            "config": add_meta(
                raw_config["cortex_llm"].get("config", {}),
                g_api_key,
                g_ut_eth,
                g_URID,
                g_robot_ip,
            ),
        },
        available_actions=agent_actions,
    )

    return RuntimeConfig(
        **raw_config,
        agent_inputs=agent_inputs,
        backgrounds=backgrounds,
        simulators=simulators,
        agent_actions=agent_actions,
        cortex_llm=cortex_llm,
    )


def add_meta(
    config: Dict,
    g_api_key: Optional[str],
    g_ut_eth: Optional[str],
    g_URID: Optional[str],
    g_robot_ip: Optional[str],
    g_mode: Optional[str] = None,
) -> dict:
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
