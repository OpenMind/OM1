"""Tests for runtime config with safety sandbox."""

import os
import tempfile

from src.runtime.config import load_mode_config


def test_load_mode_config_with_safety_sandbox():
    """Test loading a mode config that includes safety_sandbox."""
    # Create a temporary config file
    config_content = """
    {
        version: "v1.0.2",
        default_mode: "test",
        api_key: "dummy",
        system_governance: "Be safe.",
        cortex_llm: { type: "OpenAILLM", config: { agent_name: "test" } },
        modes: {
            test: {
                display_name: "Test",
                description: "Test mode",
                system_prompt_base: "You are a test.",
                hertz: 1,
                safety_sandbox: {
                    enabled: true,
                    simulator: "WebSim",
                    simulation_timeout: 2.0,
                    obstacle_margin: 0.3
                },
                agent_inputs: [],
                agent_actions: []
            }
        }
    }
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json5", delete=False) as f:
        f.write(config_content)
        tmp_path = f.name

    try:
        config = load_mode_config("dummy", mode_source_path=tmp_path)
        mode = config.modes["test"]
        assert mode._raw_safety_sandbox == {
            "enabled": True,
            "simulator": "WebSim",
            "simulation_timeout": 2.0,
            "obstacle_margin": 0.3,
        }
    finally:
        os.unlink(tmp_path)


def test_load_mode_config_without_safety_sandbox():
    """Test loading a mode config without safety_sandbox."""
    config_content = """
    {
        version: "v1.0.2",
        default_mode: "test",
        api_key: "dummy",
        system_governance: "Be safe.",
        cortex_llm: { type: "OpenAILLM", config: { agent_name: "test" } },
        modes: {
            test: {
                display_name: "Test",
                description: "Test mode",
                system_prompt_base: "You are a test.",
                hertz: 1,
                agent_inputs: [],
                agent_actions: []
            }
        }
    }
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json5", delete=False) as f:
        f.write(config_content)
        tmp_path = f.name

    try:
        config = load_mode_config("dummy", mode_source_path=tmp_path)
        mode = config.modes["test"]
        assert mode._raw_safety_sandbox == {}
    finally:
        os.unlink(tmp_path)
