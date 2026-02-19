from dataclasses import dataclass
from typing import Any, Optional
from unittest.mock import MagicMock, patch

from actions.orchestrator import ActionResult
from fuser import Fuser
from inputs.base import Sensor, SensorConfig
from runtime.config import RuntimeConfig


@dataclass
class MockSensor(Sensor[SensorConfig, Any]):
    def __init__(self):
        super().__init__(SensorConfig())

    def formatted_latest_buffer(self):
        return "test input"


def create_mock_config() -> RuntimeConfig:
    mock_config = MagicMock(spec=RuntimeConfig)
    mock_config.system_prompt_base = "base"
    mock_config.system_governance = "governance"
    mock_config.system_prompt_examples = ""
    mock_config.agent_actions = []
    return mock_config


def test_no_feedback_when_empty_promises():
    """Empty promise list should produce no feedback section in prompt."""
    config = create_mock_config()

    with patch("fuser.IOProvider", return_value=MagicMock()):
        fuser = Fuser(config)
        result = fuser.fuse([MockSensor()], [])

    assert "PREVIOUS ACTION RESULTS" not in result


def test_successful_action_feedback_in_prompt():
    """Successful action should show OK in feedback."""
    config = create_mock_config()
    results = [
        ActionResult(action_type="speak", action_value="Hello!", success=True),
    ]

    with patch("fuser.IOProvider", return_value=MagicMock()):
        fuser = Fuser(config)
        result = fuser.fuse([MockSensor()], results)

    assert "PREVIOUS ACTION RESULTS:" in result
    assert "- speak('Hello!'): OK" in result


def test_failed_action_feedback_in_prompt():
    """Failed action should show FAILED with error in feedback."""
    config = create_mock_config()
    results = [
        ActionResult(
            action_type="move",
            action_value="forward",
            success=False,
            error="TimeoutError: Connection timed out",
        ),
    ]

    with patch("fuser.IOProvider", return_value=MagicMock()):
        fuser = Fuser(config)
        result = fuser.fuse([MockSensor()], results)

    assert "PREVIOUS ACTION RESULTS:" in result
    assert "- move('forward'): FAILED (TimeoutError: Connection timed out)" in result


def test_mixed_feedback_in_prompt():
    """Mixed success/failure should show both."""
    config = create_mock_config()
    results = [
        ActionResult(action_type="speak", action_value="Hi", success=True),
        ActionResult(
            action_type="move",
            action_value="forward",
            success=False,
            error="TimeoutError",
        ),
    ]

    with patch("fuser.IOProvider", return_value=MagicMock()):
        fuser = Fuser(config)
        result = fuser.fuse([MockSensor()], results)

    assert "- speak('Hi'): OK" in result
    assert "- move('forward'): FAILED (TimeoutError)" in result


def test_feedback_appears_before_available_inputs():
    """Action feedback should appear before AVAILABLE INPUTS in the prompt."""
    config = create_mock_config()
    results = [
        ActionResult(action_type="speak", action_value="test", success=True),
    ]

    with patch("fuser.IOProvider", return_value=MagicMock()):
        fuser = Fuser(config)
        result = fuser.fuse([MockSensor()], results)

    feedback_pos = result.index("PREVIOUS ACTION RESULTS")
    inputs_pos = result.index("AVAILABLE INPUTS")
    assert feedback_pos < inputs_pos
