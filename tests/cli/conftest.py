"""Shared fixtures for CLI tests."""

import re
from pathlib import Path
from typing import Generator

import pytest
from typer.testing import CliRunner as _CliRunner

from cli.main import app


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from text."""
    ansi_pattern = re.compile(r"\x1b\[[0-9;]*m")
    return ansi_pattern.sub("", text)


class StrippedResult:
    """Wrapper for CLI result with ANSI-stripped output."""

    def __init__(self, result):
        """Initialize with original result."""
        self._result = result
        self._output = strip_ansi(result.output)

    @property
    def output(self) -> str:
        """Return ANSI-stripped output."""
        return self._output

    @property
    def exit_code(self) -> int:
        """Return exit code."""
        return self._result.exit_code

    @property
    def exception(self):
        """Return exception if any."""
        return self._result.exception


class CliRunner(_CliRunner):
    """CLI runner that strips ANSI codes from output for reliable assertions."""

    def invoke(self, *args, **kwargs):
        """Invoke CLI and return result with ANSI codes stripped."""
        result = super().invoke(*args, **kwargs)
        return StrippedResult(result)


@pytest.fixture
def runner() -> CliRunner:
    """Create a CLI test runner that strips ANSI codes from output."""
    return CliRunner()


@pytest.fixture
def cli_app():
    """Return the CLI application."""
    return app


@pytest.fixture
def temp_config_dir(tmp_path: Path) -> Generator[Path, None, None]:
    """Create a temporary config directory."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    basic_config = config_dir / "test_basic.json5"
    basic_config.write_text(
        """{
  version: "v1.0.1",
  hertz: 1,
  name: "test_basic",
  api_key: "openmind_free",
  system_prompt_base: "Test prompt",
  agent_inputs: [],
  cortex_llm: {
    type: "OpenAILLM",
    config: {
      agent_name: "Test",
      history_length: 10,
    },
  },
  agent_actions: [],
  backgrounds: [],
}
""",
        encoding="utf-8",
    )

    mode_aware_config = config_dir / "test_modes.json5"
    mode_aware_config.write_text(
        """{
  version: "v1.0.1",
  default_mode: "default",
  allow_manual_switching: true,
  mode_memory_enabled: true,
  api_key: "openmind_free",
  cortex_llm: {
    type: "OpenAILLM",
    config: {
      agent_name: "Test",
      history_length: 10,
    },
  },
  modes: {
    default: {
      display_name: "Default Mode",
      description: "Test mode",
      system_prompt_base: "Test prompt",
      hertz: 1,
      agent_inputs: [],
      agent_actions: [],
    },
  },
  transition_rules: [],
}
""",
        encoding="utf-8",
    )

    yield config_dir


@pytest.fixture
def mock_config_dir(temp_config_dir: Path, monkeypatch):
    """Mock the config directory to use temp directory."""
    from cli.commands import init as init_module
    from cli.utils import config as config_module

    monkeypatch.setattr(config_module, "get_config_dir", lambda: temp_config_dir)
    monkeypatch.setattr(init_module, "get_config_dir", lambda: temp_config_dir)
    return temp_config_dir
