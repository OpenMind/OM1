"""Adapter that wraps WebSim behind a minimal, testable interface."""

from __future__ import annotations

import socket
from contextlib import closing
from typing import Any

from llm.output_model import Action


class WebSimAdapter:
    """Thin wrapper exposing only the move/speak/face operations the MCP server needs."""

    def __init__(self, websim: Any):
        """Wrap an already-constructed WebSim-like object (used directly in tests with mocks)."""
        self._websim = websim

    @classmethod
    def create(cls, host: str, port: int) -> "WebSimAdapter":
        """Factory that also starts a real WebSim on (host, port)."""
        cls.ensure_port_free(host, port)
        # Imported here so unit tests for the adapter don't need WebSim.
        from simulators.base import SimulatorConfig
        from simulators.plugins.WebSim import WebSim

        return cls(WebSim(SimulatorConfig(host=host, port=port)))

    @staticmethod
    def ensure_port_free(host: str, port: int) -> None:
        """Raise RuntimeError if ``(host, port)`` is already accepting connections."""
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            if s.connect_ex((host, port)) == 0:
                raise RuntimeError(
                    f"Port {port} on {host} is already in use. "
                    f"Set OM1_WEBSIM_PORT to a free port or kill the conflict."
                )

    def move(self, action: str) -> None:
        """Dispatch a ``move`` Action to the underlying WebSim."""
        self._websim.sim([Action(type="move", value=action)])

    def speak(self, text: str) -> None:
        """Dispatch a ``speak`` Action to the underlying WebSim."""
        self._websim.sim([Action(type="speak", value=text)])

    def face(self, emotion: str) -> None:
        """Dispatch an ``emotion`` Action to the underlying WebSim."""
        self._websim.sim([Action(type="emotion", value=emotion)])
