"""Runtime configuration for the OM1 MCP server."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ServerConfig:
    """Immutable runtime configuration for the OM1 MCP server."""

    websim_host: str = "127.0.0.1"
    websim_port: int = 8000
    log_level: str = "WARNING"

    @classmethod
    def from_env(cls) -> "ServerConfig":
        """Build a ServerConfig from OM1_WEBSIM_HOST / OM1_WEBSIM_PORT / OM1_LOG_LEVEL env vars."""
        return cls(
            websim_host=os.getenv("OM1_WEBSIM_HOST", "127.0.0.1"),
            websim_port=int(os.getenv("OM1_WEBSIM_PORT", "8000")),
            log_level=os.getenv("OM1_LOG_LEVEL", "WARNING"),
        )
