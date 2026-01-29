import asyncio
import json
import time
from typing import Any, Dict, List, Optional

from pydantic import Field

from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput
from providers.io_provider import IOProvider


class MCPToolResultsConfig(SensorConfig):
    """Configuration for MCP tool results input."""

    poll_interval: float = Field(default=0.5, description="Polling interval in seconds")
    max_items: int = Field(default=5, description="Max MCP results per tick")
    variable_key: str = Field(
        default="mcp_results",
        description="IOProvider dynamic variable key holding MCP results",
    )
    descriptor: str = Field(
        default="Recent MCP tool results",
        description="Descriptor label for the LLM prompt section",
    )


class MCPToolResults(FuserInput[MCPToolResultsConfig, Optional[List[Dict[str, Any]]]]):
    """Input plugin that surfaces MCP tool call results to the LLM."""

    def __init__(self, config: MCPToolResultsConfig):
        super().__init__(config)
        self.io_provider = IOProvider()
        self.messages: List[Message] = []

    async def _poll(self) -> Optional[List[Dict[str, Any]]]:
        await asyncio.sleep(self.config.poll_interval)

        results = self.io_provider.get_dynamic_variable(self.config.variable_key)
        if not results or not isinstance(results, list):
            return None

        # Clear stored results once read
        self.io_provider.add_dynamic_variable(self.config.variable_key, [])
        return results[: self.config.max_items]

    async def _raw_to_text(
        self, raw_input: Optional[List[Dict[str, Any]]]
    ) -> Optional[Message]:
        if raw_input is None:
            return None

        lines = []
        for entry in raw_input:
            tool = entry.get("tool", "unknown")
            output = entry.get("output", "")
            error = entry.get("error")
            if error:
                lines.append(f"{tool}: ERROR - {error}")
            else:
                formatted = output
                if isinstance(output, (dict, list)):
                    formatted = json.dumps(output)
                lines.append(f"{tool}: {formatted}")

        if not lines:
            return None

        return Message(timestamp=time.time(), message="\n".join(lines))

    async def raw_to_text(self, raw_input: Optional[List[Dict[str, Any]]]):
        if raw_input is None:
            return
        pending = await self._raw_to_text(raw_input)
        if pending:
            self.messages.append(pending)

    def formatted_latest_buffer(self) -> Optional[str]:
        if not self.messages:
            return None

        message = self.messages[-1]
        result = f"""
INPUT: {self.config.descriptor}
// START
{message.message}
// END
"""

        self.io_provider.add_input(
            self.__class__.__name__, message.message, message.timestamp
        )
        self.messages = []
        return result
