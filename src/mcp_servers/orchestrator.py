import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List

from mcp_servers.client import MCPClientManager
from runtime.config import RuntimeConfig


@dataclass
class ToolResult:
    """Result from a single MCP tool execution."""

    tool_key: str
    success: bool
    content: str


class MCPOrchestrator:
    """MCP tool helper used by the cortex tick loop.

    This class provides utilities for splitting actions,
    executing MCP tools concurrently, and building recall prompts.

    Parameters
    ----------
    config : RuntimeConfig
        The runtime configuration object
    max_concurrency : int
        Maximum number of concurrent tool executions per round.
    """

    def __init__(
        self,
        config: RuntimeConfig,
        max_concurrency: int = 5,
        max_rounds: int = 5,
    ) -> None:
        self._config = config
        self._mcp_client: MCPClientManager = config.mcp_servers  # type: ignore
        self._max_concurrency = max_concurrency
        self.max_rounds = max_rounds

    async def start(self) -> None:
        """Connect to MCP servers and inject their tool schemas into the LLM."""
        await self._mcp_client.start()

        llm = self._config.cortex_llm
        mcp_schemas = self._mcp_client.get_tool_schemas()
        base_schemas = [
            schema
            for schema in llm.function_schemas
            if not schema.get("function", {}).get("name", "").startswith("mcp_")
        ]
        llm.function_schemas = base_schemas + mcp_schemas

        logging.info(f"MCP Orchestrator started with {len(mcp_schemas)} MCP tools, " f"{len(base_schemas)} base tools")

    async def execute_mcp_actions(self, actions: list, succeeded_calls: set) -> tuple:
        """Execute MCP actions that haven't succeeded yet.

        Filters out already-succeeded actions, executes new ones,
        and updates the succeeded_calls set in-place.

        Parameters
        ----------
        actions : list
            All actions from LLM output
        succeeded_calls : set
            Set of already-succeeded action signatures (updated in-place)

        Returns
        -------
        tuple
            (results, mcp_actions) if there are new actions to execute,
            (None, None) if no new actions remain
        """
        all_mcp_actions = [a for a in actions if self._mcp_client.is_mcp_tool(a.type)]

        mcp_actions = [a for a in all_mcp_actions if self.build_call_signature(a) not in succeeded_calls]

        if not mcp_actions:
            return None, None

        results = await self._execute_mcp_actions(mcp_actions)

        for action, result in zip(mcp_actions, results):
            if result.success:
                succeeded_calls.add(self.build_call_signature(action))

        return results, mcp_actions

    def extract_om1_actions(self, actions: list) -> list:
        """Extract OM1 actions from a list of actions.

        Parameters
        ----------
        actions : list
            List of all actions from the LLM output.

        Returns
        -------
        list
            List of OM1 actions.
        """
        return [a for a in actions if not self._mcp_client.is_mcp_tool(a.type)]

    async def _execute_mcp_actions(self, actions: list) -> List[ToolResult]:
        """Execute a list of MCP actions concurrently.

        Parameters
        ----------
        actions : list
            MCP actions to execute in parallel.

        Returns
        -------
        List[ToolResult]
            List of tool results.
        """
        semaphore = asyncio.Semaphore(self._max_concurrency)

        async def _guarded(action: Any) -> ToolResult:
            async with semaphore:
                return await self._execute_single_tool(action)

        return await asyncio.gather(*(_guarded(a) for a in actions))

    async def _execute_single_tool(self, action: Any, timeout: float = 10.0) -> ToolResult:
        """Execute one MCP tool call with a timeout.

        Parameters
        ----------
        action : Any
            The MCP action to execute.
        timeout : float
            Maximum execution time in seconds.

        Returns
        -------
        ToolResult
            The execution result containing success flag and content.
        """
        try:
            args = self._parse_arguments(action)
            content = await asyncio.wait_for(self._mcp_client.call_tool(action.type, args), timeout=timeout)
            logging.info(f"MCP tool {action.type} returned: {content}")

            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict) and "error" in parsed:
                    return ToolResult(tool_key=action.type, success=False, content=content)
            except (json.JSONDecodeError, TypeError):
                pass

            return ToolResult(tool_key=action.type, success=True, content=content)
        except Exception as exc:
            logging.error(f"Error calling {action.type}: {exc}")
            return ToolResult(
                tool_key=action.type,
                success=False,
                content=f"Error: {exc}",
            )

    def build_result_prompt(
        self,
        original_prompt: str,
        latest_results: List[ToolResult],
    ) -> str:
        """Build the follow-up prompt with the latest tool results.

        Parameters
        ----------
        original_prompt : str
            The initial user prompt.
        latest_results : List[ToolResult]
            Results from the most recent tool-execution round.

        Returns
        -------
        str
            Formatted recall prompt string.
        """
        lines = []
        for result in latest_results:
            status = "OK" if result.success else "FAILED"
            lines.append(f"[{result.tool_key}] {status}: {result.content}")
        result_block = "\n".join(lines)

        return (
            f"{original_prompt}\n\n"
            f"[Tool Results]\n{result_block}\n\n"
            f"[Next Step]\n"
            f"Do NOT re-call tools you have already called successfully. "
            f"If all needed info is available, respond with your final actions. "
            f"Otherwise call only the necessary tools in one batch.\n"
        )

    def build_call_signature(self, action: Any) -> str:
        """Build a signature for an action.

        Parameters
        ----------
        action : Any
            The action to build a signature for.

        Returns
        -------
        str
            Signature for the action.
        """
        args = self._parse_arguments(action)
        return f"{action.type}|{json.dumps(args, sort_keys=True, default=str)}"

    def _parse_arguments(self, action: Any) -> Dict[str, Any]:
        """Parse the action value for MCP tool args."""
        value = action.value
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, dict):
                    return parsed
            except (json.JSONDecodeError, TypeError):
                pass
            return {"action": value}
        return {"action": str(value)}

    async def stop(self) -> None:
        """Stop all MCP server connections."""
        await self._mcp_client.stop()
