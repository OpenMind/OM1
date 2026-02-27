import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List

from llm.output_model import CortexOutputModel
from mcp_servers.client import MCPClientManager

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Result from a single MCP tool execution."""

    tool_key: str
    success: bool
    content: str


class MCPOrchestrator:
    """Orchestrate MCP tool execution between LLM and action dispatch.

    Intercepts MCP tool calls, executes them, and re-calls the LLM with results.

    Parameters
    ----------
    mcp_client : MCPClientManager
        Connected MCP client with tool schemas.
    llm : Any
        The LLM instance (used to inject tool schemas).
    max_concurrency : int
        Maximum number of MCP tools to execute concurrently.
    """

    def __init__(
        self,
        mcp_client: MCPClientManager,
        llm: Any,
        max_concurrency: int = 5,
    ):
        self._mcp_client = mcp_client
        self._max_concurrency = max_concurrency

        mcp_schemas = mcp_client.get_tool_schemas()
        llm.function_schemas.extend(mcp_schemas)
        logger.info(f"MCPOrchestrator initialized with {len(mcp_schemas)} tools")

    async def process(self, output: Any, prompt: str, llm: Any) -> Any:
        """Process LLM output, execute MCP tools if needed.

        Parameters
        ----------
        output : CortexOutputModel
            The LLM's output containing actions.
        prompt : str
            The original prompt (used for re-calling LLM).
        llm : Any
            The LLM instance for follow-up inference.

        Returns
        -------
        CortexOutputModel
            Final output with merged actions.
        """
        if output is None or not hasattr(output, "actions"):
            return output

        mcp_actions = self._get_mcp_actions(output.actions)

        if not mcp_actions:
            return output

        # Preserve OM1 actions to avoid actions loss
        om1_actions = [
            a for a in output.actions if not self._mcp_client.is_mcp_tool(a.type)
        ]

        logger.info(
            f"MCP: executing {len(mcp_actions)} tool(s), preserving {len(om1_actions)} OM1 action(s)"
        )

        results = await self._execute_tools(mcp_actions)
        second_output = await self._recall_llm(llm, prompt, results)

        if second_output is None or not hasattr(second_output, "actions"):
            return CortexOutputModel(actions=om1_actions) if om1_actions else output

        merged = om1_actions + second_output.actions
        return CortexOutputModel(actions=merged)

    def _get_mcp_actions(self, actions: list) -> list:
        """Extract MCP tool calls from action list."""
        return [a for a in actions if self._mcp_client.is_mcp_tool(a.type)]

    def _parse_arguments(self, action: Any) -> Dict[str, Any]:
        """Extract tool arguments from an action's value."""
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

    async def _execute_single_tool(self, action: Any) -> ToolResult:
        """Execute a single MCP tool call with error handling."""
        try:
            args = self._parse_arguments(action)
            content = await self._mcp_client.call_tool(action.type, args)
            logger.info(f"MCP tool {action.type} returned: {content}")
            return ToolResult(tool_key=action.type, success=True, content=content)
        except Exception as e:
            logger.error(f"Error calling {action.type}: {e}")
            return ToolResult(
                tool_key=action.type,
                success=False,
                content=f"Error: {e}",
            )

    async def _execute_tools(self, actions: list) -> List[ToolResult]:
        """Execute multiple MCP tools concurrently."""
        semaphore = asyncio.Semaphore(self._max_concurrency)

        async def _guarded(action: Any) -> ToolResult:
            async with semaphore:
                return await self._execute_single_tool(action)

        return await asyncio.gather(*(_guarded(a) for a in actions))

    def _build_result_prompt(
        self, original_prompt: str, results: List[ToolResult]
    ) -> str:
        """Build a follow-up prompt that includes tool results."""
        lines = []
        for r in results:
            status = "OK" if r.success else "FAILED"
            lines.append(f"[{r.tool_key}] ({status}): {r.content}")

        result_block = "\n".join(lines)
        return (
            f"{original_prompt}\n\n"
            f"TOOL RESULTS:\n{result_block}\n\n"
            f"Based on the tool results above, respond using the speak action "
            f"to tell the user the information. Summarize concisely."
        )

    async def _recall_llm(
        self, llm: Any, prompt: str, results: List[ToolResult]
    ) -> Any:
        """Re-call the LLM with tool results to generate the final response."""
        new_prompt = self._build_result_prompt(prompt, results)
        logger.info("MCP execution complete, recall LLM")
        return await llm.ask(new_prompt)

    async def close(self) -> None:
        """Close underlying MCP server connections."""
        await self._mcp_client.close_all()
