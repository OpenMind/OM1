import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Set

from llm.output_model import CortexOutputModel
from mcp_servers.client import MCPClientManager

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Result from a single MCP tool execution."""

    tool_key: str
    success: bool
    content: str


@dataclass
class RoundRecord:
    """Record of a single orchestration round."""

    round_num: int
    tools_called: List[str]
    results: List[ToolResult]


class MCPOrchestrator:
    """Orchestrate multi-round MCP tool execution."""

    def __init__(
        self,
        mcp_client: MCPClientManager,
        llm: Any,
        max_concurrency: int = 5,
    ):
        self._mcp_client = mcp_client
        self._max_concurrency = max_concurrency

        mcp_schemas = mcp_client.get_tool_schemas()
        base_schemas = [
            schema
            for schema in llm.function_schemas
            if not schema.get("function", {}).get("name", "").startswith("mcp_")
        ]
        llm.function_schemas = base_schemas + mcp_schemas

        logger.info(
            f"MCP Orchestrator initialized with {len(mcp_schemas)} MCP tools, "
            f"{len(base_schemas)} base tools"
        )

    async def process(
        self,
        output: Any,
        prompt: str,
        llm: Any,
        dispatch_om1=None,
        max_rounds: int = 5,
    ) -> Any:
        """Execute MCP tools in multi-round loop."""
        if output is None or not hasattr(output, "actions"):
            return output

        history: List[RoundRecord] = []
        succeeded_calls: Set[str] = set()

        for round_idx in range(max_rounds):
            # Extract MCP actions from output
            mcp_actions = self._extract_mcp_actions(output.actions)
            if not mcp_actions:
                break

            # Filter out duplicate actions in all rounds
            new_actions = self._filter_new_actions(mcp_actions, succeeded_calls)
            if not new_actions:
                break

            # Extract OM1 actions from output
            om1_actions = [
                action
                for action in output.actions
                if not self._mcp_client.is_mcp_tool(action.type)
            ]

            # Start OM1 actions
            if om1_actions and dispatch_om1:
                await dispatch_om1(om1_actions)

            logger.info(
                f"MCP round {round_idx + 1}/{max_rounds}: "
                f"executing {len(new_actions)} tool(s)"
            )

            results = await self._execute_tools(new_actions)

            for action, result in zip(new_actions, results):
                if result.success:
                    succeeded_calls.add(self._build_call_signature(action))

            history.append(
                RoundRecord(
                    round_num=round_idx + 1,
                    tools_called=[action.type for action in new_actions],
                    results=results,
                )
            )

            output = await self._recall_llm(llm, prompt, history)

            if output is None or not hasattr(output, "actions"):
                return None

        # If there are still mcp actions in the output after max_rounds, remove them
        if output and hasattr(output, "actions"):
            final_actions = [
                action
                for action in output.actions
                if not action.type.startswith("mcp_")
            ]
            return CortexOutputModel(actions=final_actions)
        return output

    def _extract_mcp_actions(self, actions: list) -> list:
        return [
            action for action in actions if self._mcp_client.is_mcp_tool(action.type)
        ]

    def _filter_new_actions(self, actions: list, succeeded: Set[str]) -> list:
        return [
            action
            for action in actions
            if self._build_call_signature(action) not in succeeded
        ]

    def _build_call_signature(self, action: Any) -> str:
        """Deterministic signature for dedup: tool_key + sorted args."""
        args = self._parse_arguments(action)
        return f"{action.type}|{json.dumps(args, sort_keys=True, default=str)}"

    def _parse_arguments(self, action: Any) -> Dict[str, Any]:
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

    async def _execute_single_tool(
        self, action: Any, timeout: float = 10.0
    ) -> ToolResult:
        try:
            args = self._parse_arguments(action)
            content = await asyncio.wait_for(
                self._mcp_client.call_tool(action.type, args), timeout=timeout
            )
            logger.info(f"MCP tool {action.type} returned: {content}")

            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict) and "error" in parsed:
                    return ToolResult(
                        tool_key=action.type, success=False, content=content
                    )
            except (json.JSONDecodeError, TypeError):
                pass

            return ToolResult(tool_key=action.type, success=True, content=content)
        except Exception as exc:
            logger.error(f"Error calling {action.type}: {exc}")
            return ToolResult(
                tool_key=action.type,
                success=False,
                content=f"Error: {exc}",
            )

    async def _execute_tools(self, actions: list) -> List[ToolResult]:
        semaphore = asyncio.Semaphore(self._max_concurrency)

        async def _guarded(action: Any) -> ToolResult:
            async with semaphore:
                return await self._execute_single_tool(action)

        return await asyncio.gather(*(_guarded(action) for action in actions))

    def _build_result_prompt(
        self,
        original_prompt: str,
        history: List[RoundRecord],
    ) -> str:
        """Build follow-up prompt."""
        # Tool results: concise, structured
        lines = []
        for record in history:
            for result in record.results:
                status = "OK" if result.success else "FAILED"
                lines.append(f"[{result.tool_key}] {status}: {result.content}")
        result_block = "\n".join(lines)

        return (
            f"{original_prompt}\n\n"
            f"[Tool Results]\n{result_block}\n\n"
            f"[Next Step]\n"
            f"Do NOT re-call any tool marked OK above. "
            f"If all needed info is available, respond with speak. "
            f"Otherwise call only the necessary tools in one batch.\n"
        )

    async def _recall_llm(
        self,
        llm: Any,
        prompt: str,
        history: List[RoundRecord],
    ) -> Any:
        """Recall LLM with tool results. Skips history to avoid pollution."""
        recall_prompt = self._build_result_prompt(prompt, history)
        logger.info("MCP recall LLM with cumulative context")
        llm._skip_state_management = True
        try:
            return await llm.ask(recall_prompt)
        finally:
            llm._skip_state_management = False

    async def close(self) -> None:
        """Close all MCP client connections."""
        await self._mcp_client.close_all()
