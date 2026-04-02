import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Set

from llm.output_model import CortexOutputModel
from mcp_servers.client import MCPClientManager


@dataclass
class ToolResult:
    """Result from a single MCP tool execution."""

    tool_key: str
    success: bool
    content: str


class MCPOrchestrator:
    """Orchestrate multi-round MCP tool execution.

    Manages the lifecycle of MCP tool calls within a single LLM tick,
    executing tools in batches and recalling the LLM with results until
    no more MCP actions are requested or the round limit is reached.

    Parameters
    ----------
    mcp_client : MCPClientManager
        The client manager for MCP server connections.
    llm : Any
        The LLM instance whose function_schemas will be extended.
    max_concurrency : int
        Maximum number of concurrent tool executions per round.
    """

    def __init__(
        self,
        mcp_client: MCPClientManager,
        llm: Any,
        max_concurrency: int = 5,
    ) -> None:
        """Initialize orchestrator and inject MCP tools into LLM schemas."""
        self._mcp_client = mcp_client
        self._max_concurrency = max_concurrency

        mcp_schemas = mcp_client.get_tool_schemas()
        base_schemas = [
            schema
            for schema in llm.function_schemas
            if not schema.get("function", {}).get("name", "").startswith("mcp_")
        ]
        llm.function_schemas = base_schemas + mcp_schemas

        logging.info(
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
        """Execute MCP tools in a multi-round loop.

        Extracts MCP actions from the LLM output, executes them
        concurrently, and recalls the LLM with results. Repeats until
        no MCP actions remain or ``max_rounds`` is reached.

        Parameters
        ----------
        output : Any
            The initial LLM output containing actions.
        prompt : str
            The original user prompt for LLM recall.
        llm : Any
            The LLM instance to recall with tool results.
        dispatch_om1 : callable, optional
            Dispatch non-MCP (OM1) actions immediately.
            Because MCP rounds are sometimes serially dependent,
            OM1 actions from earlier rounds cannot be carried
            over to the final output. This callback ensures
            they are dispatched.
        max_rounds : int
            Maximum number of tool-execution rounds.

        Returns
        -------
        Any
            Final LLM output with MCP actions removed.
        """
        if output is None or not hasattr(output, "actions"):
            return output

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

            # Dispatch OM1 actions and remove from output
            if om1_actions and dispatch_om1:
                await dispatch_om1(om1_actions)
                output.actions = [
                    action
                    for action in output.actions
                    if self._mcp_client.is_mcp_tool(action.type)
                ]

            logging.info(
                f"MCP round {round_idx + 1}/{max_rounds}: "
                f"executing {len(new_actions)} tool(s)"
            )

            results = await self._execute_tools(new_actions)

            for action, result in zip(new_actions, results):
                if result.success:
                    succeeded_calls.add(self._build_call_signature(action))

            output = await self._recall_llm(llm, prompt, results)

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
        """Return only the actions that target an MCP tool.

        Parameters
        ----------
        actions : list
            List of all actions from the LLM output.

        Returns
        -------
        list
            List of actions targeting MCP tools.
        """
        return [
            action for action in actions if self._mcp_client.is_mcp_tool(action.type)
        ]

    def _filter_new_actions(self, actions: list, succeeded: Set[str]) -> list:
        """Filter out actions whose call signature already succeeded.

        Parameters
        ----------
        actions : list
            List of MCP actions to filter.
        succeeded : Set[str]
            Set of already succeeded action signatures.

        Returns
        -------
        list
            List of new actions to execute.
        """
        return [
            action
            for action in actions
            if self._build_call_signature(action) not in succeeded
        ]

    def _build_call_signature(self, action: Any) -> str:
        """Deterministic signature for dedup: tool_key + sorted args.

        Parameters
        ----------
        action : Any
            The action object to build a signature for.

        Returns
        -------
        str
            Deterministic string signature of the action.
        """
        args = self._parse_arguments(action)
        return f"{action.type}|{json.dumps(args, sort_keys=True, default=str)}"

    def _parse_arguments(self, action: Any) -> Dict[str, Any]:
        """Parse the action value into a dict suitable for MCP tool args.

        Parameters
        ----------
        action : Any
            The action object containing arguments.

        Returns
        -------
        Dict[str, Any]
            Parsed dictionary of tool arguments.
        """
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
            The execution result containing success and content.
        """
        try:
            args = self._parse_arguments(action)
            content = await asyncio.wait_for(
                self._mcp_client.call_tool(action.type, args), timeout=timeout
            )
            logging.info(f"MCP tool {action.type} returned: {content}")

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
            logging.error(f"Error calling {action.type}: {exc}")
            return ToolResult(
                tool_key=action.type,
                success=False,
                content=f"Error: {exc}",
            )

    async def _execute_tools(self, actions: list) -> List[ToolResult]:
        """Execute multiple MCP tools concurrently with a semaphore.

        Parameters
        ----------
        actions : list
            List of new MCP actions to execute.

        Returns
        -------
        List[ToolResult]
            List of parsed tool execution results.
        """
        semaphore = asyncio.Semaphore(self._max_concurrency)

        async def _guarded(action: Any) -> ToolResult:
            async with semaphore:
                return await self._execute_single_tool(action)

        return await asyncio.gather(*(_guarded(action) for action in actions))

    def _build_result_prompt(
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
            List of recent tool execution results.

        Returns
        -------
        str
            The formatted follow-up prompt string.
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

    async def _recall_llm(
        self,
        llm: Any,
        prompt: str,
        latest_results: List[ToolResult],
    ) -> Any:
        """Recall LLM with the latest tool results.

        Parameters
        ----------
        llm : Any
            The LLM instance.
        prompt : str
            The original user prompt.
        latest_results : List[ToolResult]
            List of recent tool execution results.

        Returns
        -------
        Any
            The output object of the next round from the LLM.
        """
        recall_prompt = self._build_result_prompt(prompt, latest_results)
        logging.info("MCP recall LLM with latest results")
        return await llm.ask(recall_prompt)

    async def stop(self) -> None:
        """Stop all MCP server connections."""
        await self._mcp_client.stop()
