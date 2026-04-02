import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Set

from llm.output_model import CortexOutputModel
from skills.loader import SkillEntry, SkillLoader

READ_SKILL_TOOL = "read_skill"


def build_tool_schema(skills: Dict[str, SkillEntry]) -> dict:
    """Generate the function-calling schema for ``read_skill``.

    Embeds skill names and descriptions directly in the schema
    description — same pattern as MCP tools being self-describing.

    Parameters
    ----------
    skills : dict
        Skill entries keyed by name.

    Returns
    -------
    dict
        OpenAI-compatible function schema.
    """
    # Build skill list for the description
    skill_items = []
    for skill in sorted(skills.values(), key=lambda s: -s.priority):
        skill_items.append(f"{skill.name} ({skill.description})")
    skill_list = ", ".join(skill_items)

    return {
        "type": "function",
        "function": {
            "name": READ_SKILL_TOOL,
            "description": (
                "Read a skill's full instructions before executing it. "
                "Call this ONCE when a task matches an available skill. "
                f"Available skills: {skill_list}"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": "Name of the skill to read",
                        "enum": list(skills.keys()),
                    }
                },
                "required": ["skill_name"],
            },
        },
    }


@dataclass
class SkillProcessResult:
    """Result from SkillOrchestrator.process()."""

    output: Any
    augmented_prompt: str


class SkillOrchestrator:
    """Multi-round orchestrator for read_skill tool calls.

    Loads skills from ``skills/``, filters by whitelist,
    and registers the ``read_skill`` function schema on the LLM.

    Parameters
    ----------
    skill_names : list
        Whitelisted skill names for this mode.
    llm : Any
        The LLM instance whose function_schemas will be extended.
    """

    SKILLS_DIR = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "../../skills")
    )

    def __init__(self, skill_names: List[str], llm: Any) -> None:
        loader = SkillLoader(self.SKILLS_DIR)
        self._skills = {
            name: entry for name, entry in loader.skills.items() if name in skill_names
        }

        missing = set(skill_names) - set(self._skills.keys())
        if missing:
            logging.warning(f"Skills not found on disk: {missing}")

        if not self._skills:
            logging.warning("No valid skills loaded")
            return

        schema = build_tool_schema(self._skills)
        llm.function_schemas = [
            s
            for s in llm.function_schemas
            if s.get("function", {}).get("name") != READ_SKILL_TOOL
        ]
        llm.function_schemas.append(schema)

        logging.info(
            f"SkillOrchestrator initialized with "
            f"{len(self._skills)} skill(s): {list(self._skills.keys())}"
        )

    async def process(
        self,
        output: Any,
        prompt: str,
        llm: Any,
        dispatch_om1=None,
        max_rounds: int = 3,
    ) -> SkillProcessResult:
        """Execute read_skill calls in a multi-round loop.

        Parameters
        ----------
        output : Any
            The initial LLM output containing actions.
        prompt : str
            The original user prompt for LLM recall.
        llm : Any
            The LLM instance to recall with tool results.
        dispatch_om1 : callable, optional
            Callback to dispatch true OM1 actions immediately.
        max_rounds : int
            Maximum rounds for read_skill calls (typically 1-2).

        Returns
        -------
        SkillProcessResult
            Contains the final output and the augmented prompt
            (original + skill instructions) for downstream use.
        """
        if output is None or not hasattr(output, "actions"):
            return SkillProcessResult(output=output, augmented_prompt=prompt)

        succeeded_calls: Set[str] = set()
        current_prompt = prompt

        for round_idx in range(max_rounds):
            # Extract read_skill actions only
            skill_actions = [a for a in output.actions if a.type == READ_SKILL_TOOL]
            if not skill_actions:
                break

            # Filter out already-succeeded calls
            new_actions = [
                a
                for a in skill_actions
                if self._call_signature(a) not in succeeded_calls
            ]
            if not new_actions:
                break

            # Dispatch OM1 actions
            om1_only_actions = [
                a
                for a in output.actions
                if a.type != READ_SKILL_TOOL and not a.type.startswith("mcp_")
            ]
            if om1_only_actions and dispatch_om1:
                await dispatch_om1(om1_only_actions)

            output.actions = [
                a
                for a in output.actions
                if a.type == READ_SKILL_TOOL or a.type.startswith("mcp_")
            ]

            logging.info(
                f"Skill round {round_idx + 1}/{max_rounds}: "
                f"executing {len(new_actions)} read_skill call(s)"
            )

            # Execute read_skill calls
            results = []
            for action in new_actions:
                result = self._execute_read_skill(action)
                results.append(result)
                if result["success"]:
                    succeeded_calls.add(self._call_signature(action))

            # Recall LLM with skill instructions
            current_prompt = self._build_recall_prompt(prompt, results)
            output = await llm.ask(current_prompt)
            if output is None or not hasattr(output, "actions"):
                return SkillProcessResult(output=None, augmented_prompt=current_prompt)

        # Remove any remaining read_skill actions
        if output and hasattr(output, "actions"):
            final_actions = [a for a in output.actions if a.type != READ_SKILL_TOOL]
            output = CortexOutputModel(actions=final_actions)

        return SkillProcessResult(output=output, augmented_prompt=current_prompt)

    def _execute_read_skill(self, action: Any) -> dict:
        """Execute a single read_skill call."""
        try:
            value = action.value
            if isinstance(value, str):
                try:
                    parsed = json.loads(value)
                    if isinstance(parsed, dict):
                        value = parsed
                except (json.JSONDecodeError, TypeError):
                    value = {"skill_name": value}
            if isinstance(value, dict):
                skill_name = value.get("skill_name", "")
            else:
                skill_name = str(value)

            skill = self._skills.get(skill_name)
            if not skill:
                available = ", ".join(self._skills.keys())
                content = (
                    f"Error: Skill '{skill_name}' not found. "
                    f"Available skills: {available}"
                )
                return {
                    "tool_key": READ_SKILL_TOOL,
                    "success": False,
                    "content": content,
                }

            logging.info(f"read_skill: loaded instructions for '{skill_name}'")
            return {
                "tool_key": READ_SKILL_TOOL,
                "success": True,
                "content": skill.instructions,
            }
        except Exception as exc:
            logging.error(f"Error in read_skill: {exc}")
            return {
                "tool_key": READ_SKILL_TOOL,
                "success": False,
                "content": f"Error reading skill: {exc}",
            }

    def _call_signature(self, action: Any) -> str:
        """Build a dedup signature for an action."""
        value = action.value
        if isinstance(value, dict):
            return f"{action.type}|{json.dumps(value, sort_keys=True)}"
        return f"{action.type}|{value}"

    def _build_recall_prompt(self, original_prompt: str, results: List[dict]) -> str:
        """Build recall prompt that includes skill instructions.

        This prompt is also passed to downstream MCPOrchestrator
        so that MCP recall rounds retain the skill context.
        """
        lines = []
        for r in results:
            status = "OK" if r["success"] else "FAILED"
            lines.append(f"[{r['tool_key']}] {status}: {r['content']}")
        result_block = "\n".join(lines)

        return (
            f"{original_prompt}\n\n"
            f"[Skill Instructions]\n{result_block}\n\n"
            f"[Next Step]\n"
            f"Follow the skill instructions above. "
            f"Use speak to report progress at each step. "
            f"Call only the necessary tools.\n"
        )
