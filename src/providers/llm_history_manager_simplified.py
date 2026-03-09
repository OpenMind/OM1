"""
Simplified LLM history manager for small local models.

Changes from the original LLMHistoryManager:
- Simplified ACTION_MAP without verbose preambles (e.g., "**** said: {}")
- Adds "greeting_conversation" to ACTION_MAP
- Conversation-focused input formatting ("User: ..." instead of sensor-style)
- Extracts plain text from JSON-wrapped action values ({"response": text})
- Prefixes summaries with "[Conversation summary - do not repeat]"
- Concise summarization prompts tuned for small models
"""

import functools
import json
import logging
from typing import Any, Awaitable, Callable, List, TypeVar

from llm import LLMConfig

from .llm_history_manager import ChatMessage, LLMHistoryManager

R = TypeVar("R")


ACTION_MAP_SIMPLIFIED = {
    "emotion": "{}",
    "speak": "{}",
    "move": "{}",
    "greeting_conversation": "{}",
}


class LLMHistoryManagerSimplified(LLMHistoryManager):
    """
    Simplified history manager for conversational small-model use cases.

    Inherits from LLMHistoryManager but overrides:
    - summarize_messages: adds "[Conversation summary - do not repeat]" prefix
    - update_history: uses "User: ..." format and simplified ACTION_MAP
    """

    def __init__(
        self,
        config: LLMConfig,
        client,
        system_prompt: str = (
            "You are a concise assistant that tracks conversation history for a "
            "robot named ****. Summarize ONLY what was said: what the user asked "
            "and what **** replied. Do NOT elaborate, add analysis, or invent "
            "details. Use plain short sentences, not tables or markdown."
        ),
        summary_command: str = (
            "\nWrite a brief summary of the conversation so far. List only what "
            "the user said and what **** replied. Keep it under 100 words. Do not "
            "repeat ****'s previous responses verbatim — just note the topic."
        ),
    ):
        super().__init__(config, client, system_prompt, summary_command)

    async def summarize_messages(self, messages: List[ChatMessage]) -> ChatMessage:
        """Summarize messages with a 'do not repeat' prefix."""
        result = await super().summarize_messages(messages)
        if result.role == "assistant" and not result.content.startswith(
            "[Conversation summary"
        ):
            result = ChatMessage(
                role="assistant",
                content=f"[Conversation summary - do not repeat] {result.content}",
            )
        return result

    @staticmethod
    def update_history() -> (
        Callable[[Callable[..., Awaitable[R]]], Callable[..., Awaitable[R]]]
    ):
        """
        Decorator to manage LLM history with simplified formatting.

        Uses "User: ..." input format and simplified ACTION_MAP.
        """

        def decorator(
            func: Callable[..., Awaitable[R]],
        ) -> Callable[..., Awaitable[R]]:
            @functools.wraps(func)
            async def wrapper(self: Any, prompt: str, *args: Any, **kwargs: Any) -> R:
                if getattr(self, "_skip_state_management", False):
                    return await func(self, prompt, *args, **kwargs)

                if self._config.history_length == 0:
                    response = await func(self, prompt, [], *args, **kwargs)
                    self.history_manager.frame_index += 1
                    return response

                self.agent_name = self._config.agent_name

                cycle = self.history_manager.frame_index
                logging.debug(f"LLM Tasking cycle debug tracker: {cycle}")

                current_tick = self.io_provider.tick_counter
                parts = []
                for input_type, input_info in self.io_provider.inputs.items():
                    if input_info.tick == current_tick:
                        logging.debug(f"LLM: {input_type} (tick #{input_info.tick})")
                        if input_info.input:
                            parts.append(input_info.input.strip())
                formatted_inputs = (
                    "User: " + " ".join(parts) if parts else "User: (no input)"
                )

                inputs = ChatMessage(role="user", content=formatted_inputs)

                logging.debug(f"Inputs: {inputs}")
                self.history_manager.history.append(inputs)

                messages = self.history_manager.get_messages()
                logging.debug(f"messages:\n{messages}")
                response = await func(self, prompt, messages, *args, **kwargs)
                logging.debug(f"Response to parse:\n{response}")

                if response is not None:

                    def _extract_text(value: str) -> str:
                        """Extract plain text from action value."""
                        try:
                            parsed = json.loads(value)
                            if isinstance(parsed, dict) and "response" in parsed:
                                return parsed["response"]
                        except (json.JSONDecodeError, TypeError):
                            pass
                        return value

                    actions_text = " | ".join(
                        ACTION_MAP_SIMPLIFIED[action.type.lower()].format(
                            _extract_text(action.value) if action.value else ""
                        )
                        for action in response.actions  # type: ignore
                        if action.type.lower() in ACTION_MAP_SIMPLIFIED
                    )
                    action_message = (
                        f"{self.agent_name}: {actions_text}"
                        if actions_text
                        else f"{self.agent_name}: (no response)"
                    )

                    self.history_manager.history.append(
                        ChatMessage(role="assistant", content=action_message)
                    )

                    if (
                        self.history_manager.config.history_length > 0
                        and len(self.history_manager.history)
                        > self.history_manager.config.history_length
                    ):
                        await self.history_manager.start_summary_task(
                            self.history_manager.history
                        )
                else:
                    if (
                        self.history_manager.history
                        and self.history_manager.history[-1].role == "user"
                    ):
                        logging.warning(
                            "LLM response failed, removing unpaired user message"
                        )
                        self.history_manager.history.pop()

                self.history_manager.frame_index += 1

                return response

            return wrapper

        return decorator
