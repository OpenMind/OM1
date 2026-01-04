import asyncio
import functools
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Awaitable, Callable, List, Optional, TypeVar, Union, Protocol

import openai

from llm import LLMConfig
from .io_provider import IOProvider

R = TypeVar("R")


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class ChatMessage:
    role: MessageRole
    content: str

    def to_openai(self) -> dict:
        return {"role": self.role.value, "content": self.content}


class ActionLike(Protocol):
    type: str
    value: Optional[str]


class ResponseWithActions(Protocol):
    actions: List[ActionLike]


ACTION_MAP = {
    "emotion": "**** felt: {}.",
    "speak": "**** said: {}",
    "move": "**** performed this motion: {}.",
}


class LLMHistoryManager:
    def __init__(
        self,
        config: LLMConfig,
        client: Union[openai.AsyncClient, openai.OpenAI],
        system_prompt: str = (
            "You are a helpful assistant that summarizes a succession of events "
            "and interactions accurately and concisely. You are watching a robot "
            "named **** interact with people and the world."
        ),
        summary_command: str = (
            "\nConsidering the new information, write an updated summary of the "
            "situation for ****."
        ),
    ):
        self.client = client
        self.config = config

        self.agent_name: str = self.config.agent_name or "the agent"

        self.system_prompt: str = system_prompt.replace(
            "****", self.agent_name
        )
        self.summary_command: str = summary_command.replace(
            "****", self.agent_name
        )

        self.frame_index: int = 0
        self.history: List[ChatMessage] = []
        self._summary_task: Optional[asyncio.Task] = None
        self._history_lock = asyncio.Lock()

        self.io_provider = IOProvider()

    async def _call_llm(self, prompt: str) -> Optional[str]:
        try:
            response = await self.client.chat.completions.create(  # type: ignore
                model=self.config.model or "gpt-4o-mini",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
            )

            if not response or not response.choices:
                return None

            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"LLM error: {type(e).__name__}: {e}")
            return None

    async def summarize_messages(
        self, messages: List[ChatMessage]
    ) -> Optional[ChatMessage]:
        if not messages:
            return None

        summary_prompt = "\n".join(m.content for m in messages)
        summary_prompt += self.summary_command

        summary = await self._call_llm(summary_prompt)
        if summary is None:
            return None

        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content=f"Previously, {summary}",
        )

    async def start_summary_task(self) -> None:
        async with self._history_lock:
            if self._summary_task and not self._summary_task.done():
                return

            snapshot = list(self.history)

        async def runner() -> None:
            summary = await self.summarize_messages(snapshot)
            if summary is None:
                return

            async with self._history_lock:
                self.history.clear()
                self.history.append(summary)

        self._summary_task = asyncio.create_task(runner())

    def get_messages(self) -> List[dict]:
        return [msg.to_openai() for msg in self.history]

    @staticmethod
    def update_history() -> (
        Callable[[Callable[..., Awaitable[R]]], Callable[..., Awaitable[R]]]
    ):
        def decorator(func: Callable[..., Awaitable[R]]) -> Callable[..., Awaitable[R]]:
            @functools.wraps(func)
            async def wrapper(self: Any, prompt: str, *args: Any, **kwargs: Any) -> R:
                if getattr(self, "_skip_state_management", False):
                    return await func(self, prompt, *args, **kwargs)

                history_manager: LLMHistoryManager = self.history_manager
                config = history_manager.config

                history_length: int = config.history_length or 0

                if history_length == 0:
                    response = await func(self, prompt, [], *args, **kwargs)
                    history_manager.frame_index += 1
                    return response

                current_tick = history_manager.io_provider.tick_counter
                agent_name = history_manager.agent_name

                sensed = f"{agent_name} sensed the following: "
                for input_type, info in history_manager.io_provider.inputs.items():
                    if info.tick == current_tick:
                        sensed += f"{input_type}. {info.input} | "

                sensed = sensed.replace("  ", " ").replace("..", ".")

                async with history_manager._history_lock:
                    history_manager.history.append(
                        ChatMessage(
                            role=MessageRole.USER,
                            content=sensed,
                        )
                    )
                    messages = history_manager.get_messages()

                response = await func(self, prompt, messages, *args, **kwargs)

                if response is not None and hasattr(response, "actions"):
                    resp = response  # type: ignore[assignment]

                    actions: List[str] = []
                    for action in getattr(resp, "actions", []):
                        action_type = getattr(action, "type", "").lower()
                        action_value = getattr(action, "value", "") or ""
                        if action_type in ACTION_MAP:
                            actions.append(
                                ACTION_MAP[action_type].format(action_value)
                            )

                    if actions:
                        action_message = (
                            "Given that information, **** took these actions: "
                            + " | ".join(actions)
                        ).replace("****", agent_name)

                        async with history_manager._history_lock:
                            history_manager.history.append(
                                ChatMessage(
                                    role=MessageRole.ASSISTANT,
                                    content=action_message,
                                )
                            )

                            if len(history_manager.history) > history_length:
                                await history_manager.start_summary_task()

                history_manager.frame_index += 1
                return response

            return wrapper

        return decorator
