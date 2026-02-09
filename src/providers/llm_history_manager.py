import asyncio
import functools
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, List, Optional, TypeVar, Union

import openai

from llm import LLMConfig

from .io_provider import IOProvider

R = TypeVar("R")


@dataclass
class ChatMessage:
    """
    Represents a chat message with role and content.
    """

    role: str
    content: str


ACTION_MAP = {
    "emotion": "**** felt: {}.",
    "speak": "**** said: {}",
    "move": "**** performed this motion: {}.",
}


class LLMHistoryManager:
    """Manages the history of interactions for LLMs.

    Includes:
    - in-memory history buffer
    - optional summarization
    - optional persistence to disk (Issue #985)
    """

    def __init__(
        self,
        config: LLMConfig,
        client: Union[openai.AsyncClient, openai.OpenAI],
        system_prompt: str = "You are a helpful assistant that summarizes a succession of events and interactions accurately and concisely. You are watching a robot named **** interact with people and the world. Your goal is to help **** remember what the robot felt, saw, and heard, and how the robot responded to those inputs.",
        summary_command: str = "\nConsidering the new information, write an updated summary of the situation for ****. Emphasize information that **** needs to know to respond to people and situations in the best possible and most compelling way.",
    ):
        """
        Initialize the LLMHistoryManager.

        Parameters
        ----------
        config : LLMConfig
            Configuration object containing LLM settings and parameters.
        client : Union[openai.AsyncClient, openai.OpenAI]
            OpenAI client instance for making API calls (async or sync).
        system_prompt : str, optional
            System prompt template for summarization. Defaults to a prompt
            that describes the assistant's role in summarizing robot interactions.
            The string "****" will be replaced with the agent name.
        summary_command : str, optional
            Command template appended to messages when requesting summaries.
            Defaults to a command asking for an updated situation summary.
            The string "****" will be replaced with the agent name.
        """
        self.client = client

        # configuration
        self.config = config
        self.agent_name = self.config.agent_name
        self.system_prompt = (
            system_prompt.replace("****", self.agent_name)
            if self.agent_name
            else system_prompt
        )
        self.summary_command = (
            summary_command.replace("****", self.agent_name)
            if self.agent_name
            else summary_command
        )

        # frame index
        self.frame_index = 0

        # task executor
        self._summary_task: Optional[asyncio.Task] = None

        # history buffer
        self.history: List[ChatMessage] = []

        # io provider
        self.io_provider = IOProvider()

        # Optional history persistence
        self._persistence_enabled = bool(
            getattr(self.config, "history_persistence_enabled", False)
        )
        self._persistence_path = str(
            getattr(self.config, "history_persistence_path", "~/.openmind/history")
        )
        if self._persistence_enabled:
            self._load_history_from_disk()

    def _safe_agent_filename(self) -> str:
        agent = (self.agent_name or "agent").strip() or "agent"
        # replace anything non alnum/dash/underscore/dot with underscore
        agent = re.sub(r"[^A-Za-z0-9._-]+", "_", agent)
        return f"{agent}.json"

    def _history_file_path(self) -> str:
        base_dir = os.path.expanduser(self._persistence_path)
        return os.path.join(base_dir, self._safe_agent_filename())

    def _load_history_from_disk(self) -> None:
        """Load conversation history from disk if present.

        This is best-effort: on any error/corruption, we log and start fresh.
        """
        try:
            base_dir = os.path.expanduser(self._persistence_path)
            os.makedirs(base_dir, mode=0o755, exist_ok=True)

            path = self._history_file_path()
            if not os.path.exists(path):
                return

            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, list):
                logging.warning("History persistence: invalid format; expected list")
                return

            loaded: List[ChatMessage] = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                role = item.get("role")
                content = item.get("content")
                if isinstance(role, str) and isinstance(content, str):
                    loaded.append(ChatMessage(role=role, content=content))

            self.history = loaded

            # Enforce history_length after load
            target_length = getattr(self.config, "history_length", None)
            if isinstance(target_length, int) and target_length >= 0:
                if target_length == 0:
                    self.history = []
                elif len(self.history) > target_length:
                    self.history = self.history[-target_length:]

            logging.info(
                f"History persistence: loaded {len(self.history)} messages from {path}"
            )
        except Exception as e:
            logging.warning(
                f"History persistence: failed to load history ({type(e).__name__}: {e}); starting fresh"
            )
            self.history = []

    def _save_history_to_disk(self) -> None:
        """Persist conversation history to disk (atomic best-effort)."""
        try:
            if not self._persistence_enabled:
                return

            base_dir = os.path.expanduser(self._persistence_path)
            os.makedirs(base_dir, mode=0o755, exist_ok=True)

            path = self._history_file_path()
            tmp = path + ".tmp"

            data = [{"role": m.role, "content": m.content} for m in self.history]

            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp, path)
        except Exception as e:
            logging.warning(
                f"History persistence: failed to save history ({type(e).__name__}: {e})"
            )

    async def summarize_messages(self, messages: List[ChatMessage]) -> ChatMessage:
        """
        Summarize a list of messages using the OpenAI API.

        Parameters
        ----------
        messages : List[ChatMessage]
            List of chat messages to summarize.

        Returns
        -------
        ChatMessage
            A new message containing the summary with role "assistant" or
            "system" (in case of errors).

        Raises
        ------
        asyncio.TimeoutError
            If the API request times out.
        openai.APIError
            If there's an error with the OpenAI API.
        """
        # Set timeout for API call
        timeout = 10.0  # seconds

        try:
            if not messages:
                logging.warning("No messages to summarize")
                return ChatMessage(role="system", content="No history to summarize")

            logging.debug(f"All raw info: {messages} len{len(messages)}")

            summary_prompt = ""

            if len(messages) == 4:
                # the normal case - previous summary and new data
                # the previous summary
                summary_prompt += f"{messages[0].content}\n"
                # actions - already part of the summary - no need to add
                # summary_prompt += f"{messages[1].content}\n"
                summary_prompt += "\nNow, the following new information has arrived. "
                summary_prompt += f"{messages[2].content}\n"
                summary_prompt += f"{messages[3].content}\n"
            else:
                for msg in messages:
                    summary_prompt += f"{msg.content}\n"

            summary_prompt += self.summary_command

            # insert actual robot name
            summary_prompt = (
                summary_prompt.replace("****", self.agent_name)
                if self.agent_name
                else summary_prompt
            )

            logging.info(f"Information to summarize:\n{summary_prompt}")

            api_kwargs = {
                "model": self.config.model or "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": summary_prompt},
                ],
            }

            if isinstance(self.client, openai.AsyncClient):
                response = await asyncio.wait_for(
                    self.client.chat.completions.create(**api_kwargs),
                    timeout=timeout,
                )
            else:
                loop = asyncio.get_running_loop()
                response = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        functools.partial(
                            self.client.chat.completions.create, **api_kwargs
                        ),
                    ),
                    timeout=timeout,
                )

            if not response or not response.choices:
                logging.error("Invalid API response format")
                return ChatMessage(
                    role="system", content="Error: Received invalid response from API"
                )

            summary = response.choices[0].message.content
            if summary is None:
                logging.error("Received empty summary from API")
                return ChatMessage(
                    role="system", content="Error: Received empty summary from API"
                )
            return ChatMessage(role="assistant", content=f"Previously, {summary}")

        except asyncio.TimeoutError:
            logging.error(f"API request timed out after {timeout} seconds")
            return ChatMessage(role="system", content="Error: API request timed out")
        except openai.APIError as e:
            logging.error(f"OpenAI API error: {e}")
            return ChatMessage(
                role="system", content=f"Error: API service unavailable: {str(e)}"
            )
        except Exception as e:
            logging.error(f"Error summarizing messages: {type(e).__name__}: {e}")
            return ChatMessage(role="system", content="Error summarizing state")

    async def start_summary_task(self, messages: List[ChatMessage]):
        """
        Start a new asynchronous task to summarize the messages.

        Parameters
        ----------
        messages : List[ChatMessage]
            List of chat messages to summarize. This list will be modified
            in-place when the summary task completes successfully.

        Notes
        -----
        If a previous summary task is still running, this method will return
        early without starting a new task. The summary result will be added
        to the messages list via a callback when the task completes.
        """
        if not messages:
            logging.warning("No messages to summarize in start_summary_task")
            return

        try:
            if self._summary_task and not self._summary_task.done():
                logging.info("Previous summary task still running")
                return

            messages_copy = messages.copy()
            num_summarized = len(messages_copy)
            self._summary_task = asyncio.create_task(
                self.summarize_messages(messages_copy)
            )

            def callback(task):
                try:
                    if task.cancelled():
                        logging.warning("Summary task was cancelled")
                        return

                    summary_message = task.result()
                    if summary_message.role == "assistant":
                        del messages[:num_summarized]
                        messages.insert(0, summary_message)
                        logging.info("Successfully summarized the state")
                    elif (
                        summary_message.role == "system"
                        and "Error" in summary_message.content
                    ):
                        logging.error(
                            f"Summarization failed: {summary_message.content}"
                        )
                        target_length = self.config.history_length
                        if target_length is not None and len(messages) > target_length:
                            excess = len(messages) - target_length
                            del messages[:excess]
                            logging.warning(
                                f"Truncated {excess} oldest messages to maintain history_length={target_length}"
                            )
                    else:
                        logging.warning(f"Unexpected summary result: {summary_message}")
                except Exception as e:
                    logging.error(
                        f"Error in summary task callback: {type(e).__name__}: {e}"
                    )
                    target_length = self.config.history_length
                    if target_length is not None and len(messages) > target_length:
                        excess = len(messages) - target_length
                        del messages[:excess]
                        logging.warning(
                            f"Truncated {excess} oldest messages after exception"
                        )

            self._summary_task.add_done_callback(callback)

        except asyncio.CancelledError:
            logging.warning("Summary task creation cancelled")
        except Exception as e:
            logging.error(f"Error starting summary task: {type(e).__name__}: {e}")
            messages.pop(0) if messages else None
            messages.pop(0) if messages else None

    def get_messages(self) -> List[dict]:
        """
        Get messages in format required by OpenAI API.

        Returns
        -------
        List[dict]
            List of message dictionaries with "role" and "content" keys,
            formatted for OpenAI API consumption.
        """
        return [{"role": msg.role, "content": msg.content} for msg in self.history]

    @staticmethod
    def update_history() -> (
        Callable[[Callable[..., Awaitable[R]]], Callable[..., Awaitable[R]]]
    ):
        """
        Decorator to manage LLM history around an async function.

        Returns
        -------
        Callable
            Decorator function.
        """

        def decorator(func: Callable[..., Awaitable[R]]) -> Callable[..., Awaitable[R]]:
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
                formatted_inputs = f"{self.agent_name} sensed the following: "
                for input_type, input_info in self.io_provider.inputs.items():
                    if input_info.tick == current_tick:
                        logging.debug(f"LLM: {input_type} (tick #{input_info.tick})")
                        logging.debug(f"LLM: {input_info}")
                        formatted_inputs += f"{input_type}. {input_info.input} | "

                formatted_inputs = formatted_inputs.replace("..", ".")
                formatted_inputs = formatted_inputs.replace("  ", " ")

                inputs = ChatMessage(role="user", content=formatted_inputs)

                logging.debug(f"Inputs: {inputs}")
                self.history_manager.history.append(inputs)

                messages = self.history_manager.get_messages()
                logging.debug(f"messages:\n{messages}")
                # this advances the frame index
                response = await func(self, prompt, messages, *args, **kwargs)
                logging.debug(f"Response to parse:\n{response}")

                if response is not None:

                    action_message = (
                        "Given that information, **** took these actions: "
                        + (
                            " | ".join(
                                ACTION_MAP[action.type.lower()].format(
                                    action.value if action.value else ""
                                )
                                for action in response.actions  # type: ignore
                                if action.type.lower() in ACTION_MAP
                            )
                        )
                    )

                    action_message = action_message.replace("****", self.agent_name)

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

                    # Persist history after mutation (best-effort)
                    if getattr(self.history_manager, "_persistence_enabled", False):
                        self.history_manager._save_history_to_disk()
                else:
                    if (
                        self.history_manager.history
                        and self.history_manager.history[-1].role == "user"
                    ):
                        logging.warning(
                            "LLM response failed, removing unpaired user message"
                        )
                        self.history_manager.history.pop()

                    # Persist history even on failure cleanup (best-effort)
                    if getattr(self.history_manager, "_persistence_enabled", False):
                        self.history_manager._save_history_to_disk()

                self.history_manager.frame_index += 1

                return response

            return wrapper

        return decorator
