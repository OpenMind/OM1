import asyncio
import functools
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
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
    """
    Manages the history of interactions for LLMs, including summarization.
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

        # history file path for persistence
        self._history_file_path = self._get_history_file_path()

        # load existing history from disk if available
        self._load_history()

    def _get_history_file_path(self) -> Path:
        """
        Get the path for the history file.

        Returns
        -------
        Path
            Path to the history file, defaults to ~/.om1/conversation_history_{agent_name}.json
        """
        om1_dir = Path.home() / ".om1"
        om1_dir.mkdir(parents=True, exist_ok=True)
        agent_suffix = f"_{self.agent_name}" if self.agent_name else ""
        return om1_dir / f"conversation_history{agent_suffix}.json"

    def _save_history(self) -> None:
        """
        Save the current conversation history to disk.

        Writes the history buffer and frame index to a JSON file.
        Errors are logged but don't interrupt the main flow.
        """
        try:
            data = {
                "agent_name": self.agent_name,
                "frame_index": self.frame_index,
                "history": [asdict(msg) for msg in self.history],
            }
            with open(self._history_file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logging.debug(f"Saved conversation history to {self._history_file_path}")
        except Exception as e:
            logging.error(f"Failed to save conversation history: {e}")

    def _load_history(self) -> None:
        """
        Load conversation history from disk if it exists.

        Restores the history buffer and frame index from a JSON file.
        Errors are logged but don't interrupt the main flow.
        """
        try:
            if not self._history_file_path.exists():
                logging.debug("No existing history file found, starting fresh")
                return

            with open(self._history_file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Restore history
            self.history = [
                ChatMessage(role=msg["role"], content=msg["content"])
                for msg in data.get("history", [])
            ]
            self.frame_index = data.get("frame_index", 0)

            logging.info(
                f"Loaded {len(self.history)} messages from {self._history_file_path}"
            )
        except Exception as e:
            logging.error(f"Failed to load conversation history: {e}")
            self.history = []
            self.frame_index = 0

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
                        messages.pop(0) if messages else None
                        messages.pop(0) if messages else None
                    else:
                        logging.warning(f"Unexpected summary result: {summary_message}")
                except asyncio.CancelledError:
                    logging.warning("Summary task callback cancelled")
                except Exception as e:
                    logging.error(
                        f"Error in summary task callback: {type(e).__name__}: {e}"
                    )
                    messages.pop(0) if messages else None
                    messages.pop(0) if messages else None

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

                self.history_manager.frame_index += 1

                # Persist history to disk after each update
                self.history_manager._save_history()

                return response

            return wrapper

        return decorator
