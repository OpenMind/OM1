import asyncio
import functools
import logging
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
        self._history_lock = asyncio.Lock()

        # io provider
        self.io_provider = IOProvider()

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
                summary_prompt += f"{messages[0].content}\n"
                summary_prompt += "\nNow, the following new information has arrived. "
                summary_prompt += f"{messages[2].content}\n"
                summary_prompt += f"{messages[3].content}\n"
            else:
                for msg in messages:
                    summary_prompt += f"{msg.content}\n"

            summary_prompt += self.summary_command
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
        """
        if not messages:
            logging.warning("No messages to summarize in start_summary_task")
            return

        try:
            if self._summary_task and not self._summary_task.done():
                logging.info("Previous summary task still running")
                return

            async with self._history_lock:
                messages_copy = self.history.copy()
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

                        async def update_history():
                            async with self._history_lock:
                                del self.history[:num_summarized]
                                self.history.insert(0, summary_message)
                            logging.info("Successfully summarized the state")

                        asyncio.run_coroutine_threadsafe(
                            update_history(), asyncio.get_event_loop()
                        )
                    elif (
                        summary_message.role == "system"
                        and "Error" in summary_message.content
                    ):
                        logging.error(
                            f"Summarization failed: {summary_message.content}"
                        )
                    else:
                        logging.warning(f"Unexpected summary result: {summary_message}")
                except Exception as e:
                    logging.error(
                        f"Error in summary task callback: {type(e).__name__}: {e}"
                    )

            self._summary_task.add_done_callback(callback)

        except Exception as e:
            logging.error(f"Error starting summary task: {type(e).__name__}: {e}")

    async def get_messages(self) -> List[dict]:
        """
        Get messages in format required by OpenAI API.
        """
        async with self._history_lock:
            return [{"role": msg.role, "content": msg.content} for msg in self.history]

    async def add_message(self, message: ChatMessage):
        """Safely append a message to history."""
        async with self._history_lock:
            self.history.append(message)

    async def clear_history(self):
        """Safely clear the history."""
        async with self._history_lock:
            self.history.clear()

    @staticmethod
    def update_history() -> (
        Callable[[Callable[..., Awaitable[R]]], Callable[..., Awaitable[R]]]
    ):
        """
        Decorator to manage LLM history around an async function.
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
                        formatted_inputs += f"{input_type}. {input_info.input} | "

                formatted_inputs = formatted_inputs.replace("..", ".")
                formatted_inputs = formatted_inputs.replace("  ", " ")

                inputs = ChatMessage(role="user", content=formatted_inputs)
                logging.debug(f"Inputs: {inputs}")

                await self.history_manager.add_message(inputs)

                messages = await self.history_manager.get_messages()
                logging.debug(f"messages:\n{messages}")

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

                    await self.history_manager.add_message(
                        ChatMessage(role="assistant", content=action_message)
                    )

                    async with self.history_manager._history_lock:
                        should_summarize = (
                            self.history_manager.config.history_length > 0
                            and len(self.history_manager.history)
                            > self.history_manager.config.history_length
                        )

                    if should_summarize:
                        async with self.history_manager._history_lock:
                            hist_copy = self.history_manager.history.copy()
                        await self.history_manager.start_summary_task(hist_copy)

                self.history_manager.frame_index += 1
                return response

            return wrapper

        return decorator
