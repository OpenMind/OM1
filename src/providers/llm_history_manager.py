import json
import os
import aiofiles
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

        # io provider
        self.io_provider = IOProvider()
        self.history_file = os.path.join(os.getcwd(), "conversation_history.jsonl")
        if os.path.exists(self.history_file):
            try:
                max_history = getattr(self.config, 'history_length', 100) or 100
                messages = []
                with open(self.history_file, 'rb') as f:
                    f.seek(0, os.SEEK_END)
                    pointer = f.tell()
                    buffer = b''
                    while pointer > 0 and len(messages) < max_history:
                        read_size = min(pointer, 4096)
                        pointer -= read_size
                        f.seek(pointer)
                        chunk = f.read(read_size) + buffer
                        raw_lines = chunk.split(b'\n')
                        # 第一行可能是被切断的，存入buffer供下轮拼接
                        buffer = raw_lines[0]
                        # 后面的行是完整的，倒序解析
                        for l in reversed(raw_lines[1:]):
                            if l.strip():
                                data = json.loads(l.decode('utf-8'))
                                messages.insert(0, ChatMessage(role=data['role'], content=data['content']))
                                if len(messages) >= max_history: break
                # 处理文件开头剩余的部分
                if len(messages) < max_history and buffer.strip():
                    data = json.loads(buffer.decode('utf-8'))
                    messages.insert(0, ChatMessage(role=data['role'], content=data['content']))
                self.history.extend(messages)
                logging.info(f"Robust persistence load: Restored {len(messages)} messages.")
            except Exception as e:
                logging.error(f"Error loading history: {e}")
