import asyncio
import logging
import time
from typing import Optional

import aiohttp
from pydantic import Field

from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput
from providers.io_provider import IOProvider


class TwitterSensorConfig(SensorConfig):
    """Configuration for TwitterInput."""

    query: str = Field(
        default="What's new in AI and technology?",
        description="Query to search for on Twitter",
    )
    poll_interval: float = Field(
        default=60.0,
        description="Seconds between API polls",
    )


class TwitterInput(FuserInput[TwitterSensorConfig, Optional[dict]]):
    """RAG-based context input from OpenMind knowledge base."""

    def __init__(self, config: Optional[TwitterSensorConfig] = None):
        if config is None:
            config = TwitterSensorConfig()

        super().__init__(config)

        self.io_provider = IOProvider()
        self.messages: list[Message] = []
        self.descriptor_for_LLM = "TwitterInput CONTEXT"
        self.api_url = "https://api.openmind.org/api/core/query"
        self.query = self.config.query
        self.poll_interval = self.config.poll_interval
        self._last_poll_time: float = 0
        self.session: Optional[aiohttp.ClientSession] = None

    async def _init_session(self):
        if self.session is None:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10)
            )

    async def _poll(self) -> Optional[dict]:
        current_time = time.time()

        if current_time - self._last_poll_time < self.poll_interval:
            await asyncio.sleep(1.0)
            return None

        self._last_poll_time = current_time
        await self._init_session()

        if self.session is None:
            return None

        try:
            async with self.session.post(
                self.api_url,
                json={"query": self.query},
                headers={"Content-Type": "application/json"},
            ) as response:
                if response.status == 200:
                    return await response.json()
                error_text = await response.text()
                logging.error(
                    f"TwitterInput: API error {response.status}: {error_text}"
                )
                return None
        except asyncio.TimeoutError:
            logging.error("TwitterInput: Request timed out")
            return None
        except aiohttp.ClientError as e:
            logging.error(f"TwitterInput: Network error: {e}")
            return None
        except Exception as e:
            logging.error(f"TwitterInput: Unexpected error: {e}")
            return None

    async def _raw_to_text(self, raw_input: Optional[dict]) -> Optional[Message]:
        if raw_input is None:
            return None

        try:
            documents = raw_input.get("results", [])
            context = "\n\n".join(
                r.get("content", {}).get("text", "")
                for r in documents
                if r.get("content", {}).get("text", "")
            )

            if not context:
                return None

            return Message(timestamp=time.time(), message=context)

        except Exception as e:
            logging.error(f"TwitterInput: Error parsing response: {e}")
            return None

    async def raw_to_text(self, raw_input: Optional[dict]):
        """Process raw input and append to message buffer."""
        pending_message = await self._raw_to_text(raw_input)
        if pending_message is not None:
            self.messages.append(pending_message)

    def formatted_latest_buffer(self) -> Optional[str]:
        """Return latest message formatted for LLM and clear buffer."""
        if not self.messages:
            return None

        latest = self.messages[-1]
        result = (
            f"\nINPUT: {self.descriptor_for_LLM}\n// START\n"
            f"{latest.message}\n// END\n"
        )

        self.io_provider.add_input(
            self.descriptor_for_LLM, latest.message, latest.timestamp
        )
        self.messages = []
        return result
