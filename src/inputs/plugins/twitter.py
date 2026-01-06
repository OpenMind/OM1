import asyncio
import logging
from queue import Empty, Queue
from typing import AsyncIterator, List, Optional

import aiohttp
from pydantic import Field

from inputs.base import SensorConfig
from inputs.base.loop import FuserInput


class TwitterSensorConfig(SensorConfig):
    """
    Configuration for Twitter Sensor.

    Parameters
    ----------
    query : str
        Query to search for on Twitter.
    """

    query: str = Field(
        default="What's new in AI and technology?",
        description="Query to search for on Twitter",
    )


class TwitterInput(FuserInput[TwitterSensorConfig, Optional[str]]):
    """Context query input handler for RAG."""

    def __init__(
        self,
        config: Optional[TwitterSensorConfig],
    ):
        """Initialize TwitterInput with configuration.

        Parameters
        ----------
        config : Optional[TwitterSensorConfig]
            Configuration object from the runtime
        """
        if config is None:
            config = TwitterSensorConfig()

        super().__init__(config)

        self.buffer: List[str] = []
        self.message_buffer: Queue[str] = Queue()
        self.api_url = "https://api.openmind.org/api/core/query"
        self.session: Optional[aiohttp.ClientSession] = None
        self.context: Optional[str] = None

        # Use getattr instead of .get() since config is an object, not a dict
        self.query = self.config.query

    async def __aenter__(self):
        """Async context manager entry."""
        await self._init_session()
        return self

    async def __aexit__(self, _exc_type, _exc_val, _exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()

    async def _init_session(self):
        """Initialize aiohttp session if not exists."""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=10)
            self.session = aiohttp.ClientSession(timeout=timeout)

    async def _query_context(self, query: str):
        """Perform context query to RAG endpoint."""
        await self._init_session()

        try:
            async with self.session.post(  # type: ignore
                self.api_url,
                json={"query": query},
                headers={"Content-Type": "application/json"},
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    if "results" in data:
                        documents = data["results"]
                        context = "\n\n".join(
                            [
                                r.get("content", {}).get("text", "")
                                for r in documents
                                if r.get("content", {}).get("text", "")
                            ]
                        )
                        self.context = context
                        self.buffer = [context]  # Replace buffer with context
                else:
                    error_text = await response.text()
                    logging.error(
                        f"Query failed with status {response.status}: {error_text}"
                    )

        except Exception as e:
            logging.error(f"Error querying context: {str(e)}")

    async def raw_to_text(self, raw_input: Optional[str] = None):
        """Convert raw input to text format and add to buffer.

        Parameters
        ----------
        raw_input : Optional[str]
            Raw input to process. If None, process from message buffer.

        Returns
        -------
        str
            The processed text
        """
        if raw_input:
            self.message_buffer.put_nowait(raw_input)

        if self.message_buffer:
            try:
                message = self.message_buffer.get_nowait()
                if message:
                    self.buffer.append(message)
                    logging.debug(f"Added to buffer: {message}")
                    return message
            except Empty:
                pass

        return ""

    async def start(self):
        """Start the input handler with initial query."""
        await self._query_context(self.query)
        self.message_buffer.put_nowait(self.query)

    async def listen(self) -> AsyncIterator[str]:
        """Listen for new messages."""
        await self.start()

        while True:
            message = await self._poll()
            if message:
                yield message
            await asyncio.sleep(0.1)

    async def _poll(self) -> Optional[str]:
        """Poll for new messages."""
        await asyncio.sleep(0.5)
        try:
            message = self.message_buffer.get_nowait()
            return message
        except Empty:
            return None

    def formatted_latest_buffer(self) -> Optional[str]:
        """Format and return the context."""
        content = (
            self.context if self.context else (self.buffer[-1] if self.buffer else None)
        )

        if not content:
            return None

        result = f"""
TwitterInput CONTEXT
// START
{content}
// END
"""
        return result

    async def initialize_with_query(self, query: str):
        """Initialize with a query."""
        logging.info(f"[TwitterInput] Initializing with query: {query}")
        self.message_buffer.put_nowait(query)  # Add query to message buffer
        await self._query_context(query)  # Immediately get context
