import asyncio
import logging
import time
from typing import Optional

import aiohttp
from pydantic import Field

from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput
from providers.io_provider import IOProvider


class NewsConfig(SensorConfig):
    """
    Configuration for News Input.

    Parameters
    ----------
    api_key : str
        NewsAPI.org API key.
    country : str
        Country code for headlines (e.g., 'us', 'gb', 'tr').
    category : str
        News category: business, entertainment, general, health, science, sports, technology.
    query : str
        Optional search query to filter news.
    max_headlines : int
        Maximum number of headlines to return.
    poll_interval : float
        Seconds between news updates (default: 900 = 15 minutes).
    """

    api_key: str = Field(description="NewsAPI.org API key")
    country: str = Field(default="us", description="Country code for headlines")
    category: str = Field(default="general", description="News category")
    query: str = Field(default="", description="Optional search query")
    max_headlines: int = Field(default=5, description="Maximum headlines to return")
    poll_interval: float = Field(
        default=900.0, description="Seconds between news updates"
    )


class NewsInput(FuserInput[NewsConfig, Optional[dict]]):
    """
    News input that fetches current headlines from NewsAPI.org.

    Provides real-time news awareness to give the robot knowledge
    of current events and trending topics.
    """

    def __init__(self, config: NewsConfig):
        """
        Initialize the News input.

        Parameters
        ----------
        config : NewsConfig
            Configuration for the news sensor.
        """
        super().__init__(config)

        self.io_provider = IOProvider()
        self.messages: list[Message] = []
        self.descriptor_for_LLM = "Current News"

        self.api_key = config.api_key
        self.country = config.country
        self.category = config.category
        self.query = config.query
        self.max_headlines = config.max_headlines
        self.poll_interval = config.poll_interval

        self._last_poll_time: float = 0

        if not self.api_key:
            logging.warning("NewsInput: API key not provided")

    async def _fetch_news(self) -> Optional[dict]:
        """
        Fetch news data from NewsAPI.org.

        Returns
        -------
        Optional[dict]
            News data or None if request failed.
        """
        if not self.api_key:
            return None

        url = "https://newsapi.org/v2/top-headlines"
        params = {
            "apiKey": self.api_key,
            "country": self.country,
            "category": self.category,
            "pageSize": self.max_headlines,
        }

        if self.query:
            params["q"] = self.query

        try:
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("status") == "ok":
                            logging.debug(
                                f"NewsInput: Fetched {len(data.get('articles', []))} articles"
                            )
                            return data
                        else:
                            logging.error(
                                f"NewsInput: API error: {data.get('message')}"
                            )
                            return None
                    else:
                        error_text = await response.text()
                        logging.error(
                            f"NewsInput: API error {response.status}: {error_text}"
                        )
                        return None
        except asyncio.TimeoutError:
            logging.error("NewsInput: Request timed out")
            return None
        except aiohttp.ClientError as e:
            logging.error(f"NewsInput: Network error: {e}")
            return None
        except Exception as e:
            logging.error(f"NewsInput: Unexpected error: {e}")
            return None

    async def _poll(self) -> Optional[dict]:
        """
        Poll for news data based on poll_interval.

        Returns
        -------
        Optional[dict]
            Fresh news data when poll interval has elapsed, None otherwise.
        """
        current_time = time.time()

        if current_time - self._last_poll_time < self.poll_interval:
            await asyncio.sleep(1.0)
            return None

        self._last_poll_time = current_time
        await asyncio.sleep(1.0)
        return await self._fetch_news()

    async def _raw_to_text(self, raw_input: Optional[dict]) -> Optional[Message]:
        """
        Convert raw news data to human-readable text.

        Parameters
        ----------
        raw_input : Optional[dict]
            Raw news data from API.

        Returns
        -------
        Optional[Message]
            Formatted news message or None.
        """
        if raw_input is None:
            return None

        try:
            articles = raw_input.get("articles", [])

            if not articles:
                return None

            headlines = []
            for i, article in enumerate(articles[: self.max_headlines], 1):
                title = article.get("title", "").strip()
                if title and title != "[Removed]":
                    source = article.get("source", {}).get("name", "Unknown")
                    headlines.append(f"{i}) {title} ({source})")

            if not headlines:
                return None

            message = "Top headlines: " + " ".join(headlines)
            return Message(timestamp=time.time(), message=message)

        except Exception as e:
            logging.error(f"NewsInput: Error parsing news data: {e}")
            return None

    async def raw_to_text(self, raw_input: Optional[dict]):
        """
        Update message buffer with processed news data.

        Parameters
        ----------
        raw_input : Optional[dict]
            Raw news data to be processed.
        """
        pending_message = await self._raw_to_text(raw_input)

        if pending_message is not None:
            self.messages.append(pending_message)

    def formatted_latest_buffer(self) -> Optional[str]:
        """
        Format and clear the latest buffer contents.

        Returns
        -------
        Optional[str]
            Formatted string for LLM or None if buffer is empty.
        """
        if len(self.messages) == 0:
            return None

        latest_message = self.messages[-1]

        result = (
            f"\nINPUT: {self.descriptor_for_LLM}\n// START\n"
            f"{latest_message.message}\n// END\n"
        )

        self.io_provider.add_input(
            self.descriptor_for_LLM, latest_message.message, latest_message.timestamp
        )
        self.messages = []

        return result
