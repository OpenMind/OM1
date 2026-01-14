import logging
from typing import Optional

from duckduckgo_search import DDGS
from pydantic import Field

from actions.base import ActionConfig, ActionConnector
from actions.web_search.interface import WebSearchInput
from providers.io_provider import IOProvider


class DuckDuckGoConfig(ActionConfig):
    """
    Configuration class for DuckDuckGo Web Search Connector.

    Parameters
    ----------
    max_results : int
        Maximum number of search results to return.
    region : str
        Region for search results (e.g., 'wt-wt' for worldwide, 'us-en' for US).
    safesearch : str
        Safe search setting: 'on', 'moderate', or 'off'.
    """

    max_results: int = Field(default=5, description="Maximum search results to return")
    region: str = Field(default="wt-wt", description="Region for search results")
    safesearch: str = Field(default="moderate", description="Safe search setting")


class DuckDuckGoConnector(ActionConnector[DuckDuckGoConfig, WebSearchInput]):
    """
    Connector for DuckDuckGo Web Search.

    This connector allows the robot to search the web using DuckDuckGo,
    providing access to current information without requiring an API key.
    """

    def __init__(self, config: DuckDuckGoConfig):
        """
        Initialize the DuckDuckGo connector.

        Parameters
        ----------
        config : DuckDuckGoConfig
            Configuration object for the connector.
        """
        super().__init__(config)
        self.io_provider = IOProvider()
        self._last_results: Optional[str] = None

    def _format_results(self, results: list) -> str:
        """
        Format search results into readable text.

        Parameters
        ----------
        results : list
            List of search result dictionaries.

        Returns
        -------
        str
            Formatted search results as text.
        """
        if not results:
            return "No results found."

        formatted = []
        for i, result in enumerate(results, 1):
            title = result.get("title", "No title")
            body = result.get("body", "No description")
            href = result.get("href", "")
            formatted.append(f"{i}. {title}\n   {body}\n   URL: {href}")

        return "\n\n".join(formatted)

    async def connect(self, output_interface: WebSearchInput) -> None:
        """
        Perform web search using DuckDuckGo.

        Parameters
        ----------
        output_interface : WebSearchInput
            The WebSearchInput interface containing the search query.
        """
        query = output_interface.action

        if not query:
            logging.warning("WebSearch: Empty search query provided")
            return

        try:
            logging.info(f"WebSearch: Searching for '{query}'")

            with DDGS() as ddgs:
                results = list(
                    ddgs.text(
                        query,
                        region=self.config.region,
                        safesearch=self.config.safesearch,
                        max_results=self.config.max_results,
                    )
                )

            formatted_results = self._format_results(results)
            self._last_results = formatted_results

            logging.info(f"WebSearch: Found {len(results)} results")
            logging.debug(f"WebSearch results:\n{formatted_results}")

            self.io_provider.add_input(
                "WebSearch", f"Query: {query}\n\n{formatted_results}", 0
            )

        except Exception as e:
            logging.error(f"WebSearch: Search failed: {str(e)}")
            self._last_results = f"Search failed: {str(e)}"

    def get_last_results(self) -> Optional[str]:
        """
        Get the results from the last search.

        Returns
        -------
        Optional[str]
            The formatted results from the last search, or None.
        """
        return self._last_results
