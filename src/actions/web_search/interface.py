from dataclasses import dataclass

from actions.base import Interface


@dataclass
class WebSearchInput:
    """
    Input interface for the Web Search action.

    Parameters
    ----------
    action : str
        The search query to look up on the web.
        Example: "latest SpaceX launch date"
    """

    action: str = ""


@dataclass
class WebSearchOutput:
    """
    Output interface for the Web Search action.

    Parameters
    ----------
    results : str
        Formatted search results as text.
    """

    results: str = ""


@dataclass
class WebSearch(Interface[WebSearchInput, WebSearchOutput]):
    """
    This action allows the robot to search the web using DuckDuckGo.

    Effect: Performs a web search with the given query and returns
    relevant results. The results can be used by the LLM to answer
    questions about current events or unknown topics.
    """

    input: WebSearchInput
    output: WebSearchOutput
