"""Mock MCP server with hardcoded weather tools for integration testing."""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("weather")


@mcp.tool()
def get_location(city: str) -> str:
    """Return the canonical location identifier for a city name."""
    return "San Francisco, CA"


@mcp.tool()
def get_weather(location: str) -> str:
    """Return the current weather for a location."""
    return "72°F, sunny"


if __name__ == "__main__":
    mcp.run()
