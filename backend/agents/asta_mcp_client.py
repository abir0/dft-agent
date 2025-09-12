"""Asta MCP client for scientific paper search."""

import asyncio
import logging
from typing import List, Optional

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from backend.settings import settings

logger = logging.getLogger(__name__)


class AstaMCPClient:
    """Client for connecting to Asta Scientific Corpus via MCP."""

    def __init__(self):
        self.client: Optional[MultiServerMCPClient] = None
        self.tools: List[BaseTool] = []
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the MCP client connection to Asta."""
        if self._initialized:
            return

        if not settings.ASTA_KEY:
            raise ValueError("ASTA_KEY not configured in environment variables")

        try:
            # Configure the MCP client for Asta
            self.client = MultiServerMCPClient(
                {
                    "asta": {
                        "transport": "streamable_http",
                        "url": "https://asta-tools.allen.ai/mcp/v1",
                        "headers": {"x-api-key": settings.ASTA_KEY.get_secret_value()},
                    }
                }
            )

            # Get available tools from the Asta MCP server
            self.tools = await self.client.get_tools()
            self._initialized = True

            logger.info(
                f"Successfully connected to Asta MCP server. Available tools: {len(self.tools)}"
            )
            for tool in self.tools:
                logger.info(f"  - {tool.name}: {tool.description}")

        except Exception as e:
            logger.error(f"Failed to initialize Asta MCP client: {e}")
            raise

    async def get_tools(self) -> List[BaseTool]:
        """Get the available Asta tools."""
        if not self._initialized:
            await self.initialize()
        return self.tools

    def get_tool_by_name(self, name: str) -> Optional[BaseTool]:
        """Get a specific tool by name."""
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None


# Global instance
_asta_client: Optional[AstaMCPClient] = None


async def get_asta_tools() -> List[BaseTool]:
    """Get Asta MCP tools for use in LangGraph agents.

    Returns:
        List of available Asta tools as LangChain BaseTool instances
    """
    global _asta_client

    if _asta_client is None:
        _asta_client = AstaMCPClient()

    return await _asta_client.get_tools()


async def get_specific_asta_tools(tool_names: List[str]) -> List[BaseTool]:
    """Get specific Asta tools by name.

    Args:
        tool_names: List of tool names to retrieve

    Returns:
        List of requested tools that were found
    """
    all_tools = await get_asta_tools()

    selected_tools = []
    for name in tool_names:
        for tool in all_tools:
            if tool.name == name:
                selected_tools.append(tool)
                break

    return selected_tools


# Helper function for synchronous contexts
def get_asta_tools_sync() -> List[BaseTool]:
    """Synchronous wrapper for getting Asta tools."""
    return asyncio.run(get_asta_tools())


if __name__ == "__main__":
    # Test the MCP client
    async def test():
        try:
            tools = await get_asta_tools()
            print(f"Successfully loaded {len(tools)} Asta tools:")
            for tool in tools:
                print(f"  - {tool.name}: {tool.description[:100]}...")
        except Exception as e:
            print(f"Error: {e}")

    asyncio.run(test())
