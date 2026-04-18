from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun


def get_search_tool():
    """DuckDuckGo web search — no API key required."""
    search = DuckDuckGoSearchRun()
    return search
