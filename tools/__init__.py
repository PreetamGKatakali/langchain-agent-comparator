from tools.search import get_search_tool
from tools.calculator import get_calculator_tool
from tools.weather import get_weather_tool

TOOL_REGISTRY = {
    "search": get_search_tool,
    "calculator": get_calculator_tool,
    "weather": get_weather_tool,
}


def load_tools(tool_names: list[str]) -> list:
    tools = []
    for name in tool_names:
        if name not in TOOL_REGISTRY:
            raise ValueError(f"Unknown tool: '{name}'. Available: {list(TOOL_REGISTRY.keys())}")
        tools.append(TOOL_REGISTRY[name]())
    return tools
