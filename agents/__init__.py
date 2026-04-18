from agents.react import build_react_agent           # langchain.agents
from agents.deep_agent import build_deep_agent       # langgraph.prebuilt
from agents.tool_calling import build_tool_calling_agent  # langchain.agents

AGENT_REGISTRY = {
    "react": build_react_agent,
    "deep_agent": build_deep_agent,
    "tool_calling": build_tool_calling_agent,
}

AGENT_DESCRIPTIONS = {
    "react": "Classic ReAct — Thought/Action/Observation loop (langchain.agents)",
    "deep_agent": "LangGraph stateful agent — graph-based, memory, streaming (langgraph.prebuilt)",
    "tool_calling": "Structured tool calling — schema-validated, no text parsing (langchain.agents)",
}


def load_agent(agent_type: str, llm, tools):
    if agent_type not in AGENT_REGISTRY:
        raise ValueError(
            f"Unknown agent type: '{agent_type}'.\n"
            f"Available types: {list(AGENT_REGISTRY.keys())}"
        )
    print(f"  → {AGENT_DESCRIPTIONS[agent_type]}")
    return AGENT_REGISTRY[agent_type](llm=llm, tools=tools)
