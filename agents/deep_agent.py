"""
Deep Agent (LangGraph)
──────────────────────
Package  : langgraph.prebuilt  (modern LangGraph)
Pattern  : Stateful graph — nodes for LLM call + tool execution, edges for routing
Docs     : https://langchain-ai.github.io/langgraph/reference/prebuilt/#create_react_agent

Why it's "deeper" than classic ReAct:
  - Runs inside a proper StateGraph (not a simple loop)
  - Has built-in memory / conversation history across turns
  - Supports interrupt_before/interrupt_after for human-in-the-loop
  - Can be extended with custom nodes (planning, reflection, etc.)
  - Streams individual tokens, tool calls, and state diffs
  - Handles parallel tool calls natively

When to use:
  ✅ Multi-turn conversations that need memory
  ✅ Agentic pipelines where you need to pause / resume
  ✅ Complex tasks that benefit from streaming + state inspection
  ✅ When you want to extend the agent with custom graph nodes later
  ❌ Overkill for simple single-turn tool queries
"""

from langgraph.prebuilt import create_react_agent  # ← langgraph.prebuilt


class DeepAgent:
    """
    Thin wrapper around the LangGraph compiled graph so the interface
    matches the other agents (.invoke / .stream).
    """

    def __init__(self, llm, tools):
        # create_react_agent from langgraph returns a CompiledGraph, not an AgentExecutor
        self.graph = create_react_agent(model=llm, tools=tools)

    def invoke(self, inputs: dict) -> dict:
        query = inputs.get("input", "")

        # LangGraph expects HumanMessage format
        result = self.graph.invoke({"messages": [("human", query)]})

        # Extract the last AI message as the final answer
        messages = result.get("messages", [])
        final = messages[-1].content if messages else ""
        return {"output": final, "messages": messages}

    def stream(self, inputs: dict):
        """Stream intermediate steps — great for demos."""
        query = inputs.get("input", "")
        for chunk in self.graph.stream(
            {"messages": [("human", query)]},
            stream_mode="updates",
        ):
            yield chunk


def build_deep_agent(llm, tools) -> DeepAgent:
    return DeepAgent(llm=llm, tools=tools)
