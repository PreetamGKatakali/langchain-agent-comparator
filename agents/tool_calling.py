"""
Tool-Calling Agent
──────────────────
Package  : langgraph.prebuilt.create_react_agent
Pattern  : Direct structured tool dispatch — no reasoning text, just action
Prompt   : Minimal system prompt — LLM calls tools immediately without
           writing out thoughts.

When to use:
  ✅ Production — fast, reliable, low token usage
  ✅ When you just need the answer, not the reasoning
  ✅ Chaining tools in a pipeline without verbose output
  ❌ Hard to debug — you won't see why the LLM chose a tool
"""

from langgraph.prebuilt import create_react_agent  # ← langgraph.prebuilt

TOOL_CALLING_SYSTEM_PROMPT = """You are a precise assistant.
Use the available tools directly to answer the question.
Do not explain your reasoning — just call the right tool and return the answer.
Be concise.
"""


class ToolCallingAgent:
    def __init__(self, llm, tools):
        self.graph = create_react_agent(
            model=llm,
            tools=tools,
            prompt=TOOL_CALLING_SYSTEM_PROMPT,
        )

    def invoke(self, inputs: dict, config: dict = None) -> dict:
        query = inputs.get("input", "")
        result = self.graph.invoke(
            {"messages": [("human", query)]},
            config=config or {},
        )
        messages = result.get("messages", [])
        final = messages[-1].content if messages else ""
        return {"output": final, "messages": messages}

    def stream(self, inputs: dict, config: dict = None):
        query = inputs.get("input", "")
        for chunk in self.graph.stream(
            {"messages": [("human", query)]},
            config=config or {},
            stream_mode="updates",
        ):
            yield chunk


def build_tool_calling_agent(llm, tools) -> ToolCallingAgent:
    return ToolCallingAgent(llm=llm, tools=tools)
