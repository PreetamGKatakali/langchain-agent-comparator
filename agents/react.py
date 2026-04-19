"""
ReAct Agent
───────────
Package  : langgraph.prebuilt.create_react_agent
Pattern  : Thought → Action → Observation → repeat
Prompt   : System prompt that forces the LLM to reason step-by-step
           before calling any tool.

When to use:
  ✅ Tasks where you want the LLM to think aloud before acting
  ✅ Debugging — you can see every reasoning step
  ✅ Open-ended Q&A with multiple tool calls
  ❌ Slower — more tokens due to reasoning text
"""

from langgraph.prebuilt import create_react_agent  # ← langgraph.prebuilt

REACT_SYSTEM_PROMPT = """You are a helpful assistant that reasons step by step.

Before calling any tool, always:
1. Think about what the question is asking
2. Decide which tool is needed and why
3. Call the tool with the right input
4. Observe the result and decide if more steps are needed

Be explicit about your reasoning at every step.
"""


class ReactAgent:
    def __init__(self, llm, tools):
        self.graph = create_react_agent(
            model=llm,
            tools=tools,
            prompt=REACT_SYSTEM_PROMPT,
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


def build_react_agent(llm, tools) -> ReactAgent:
    return ReactAgent(llm=llm, tools=tools)
