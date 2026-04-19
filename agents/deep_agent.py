"""
Deep Agent
──────────
Package  : langgraph.prebuilt.create_react_agent
Pattern  : Plan first → execute each step — deeper multi-phase reasoning
Prompt   : System prompt that forces full upfront planning before tool use.

When to use:
  ✅ Complex multi-step tasks (research + calculate + summarize)
  ✅ When you want the agent to map out the full plan before acting
  ✅ Multi-turn conversations (stateful graph keeps message history)
  ✅ Supports streaming and interrupt/resume (human-in-the-loop)
  ❌ Overkill for simple single-tool queries
"""

from langgraph.prebuilt import create_react_agent  # ← langgraph.prebuilt

DEEP_AGENT_SYSTEM_PROMPT = """You are a deep reasoning assistant.

When given a task:
1. First, lay out a complete plan — list every step needed to answer the question
2. Then execute each step one by one using the available tools
3. After all steps are done, synthesize the results into a final answer

Always think about the full picture before taking any action.
If a step produces unexpected results, revise your plan before continuing.
"""


class DeepAgent:
    def __init__(self, llm, tools):
        self.graph = create_react_agent(
            model=llm,
            tools=tools,
            prompt=DEEP_AGENT_SYSTEM_PROMPT,
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


def build_deep_agent(llm, tools) -> DeepAgent:
    return DeepAgent(llm=llm, tools=tools)
