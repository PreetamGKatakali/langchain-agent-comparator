"""
ReAct Agent
───────────
Package  : langchain.agents  (classic LangChain)
Pattern  : Thought → Action → Observation → repeat
Docs     : https://python.langchain.com/docs/how_to/agent_executor/

When to use:
  ✅ Step-by-step tasks — needs to think aloud before acting
  ✅ Open-ended Q&A with tool use
  ✅ When you want full transparency of the reasoning chain
  ✅ Works with any LLM (no function-calling support needed)
  ❌ Verbose — generates a lot of text per step
  ❌ Fragile if LLM doesn't follow Thought/Action format strictly
"""

from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent  # ← langchain.agents


def build_react_agent(llm, tools) -> AgentExecutor:
    # Standard hwchase17/react prompt — defines the Thought/Action/Observation format
    prompt = hub.pull("hwchase17/react")

    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=10,
        return_intermediate_steps=True,
    )
