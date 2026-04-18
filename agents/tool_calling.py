"""
Tool-Calling Agent
──────────────────
Package  : langchain.agents  (create_tool_calling_agent)
Pattern  : LLM emits structured JSON tool calls — no free-text parsing
Docs     : https://python.langchain.com/docs/how_to/agent_executor/

Why it's different from ReAct:
  - No Thought/Action text — LLM directly outputs a structured tool call
  - Faster: skips the text-parsing step entirely
  - More reliable tool dispatch (validated schema, not regex)
  - Requires a model that supports function/tool calling (GPT-4o, Claude, Gemini)

When to use:
  ✅ Production systems needing reliable, schema-validated tool invocation
  ✅ When tool inputs must be strongly typed
  ✅ Faster response when reasoning transparency is not a priority
  ❌ Less visible reasoning — harder to debug why the LLM chose a tool
  ❌ Won't work with models that don't support function calling
"""

from langchain.agents import AgentExecutor, create_tool_calling_agent  # ← langchain.agents
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


def build_tool_calling_agent(llm, tools) -> AgentExecutor:
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a helpful assistant with access to tools. "
            "Use them whenever needed to give accurate, grounded answers."
        )),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=10,
        return_intermediate_steps=True,
    )
