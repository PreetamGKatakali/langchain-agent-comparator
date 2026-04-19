"""
main.py — Single entry point.

Change ONE line in config/agent_config.yaml:
    agent:
      type: "react"        # ← Thought/Action/Observation loop  (langchain.agents)
      type: "deep_agent"   # ← Stateful LangGraph agent         (langgraph.prebuilt)
      type: "tool_calling" # ← Structured JSON tool dispatch    (langchain.agents)

Usage:
    python main.py                                         # interactive prompt
    python main.py --query "What is the weather in Delhi?"
    python main.py --stream                                # stream output (deep_agent only)
    python main.py --benchmark                             # compare all agent types
"""

import argparse
import yaml
from pathlib import Path
from agents import load_agent, AGENT_DESCRIPTIONS
from tools import load_tools
from llm import get_llm
from logger import ToolLogger
CONFIG_PATH = Path(__file__).parent / "config" / "agent_config.yaml"


def main():
    parser = argparse.ArgumentParser(description="LangChain Agent Switcher")
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--stream", action="store_true", help="Stream output (deep_agent only)")
    parser.add_argument("--benchmark", action="store_true", help="Run all agents on all test queries")
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    agent_type = cfg["agent"]["type"]
    model      = cfg["agent"]["llm"]["model"]
    temperature = cfg["agent"]["llm"]["temperature"]

    print(f"\n{'='*55}")
    print(f"  Agent Type  : {agent_type.upper()}")
    print(f"  Tools       : {cfg['tools']}")
    print(f"  Package     : {AGENT_DESCRIPTIONS[agent_type]}")
    print(f"{'='*55}")

    if args.benchmark:
        from benchmarks.runner import run_benchmark
        run_benchmark()
        return

    llm   = get_llm(model=model, temperature=temperature)
    tools = load_tools(cfg["tools"])
    agent = load_agent(agent_type, llm=llm, tools=tools)

    query = args.query or input("\nEnter your query: ").strip()
    print(f"\nQuery: {query}\n")

    tool_logger = ToolLogger()
    run_config = {"callbacks": [tool_logger]}

    if args.stream and hasattr(agent, "stream"):
        print("[Streaming output...]\n")
        for chunk in agent.stream({"input": query}, config=run_config):
            for node, update in chunk.items():
                messages = update.get("messages", [])
                for msg in messages:
                    print(f"[{node}] {msg.content}")
        tool_logger.print_summary()
        return

    response = agent.invoke({"input": query}, config=run_config)
    print(f"\n{'='*55}")
    print("FINAL ANSWER:")
    print(response.get("output", str(response)))
    tool_logger.print_summary()


if __name__ == "__main__":
    main()
