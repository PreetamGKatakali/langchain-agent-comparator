"""
Benchmark runner — measures time, token usage, and step count per agent type.
Run directly:  python -m benchmarks.runner
"""

import time
import yaml
from pathlib import Path
from langchain_openai import ChatOpenAI

from agents import load_agent
from tools import load_tools


CONFIG_PATH = Path(__file__).parent.parent / "config" / "agent_config.yaml"


def run_benchmark():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    tools = load_tools(cfg["tools"])
    queries = cfg["benchmark"]["test_queries"]
    agent_types = ["react", "plan_execute", "tool_calling"]

    results = {}

    for agent_type in agent_types:
        print(f"\n{'='*60}")
        print(f"  Agent: {agent_type.upper()}")
        print(f"{'='*60}")

        llm = ChatOpenAI(
            model=cfg["agent"]["llm"]["model"],
            temperature=cfg["agent"]["llm"]["temperature"],
        )
        agent = load_agent(agent_type, llm=llm, tools=tools)
        agent_results = []

        for query in queries:
            print(f"\n  Query: {query[:80]}...")
            start = time.perf_counter()
            try:
                response = agent.invoke({"input": query})
                elapsed = time.perf_counter() - start
                output = response.get("output", str(response))
                agent_results.append({
                    "query": query,
                    "output": output[:200],
                    "time_sec": round(elapsed, 2),
                    "status": "ok",
                })
                print(f"  ✓ Done in {elapsed:.2f}s")
            except Exception as e:
                elapsed = time.perf_counter() - start
                agent_results.append({
                    "query": query,
                    "error": str(e),
                    "time_sec": round(elapsed, 2),
                    "status": "error",
                })
                print(f"  ✗ Error: {e}")

        results[agent_type] = agent_results

    _print_summary(results)
    return results


def _print_summary(results: dict):
    print(f"\n{'='*60}")
    print("  BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"{'Agent':<20} {'Avg Time (s)':<15} {'Success Rate'}")
    print("-" * 50)
    for agent_type, runs in results.items():
        ok = sum(1 for r in runs if r["status"] == "ok")
        avg_time = sum(r["time_sec"] for r in runs) / len(runs)
        print(f"{agent_type:<20} {avg_time:<15.2f} {ok}/{len(runs)}")


if __name__ == "__main__":
    run_benchmark()
