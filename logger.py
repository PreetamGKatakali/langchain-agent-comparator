"""
logger.py — Tool execution tracer.

Hooks into every tool call via LangChain callbacks and prints:
  - tool name + input when it starts
  - output + time taken when it ends
  - a summary table at the end

Usage:
    from logger import ToolLogger
    logger = ToolLogger()
    agent.invoke({"input": query}, config={"callbacks": [logger]})
    logger.print_summary()
"""

import time
from langchain_core.callbacks import BaseCallbackHandler


CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


class ToolLogger(BaseCallbackHandler):

    def __init__(self):
        self.logs = []          # list of completed tool call records
        self._start_time = None
        self._current_tool = None
        self._step = 0

    # ── called just before a tool runs ──────────────────────────────
    def on_tool_start(self, serialized: dict, input_str: str, **kwargs):
        self._step += 1
        self._current_tool = serialized.get("name", "unknown")
        self._start_time = time.perf_counter()

        print(f"\n{CYAN}{BOLD}┌─ Tool Call #{self._step}{RESET}")
        print(f"{CYAN}│  Tool   : {BOLD}{self._current_tool}{RESET}")
        print(f"{CYAN}│  Input  : {input_str}{RESET}")
        print(f"{CYAN}└{'─'*40}{RESET}")

    # ── called after a tool returns ──────────────────────────────────
    def on_tool_end(self, output: str, **kwargs):
        elapsed = time.perf_counter() - self._start_time
        output_preview = str(output)[:200].replace("\n", " ")

        print(f"{GREEN}{BOLD}┌─ Tool Result #{self._step}{RESET}")
        print(f"{GREEN}│  Tool   : {self._current_tool}{RESET}")
        print(f"{GREEN}│  Output : {output_preview}{'...' if len(str(output)) > 200 else ''}{RESET}")
        print(f"{GREEN}│  Time   : {elapsed:.2f}s{RESET}")
        print(f"{GREEN}└{'─'*40}{RESET}")

        self.logs.append({
            "step":   self._step,
            "tool":   self._current_tool,
            "input":  kwargs.get("input", ""),
            "output": output_preview,
            "time":   round(elapsed, 2),
            "status": "ok",
        })

    # ── called if a tool raises an exception ────────────────────────
    def on_tool_error(self, error: Exception, **kwargs):
        elapsed = time.perf_counter() - self._start_time

        print(f"{RED}{BOLD}┌─ Tool Error #{self._step}{RESET}")
        print(f"{RED}│  Tool  : {self._current_tool}{RESET}")
        print(f"{RED}│  Error : {error}{RESET}")
        print(f"{RED}└{'─'*40}{RESET}")

        self.logs.append({
            "step":   self._step,
            "tool":   self._current_tool,
            "input":  "",
            "output": str(error),
            "time":   round(elapsed, 2),
            "status": "error",
        })

    # ── summary table printed after agent finishes ──────────────────
    def print_summary(self):
        if not self.logs:
            print(f"\n{YELLOW}No tools were called.{RESET}")
            return

        total_time = sum(r["time"] for r in self.logs)

        print(f"\n{BOLD}{'='*60}{RESET}")
        print(f"{BOLD}  TOOL EXECUTION SUMMARY  ({len(self.logs)} calls | {total_time:.2f}s total){RESET}")
        print(f"{BOLD}{'='*60}{RESET}")
        print(f"{'#':<4} {'Tool':<14} {'Time':>6}   {'Status':<8}  Output Preview")
        print(f"{'-'*60}")
        for r in self.logs:
            status_color = GREEN if r["status"] == "ok" else RED
            preview = r["output"][:45] + "..." if len(r["output"]) > 45 else r["output"]
            print(
                f"{r['step']:<4} "
                f"{r['tool']:<14} "
                f"{r['time']:>5.2f}s   "
                f"{status_color}{r['status']:<8}{RESET}  "
                f"{preview}"
            )
        print(f"{BOLD}{'='*60}{RESET}\n")

    def reset(self):
        self.logs = []
        self._step = 0
