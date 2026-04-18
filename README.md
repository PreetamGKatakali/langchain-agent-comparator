# LangChain Agent Comparator

Switch between three LangChain/LangGraph agent architectures with **one line** in a YAML file — then benchmark them side by side.

---

## Agent Types

| Type | Package | When to use |
|------|---------|-------------|
| `react` | `langchain.agents` | Transparent step-by-step reasoning (Thought → Action → Observation loop) |
| `deep_agent` | `langgraph.prebuilt` | Stateful graph agent with memory and real-time streaming |
| `tool_calling` | `langchain.agents` | Production use — structured JSON tool dispatch, no text parsing |

---

## Quick Start

```bash
# 1. Clone & install
git clone https://github.com/<your-handle>/langchain-agent-comparator
cd langchain-agent-comparator
pip install -r requirements.txt

# 2. Set your API key
cp .env.example .env
# edit .env → add OPENAI_API_KEY (and optionally OPENWEATHER_API_KEY)

# 3. Pick your agent — change ONE line
# config/agent_config.yaml → agent.type: "react" | "deep_agent" | "tool_calling"

# 4. Run
python main.py --query "What is the weather in Mumbai and calculate 15% tip on 500 INR?"
```

**Other run modes:**

```bash
python main.py                    # interactive prompt
python main.py --stream           # stream node-by-node output (deep_agent only)
python main.py --benchmark        # compare all three agents
```

---

## Benchmark

```bash
python main.py --benchmark
```

Runs every test query from `config/agent_config.yaml` against all three agent types and prints a side-by-side comparison:

```
Agent            Avg Time (s)    Success Rate
─────────────────────────────────────────────
react            4.21            4/4
deep_agent       3.87            4/4
tool_calling     2.93            4/4
```

---

## Project Structure

```
langchain-agent-comparator/
├── agents/
│   ├── __init__.py       # AGENT_REGISTRY — maps type names to builders
│   ├── react.py          # ReAct: Thought/Action/Observation loop
│   ├── deep_agent.py     # LangGraph stateful graph agent with memory
│   └── tool_calling.py   # Structured JSON tool dispatch
├── tools/
│   ├── __init__.py       # TOOL_REGISTRY — maps tool names to factories
│   ├── search.py         # DuckDuckGo (no API key needed)
│   ├── calculator.py     # Safe math evaluator
│   └── weather.py        # OpenWeatherMap (falls back to mock if no key)
├── benchmarks/
│   └── runner.py         # Loops all agents × all test queries
├── config/
│   └── agent_config.yaml # Switch agent type and configure tools here
└── main.py               # Single entry point
```

---

## Configuration

Edit `config/agent_config.yaml`:

```yaml
agent:
  type: "react"        # ← change this one line
  llm:
    model: "gpt-4o"
    temperature: 0

tools:
  - search
  - calculator
  - weather

benchmark:
  test_queries:
    - "What is the capital of France and what is 2024 multiplied by 7?"
    - ...
```

---

## Extending

### Add a new agent (3 steps)

1. Create `agents/my_agent.py` with a `build_my_agent(llm, tools)` function
2. Register it in `agents/__init__.py`:
   ```python
   from agents.my_agent import build_my_agent
   AGENT_REGISTRY["my_agent"] = build_my_agent
   ```
3. Set `agent.type: "my_agent"` in the YAML

### Add a new tool (3 steps)

1. Create `tools/my_tool.py` with a `get_my_tool()` function returning a `@tool`
2. Register it in `tools/__init__.py`:
   ```python
   from tools.my_tool import get_my_tool
   TOOL_REGISTRY["my_tool"] = get_my_tool
   ```
3. Add `"my_tool"` to the `tools:` list in the YAML

---

## Requirements

- Python 3.10+
- OpenAI API key (`OPENAI_API_KEY`)
- Optional: OpenWeatherMap key (`OPENWEATHER_API_KEY`) — falls back to mock data without it

```bash
pip install -r requirements.txt
```
