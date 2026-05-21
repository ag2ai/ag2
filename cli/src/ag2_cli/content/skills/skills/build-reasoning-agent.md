---
name: build-reasoning-agent
description: Build an AG2 ReasoningAgent that uses tree-of-thought reasoning with beam search, MCTS, or LATS strategies. Use when the user needs advanced reasoning for complex problem solving.
---

# Build Reasoning Agent

You are an expert at building AG2 reasoning agents. When the user wants enhanced reasoning capabilities:

## 1. Understand the Requirements

Ask the user:
- What kind of problem? (Mathematical, analytical, creative, coding)
- How much reasoning depth is needed? (Simple → beam search, Complex → MCTS/LATS)
- Is there a grading/evaluation component? (Use a separate grader LLM config)

## 2. Basic Reasoning Agent

```python
import os
from autogen import LLMConfig, UserProxyAgent
from autogen.agents.experimental import ReasoningAgent

llm_config = LLMConfig(
    {"model": "gpt-4o-mini", "api_key": os.environ["OPENAI_API_KEY"]}
)

# ReasoningAgent explores multiple reasoning paths using tree-of-thought
reasoning_agent = ReasoningAgent(
    name="reasoner",
    llm_config=llm_config,
    reason_config={
        "method": "beam_search",  # Explore multiple paths in parallel
        "beam_size": 3,           # Number of parallel reasoning paths
        "max_depth": 4,           # Maximum reasoning steps
    },
)

user = UserProxyAgent(
    name="user",
    human_input_mode="NEVER",
    code_execution_config=False,
)

result = await user.a_run(
    reasoning_agent,
    message="What is the probability of getting exactly 3 heads in 5 coin flips?",
)
await result.process()
```

## 3. Reasoning Methods

| Method | Best For | Description |
|--------|----------|-------------|
| `beam_search` | General reasoning | Explores top-k paths at each depth |
| `mcts` | Exploration-heavy problems | Monte Carlo Tree Search with UCT |
| `lats` | Per-step evaluation | Language Agent Tree Search with step rewards |
| `dfs` | Simple chain-of-thought | Depth-first (equivalent to beam_size=1) |

### Beam Search (Default)

```python
reason_config = {
    "method": "beam_search",
    "beam_size": 3,           # Parallel paths (default: 3)
    "max_depth": 4,           # Max reasoning steps (default: 3)
    "answer_approach": "pool", # "pool" combines paths, "best" picks top one
}
```

### MCTS (Monte Carlo Tree Search)

```python
reason_config = {
    "method": "mcts",
    "nsim": 10,               # Number of simulations (default: 3)
    "exploration_constant": 1.41,  # UCT exploration parameter
    "max_depth": 4,
}
```

### LATS (Language Agent Tree Search)

```python
reason_config = {
    "method": "lats",
    "nsim": 5,
    "forest_size": 3,         # Number of independent trees
    "max_depth": 4,
}
```

## 4. Separate Grader Model

Use a different (possibly stronger) model for grading reasoning paths:

```python
grader_config = LLMConfig(
    {"model": "gpt-4o", "api_key": os.environ["OPENAI_API_KEY"]}
)

reasoning_agent = ReasoningAgent(
    name="reasoner",
    llm_config=llm_config,
    grader_llm_config=grader_config,  # Stronger model evaluates paths
    reason_config={"method": "beam_search", "beam_size": 5},
)
```

## 5. ReasoningAgent in Group Chat

```python
from autogen import ConversableAgent
from autogen.agentchat import run_group_chat
from autogen.agentchat.group.patterns import RoundRobinPattern

reasoner = ReasoningAgent(
    name="reasoner",
    llm_config=llm_config,
    reason_config={"method": "beam_search", "beam_size": 3, "max_depth": 3},
    description="Performs deep reasoning on complex problems.",
)

executor = ConversableAgent(
    name="executor",
    system_message="You take the reasoner's solution and implement it step by step.",
    llm_config=llm_config,
    human_input_mode="NEVER",
    description="Implements the reasoned solution.",
)

user = ConversableAgent(name="user", llm_config=False, human_input_mode="NEVER")

result = run_group_chat(
    pattern=RoundRobinPattern(
        initial_agent=reasoner,
        agents=[reasoner, executor],
        user_agent=user,
    ),
    messages="Design an algorithm to find the shortest path in a weighted graph with negative edges.",
    max_rounds=8,
)
```

## 6. Rules

- Import from `autogen.agents.experimental`, not `autogen.agentchat`
- `beam_size` and `answer_approach` as top-level params are deprecated — use `reason_config` dict instead
- Higher `beam_size` / `nsim` = better reasoning but more LLM calls and cost
- Use `grader_llm_config` with a stronger model for better path evaluation
- Start with `beam_search` (simplest), move to `mcts`/`lats` if needed
- Set `max_depth` to limit reasoning depth — deeper isn't always better
- `forest_size > 1` runs multiple independent reasoning trees (more diverse solutions)
