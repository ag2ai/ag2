<a name="readme-top"></a>

<p align="center">
  <!-- The image URL points to the GitHub-hosted content, ensuring it displays correctly on the PyPI website.-->
  <img src="https://raw.githubusercontent.com/ag2ai/ag2/27b37494a6f72b1f8050f6bd7be9a7ff232cf749/website/static/img/ag2.svg" width="150" title="hover text">

  <br>
  <br>

  <a href="https://www.pepy.tech/projects/ag2">
    <img src="https://static.pepy.tech/personalized-badge/ag2?period=month&units=international_system&left_color=grey&right_color=green&left_text=downloads/month" alt="Downloads"/>
  </a>

  <a href="https://pypi.org/project/ag2/">
    <img src="https://img.shields.io/pypi/v/ag2?label=PyPI&color=green">
  </a>

  <img src="https://img.shields.io/pypi/pyversions/ag2.svg?label=">

  <a href="https://github.com/ag2ai/ag2/actions/workflows/python-package.yml">
    <img src="https://github.com/ag2ai/ag2/actions/workflows/python-package.yml/badge.svg">
  </a>
  <a href="https://discord.gg/pAbnFJrkgZ">
    <img src="https://img.shields.io/discord/1153072414184452236?logo=discord&style=flat">
  </a>

  <br>

  <a href="https://x.com/ag2oss">
    <img src="https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Follow%20%40ag2ai">
  </a>
</p>

<p align="center">
  <a href="https://docs.ag2.ai/">📚 Documentation</a> |
  <a href="https://playground.ag2.ai">💡 Playground</a> |
  <a href="https://github.com/ag2ai/build-with-ag2">💡 Examples</a> |
  <a href="https://docs.ag2.ai/docs/contributor-guide/contributing/">🤝 Contributing</a> |
  <a href="#related-papers">📝 Cite paper</a> |
  <a href="https://discord.gg/pAbnFJrkgZ">💬 Join Discord</a> |
  <a href="#ag2-classic-the-autogen-namespace">🏛️ AG2 Classic</a>
</p>


> [!IMPORTANT]
> **Looking for `ConversableAgent`, `GroupChat`, or `import autogen`? That's now [AG2 Classic](#ag2-classic-the-autogen-namespace).**
>
> As of AG2 v1.0, the protocol-driven framework is the top-level package, imported as `ag2`. The classic framework has moved to its own repository — [**ag2ai/ag2-classic**](https://github.com/ag2ai/ag2-classic), documented at [**classic.docs.ag2.ai**](https://classic.docs.ag2.ai). It is still maintained and installable; nothing you have built stops working.
>
> This repository (`pip install ag2`) no longer ships the `autogen` import name or the classic agent classes.

# AG2: Open-Source AgentOS for AI Agents

AG2 is an open-source programming framework for building AI agents and facilitating cooperation among multiple agents to solve tasks. AG2 aims to streamline the development and research of agentic AI. It offers features such as agents capable of interacting with each other, facilitates the use of various large language models (LLMs) and tool use support, autonomous and human-in-the-loop workflows, and multi-agent conversation patterns.

The project is currently maintained by a [dynamic group of volunteers](MAINTAINERS.md) from several organizations. Contact project administrators Chi Wang and Qingyun Wu via [support@ag2.ai](mailto:support@ag2.ai) if you are interested in becoming a maintainer.

## Table of contents

- [AG2: Open-Source AgentOS for AI Agents](#ag2-open-source-agentos-for-ai-agents)
  - [Table of contents](#table-of-contents)
  - [AG2 Classic (the `autogen.*` namespace)](#ag2-classic-the-autogen-namespace)
  - [Getting started](#getting-started)
    - [Installation](#installation)
    - [Setup your API keys](#setup-your-api-keys)
    - [Run your first agent](#run-your-first-agent)
  - [Example applications](#example-applications)
  - [Introduction of different agent concepts](#introduction-of-different-agent-concepts)
    - [Agents](#agents)
    - [Tools](#tools)
    - [Human in the Loop](#human-in-the-loop)
    - [Orchestrating Multiple Agents](#orchestrating-multiple-agents)
    - [The agent harness: knowledge and compaction](#the-agent-harness-knowledge-and-compaction)
    - [Advanced agentic design patterns](#advanced-agentic-design-patterns)
  - [Announcements](#announcements)
  - [Code style and linting](#code-style-and-linting)
  - [Related papers](#related-papers)
  - [Contributors Wall](#contributors-wall)
  - [Cite the project](#cite-the-project)
  - [License](#license)

## AG2 Classic (the `autogen.*` namespace)

**AG2 Classic** is the original AutoGen-derived framework: the `autogen.*` import namespace and its agent classes — `ConversableAgent`, `AssistantAgent`, `UserProxyAgent`, `GroupChat` / `GroupChatManager`, swarms, `register_function`, `LLMConfig` / `OAI_CONFIG_LIST`, and the nested- and sequential-chat patterns.

It now lives in its own repository and has its own documentation site:

| | AG2 Classic | AG2 (this repo) |
|---|---|---|
| **Repository** | [ag2ai/ag2-classic](https://github.com/ag2ai/ag2-classic) | [ag2ai/ag2](https://github.com/ag2ai/ag2) |
| **Documentation** | [classic.docs.ag2.ai](https://classic.docs.ag2.ai) | [docs.ag2.ai](https://docs.ag2.ai) |
| **Import** | `import autogen` | `import ag2` |
| **Core agent** | `ConversableAgent` | `Agent` |
| **Multi-agent** | `GroupChat`, swarms, nested chats | [Network](https://docs.ag2.ai/docs/user-guide/network/overview/) (hub + channels) |

### Are you using AG2 Classic?

If any of the following appear in your code, you are on Classic — stay on it, and use [classic.docs.ag2.ai](https://classic.docs.ag2.ai):

```python
import autogen                                    # the autogen.* namespace
from autogen import ConversableAgent, GroupChat   # classic agent classes
from autogen import AssistantAgent, UserProxyAgent
```

Classic remains maintained and installable. **Your existing code keeps working** — pin the classic distribution instead of `ag2>=1.0`:

```bash
pip install ag2-classic
```

> [!NOTE]
> AG2 v1.0 (`pip install ag2`) is **not** a drop-in upgrade from Classic. The agent model, orchestration, and imports all changed. See the [group chat migration guide](https://docs.ag2.ai/docs/user-guide/network/migration_from_group_chat/).

The rest of this README covers **AG2 v1.0** (`import ag2`).

## Getting started

For a step-by-step walk through of AG2 concepts and code, see the [Quick Start](https://docs.ag2.ai/docs/user-guide/quick-start/) in our documentation.

### Installation

AG2 requires **Python version >= 3.10**. AG2 is available as `ag2` on PyPI.

**Windows/Linux:**
```bash
pip install ag2[openai]
```

**Mac:**
```bash
pip install 'ag2[openai]'
```

Minimal dependencies are installed by default. Install the extra that matches your model provider — `ag2[openai]`, `ag2[anthropic]`, `ag2[gemini]`, `ag2[ollama]`, and so on.

### Setup your API keys

Each provider config reads its standard environment variable, so keys never need to be hardcoded or checked in:

```bash
export OPENAI_API_KEY="your-api-key"       # or ANTHROPIC_API_KEY, GEMINI_API_KEY, ...
```

You can also pass a key explicitly with `OpenAIConfig(model="gpt-4o-mini", api_key=...)` — useful when each request brings its own key.

### Run your first agent

AG2 is async throughout. `Agent.ask(...)` starts a turn and returns an `AgentReply`; the text is in `reply.body`.

```python
import asyncio

from ag2 import Agent
from ag2.config import OpenAIConfig

agent = Agent(
    "assistant",
    prompt="You are a helpful assistant.",
    config=OpenAIConfig(model="gpt-4o-mini"),
)


async def main() -> None:
    reply = await agent.ask("Summarize the main differences between Python lists and tuples.")
    print(reply.body)


asyncio.run(main())
```

## Example applications

We maintain both a live playground and a dedicated repository with a wide range of applications to help you get started with various use cases, and a set of runnable code examples in the documentation.

- [AG2 Playground](https://playground.ag2.ai)
- [Build with AG2](https://github.com/ag2ai/build-with-ag2)
- [Code Examples](https://docs.ag2.ai/docs/user-guide/code_examples/code_examples/)

## Introduction of different agent concepts

We have several agent concepts in AG2 to help you build your AI agents. We introduce the most common ones here.

- **Agents**: `Agent` is the core building block — it talks to a model provider, calls tools, and returns a reply.
- **Tools**: Plain Python functions, decorated with `@tool`, that the agent can invoke.
- **Human in the loop**: Pause a run to collect confirmation or missing information from a person.
- **Orchestrating multiple agents**: Coordinate several agents over a hub and typed channels using the **Network**.
- **The agent harness**: Opt-in primitives layered onto an agent — persistent knowledge, context assembly, and history compaction.
- **Advanced Concepts**: Structured outputs, middleware, observers, telemetry, evaluation, and more.

### Agents

The `Agent` is the fundamental building block of AG2. `ask()` runs a turn; calling `ask()` on the returned reply continues the *same* conversation, preserving its history.

```python
import asyncio

from ag2 import Agent
from ag2.config import OpenAIConfig

reviewer = Agent(
    "reviewer",
    prompt=(
        "You are a code reviewer. Analyze the provided code and suggest improvements. "
        "Do not generate code, only suggest improvements."
    ),
    config=OpenAIConfig(model="gpt-4o-mini"),
)


async def main() -> None:
    reply = await reviewer.ask("def fib(n): return n if n < 2 else fib(n-1) + fib(n-2)")
    print(reply.body)

    # Continue the same conversation — prior turns stay in scope.
    follow_up = await reply.ask("Which of those matters most for large n?")
    print(follow_up.body)


asyncio.run(main())
```

---

### Tools

Agents gain significant utility through **tools**, which extend their capabilities with external data, APIs, or functions. Decorate a Python function with `@tool` and pass it to the agent — AG2 runs the full tool-calling loop: the model decides when to call it, AG2 executes it, and the result is fed back.

```python
import asyncio
from datetime import datetime

from ag2 import Agent, tool
from ag2.config import OpenAIConfig


@tool
async def get_weekday(date_string: str) -> str:
    """Get the day of the week for a given date, formatted as YYYY-MM-DD."""
    return datetime.strptime(date_string, "%Y-%m-%d").strftime("%A")


date_agent = Agent(
    "date_agent",
    prompt="You find the day of the week for a given date.",
    config=OpenAIConfig(model="gpt-4o-mini"),
    tools=[get_weekday],
)


async def main() -> None:
    reply = await date_agent.ask("I was born on 1995-03-25, what day was it?")
    print(reply.body)


asyncio.run(main())
```

---

### Human in the Loop

Human oversight is often essential for validating or guiding AI outputs. Call `context.input(...)` inside a tool to pause the run and ask a person — your `hitl_hook` decides how that question is answered (CLI prompt, web UI, queue, …).

```python
import asyncio

from ag2 import Agent, Context, tool
from ag2.config import OpenAIConfig
from ag2.events import HumanInputRequest, HumanMessage


@tool
async def publish_lesson_plan(context: Context, plan: str) -> str:
    """Publish a lesson plan, once a human educator has approved it."""
    answer = await context.input(f"Approve this lesson plan?\n\n{plan}")
    if answer.strip().lower().startswith("y"):
        return "Published."
    return f"Rejected by the educator: {answer}"


def hitl_hook(event: HumanInputRequest) -> HumanMessage:
    # Block here for real input — a CLI prompt, a web UI, a queue.
    return HumanMessage(content=input(f"{event.content}\n> "))


teacher = Agent(
    "teacher",
    prompt=(
        "Draft a short lesson plan, then call the publish_lesson_plan tool to have it published. "
        "Approval is collected by the tool — never ask for approval in plain text."
    ),
    config=OpenAIConfig(model="gpt-4o-mini"),
    tools=[publish_lesson_plan],
    hitl_hook=hitl_hook,
)


async def main() -> None:
    reply = await teacher.ask("Let's introduce our kids to the solar system.")
    print(reply.body)


asyncio.run(main())
```

---

### Orchestrating Multiple Agents

When two or more agents need to work together, AG2 uses the **Network**: a `Hub` that owns the registry, the write-ahead log, and the audit trail, with agents talking over typed **channels**. This replaces the classic `GroupChat` / swarm / nested-chat patterns.

Here a `conversation` channel — a free-form two-party session where either side may speak at any time — connects a planner and a reviewer:

```python
import asyncio

from ag2 import Agent
from ag2.config import OpenAIConfig
from ag2.knowledge import MemoryKnowledgeStore
from ag2.network import EV_TEXT, Hub

MAX_MESSAGES = 4


async def wait_for_messages(hub: Hub, channel_id: str, expected: int, timeout: float = 120.0) -> None:
    """Poll the hub's write-ahead log until `expected` messages have been exchanged."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        wal = await hub.read_wal(channel_id)
        if sum(1 for e in wal if e.event_type == EV_TEXT) >= expected:
            return
        await asyncio.sleep(0.05)
    raise TimeoutError(f"only reached {expected} messages before timing out")


async def main() -> None:
    config = OpenAIConfig(model="gpt-4o-mini")
    hub = await Hub.open(MemoryKnowledgeStore(), ttl_sweep_interval=0)

    planner = await hub.register(
        Agent("planner", prompt="You plan school lessons. Reply in one short sentence.", config=config),
    )
    reviewer = await hub.register(
        Agent("reviewer", prompt="You critique lesson plans. Reply in one short sentence.", config=config),
    )

    # Free-form 2-party channel: either side may speak at any time.
    channel = await planner.open(type="conversation", target="reviewer")
    await channel.send("What should a fourth-grade solar system lesson lead with?", audience=[reviewer.agent_id])

    # Both default handlers reply to every inbound message, so the conversation
    # auto-drives. `conversation` never auto-closes — we bound it and close it.
    await wait_for_messages(hub, channel.channel_id, expected=MAX_MESSAGES)
    await channel.close()

    # Replay the exchange from the hub's write-ahead log.
    names = {planner.agent_id: "planner", reviewer.agent_id: "reviewer"}
    for envelope in await hub.read_wal(channel.channel_id):
        if envelope.event_type == EV_TEXT:
            print(f"{names[envelope.sender_id]}: {envelope.event_data['text']}")

    await hub.close()


asyncio.run(main())
```

Channels come in several shapes — `conversation` (free-form, two parties), `consulting` (strict one-question-one-reply, auto-closing), `discussion` (round-robin across N agents), and `workflow` (a declarative `TransitionGraph` for conditional handoffs, which is the closest analogue to a classic `GroupChat`). See the [Network guide](https://docs.ag2.ai/docs/user-guide/network/overview/).

### The agent harness: knowledge and compaction

A bare `Agent` is just a model loop. The **harness** is the set of opt-in primitives you compose onto it. Two of the most useful:

- **`knowledge=`** — a `KnowledgeStore` the agent can read and write, so it remembers across runs.
- **`compact=`** — a strategy that caps history growth. `SummarizeCompact` folds the dropped turns into a summary rather than discarding them outright.

Pair the store with a `WorkingMemoryPolicy` in `assembly=` and the agent's memory is injected into the system prompt on every turn — recall no longer depends on the model *choosing* to look it up.

```python
import asyncio
from pathlib import Path

from ag2 import Agent, KnowledgeConfig, MemoryStream
from ag2.compact import CompactTrigger, SummarizeCompact
from ag2.config import OpenAIConfig
from ag2.events import CompactionCompleted, CompactionFailed
from ag2.knowledge import DiskKnowledgeStore
from ag2.policies import WorkingMemoryPolicy

config = OpenAIConfig(model="gpt-4o-mini")

# Disk-backed, so anything the agent remembers survives the process exiting.
STORE_DIR = Path("./knowledge_demo")
store = DiskKnowledgeStore(STORE_DIR)

agent = Agent(
    "tutor",
    prompt=(
        "You are a science tutor for a fourth-grade teacher. "
        "Whenever you learn a durable fact about the teacher or their class, use the knowledge "
        "tool to `read` `memory/working.md`, then `write` it back with the new fact appended as "
        "a bullet. Never drop a fact that is already there. "
        "Keep every reply to one or two sentences."
    ),
    config=config,
    knowledge=KnowledgeConfig(
        store=store,
        # Summarize the dropped history instead of forgetting it outright.
        compact=SummarizeCompact(target=4, config=config),
        compact_trigger=CompactTrigger(max_events=8),
    ),
    # Injects memory/working.md into the system prompt on every turn, so recall
    # does not depend on the model choosing to call the knowledge tool.
    assembly=[WorkingMemoryPolicy()],
)

stream = MemoryStream()


# Subscribe to both outcomes — watching only Completed would hide a failing strategy.
@stream.where(CompactionCompleted).subscribe()
async def on_compacted(event: CompactionCompleted) -> None:
    print(f"  [compacted: {event.events_before} -> {event.events_after} events via {event.strategy}]")


@stream.where(CompactionFailed).subscribe()
async def on_compaction_failed(event: CompactionFailed) -> None:
    print(f"  [compaction FAILED: {event.strategy}]")


async def main() -> None:
    # A returning session: no conversation history at all, so anything the agent
    # recalls here came off disk, via the knowledge store.
    if STORE_DIR.exists():
        reply = await agent.ask("What do you remember about me and my class?", stream=stream)
        print(f"tutor: {reply.body}")
        return

    reply = await agent.ask("I'm Dana. I teach 4th grade at Rosewood Elementary, 26 students.", stream=stream)
    print(f"tutor: {reply.body}")

    for turn in ["My class struggles most with why we have seasons.", "Suggest one hands-on demo for that."]:
        reply = await reply.ask(turn)
        print(f"tutor: {reply.body}")


asyncio.run(main())
```

Run it twice. The first run fills `memory/working.md` and trips compaction; the second starts with an empty history and still knows who Dana is:

```text
  [compacted: 18 -> 3 events via SummarizeCompact]
tutor: I remember that you, Dana, teach 4th grade at Rosewood Elementary with 26 students,
       and your class struggles the most with understanding why we have seasons.
```

See the [Agent Harness guide](https://docs.ag2.ai/docs/user-guide/agent_harness/) for `assembly=`, `tasks=`, aggregation, and the full turn lifecycle.

### Advanced agentic design patterns

AG2 supports more advanced concepts to help you build your AI agent workflows. You can find more information in the documentation.

- [Structured Output](https://docs.ag2.ai/docs/user-guide/structured_output/)
- [Multi-Agent Network](https://docs.ag2.ai/docs/user-guide/network/overview/)
- [Knowledge & Memory](https://docs.ag2.ai/docs/user-guide/advanced/knowledge_store/)
- [Middleware](https://docs.ag2.ai/docs/user-guide/middleware/)
- [Telemetry](https://docs.ag2.ai/docs/user-guide/telemetry/)
- [Evaluation](https://docs.ag2.ai/docs/user-guide/evaluation/evaluation/)
- [Testing](https://docs.ag2.ai/docs/user-guide/testing/)

## Code style and linting

This project uses [prek](https://github.com/j178/prek) hooks to maintain code quality. Before contributing:

1. Install prek:

```bash
pip install prek
prek install
```

2. The hooks will run automatically on commit, or you can run them manually:

```bash
prek run --all-files
```

## Contributors Wall

<a href="https://github.com/ag2ai/ag2/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=ag2ai/ag2&max=204" />
</a>

## License

This project is licensed under the [Apache License, Version 2.0 (Apache-2.0)](./LICENSE).

- Modifications and additions made in this fork are licensed under the Apache License, Version 2.0. See the [LICENSE](./LICENSE) file for the full license text.

We have documented these changes for clarity and to ensure transparency with our user and contributor community. For more details, please see the [NOTICE](./NOTICE.md) file.
