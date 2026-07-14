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
  <a href="https://github.com/ag2ai/build-with-ag2">💡 Examples</a> |
  <a href="https://docs.ag2.ai/latest/docs/contributor-guide/contributing">🤝 Contributing</a> |
  <a href="#related-papers">📝 Cite paper</a> |
  <a href="https://discord.gg/pAbnFJrkgZ">💬 Join Discord</a> |
  <a href="#ag2-classic-the-autogen-namespace">🏛️ AG2 Classic</a>
</p>

<p align="center">
AG2 was evolved from AutoGen. Fully open-sourced. We invite collaborators from all organizations to contribute.
</p>

> [!IMPORTANT]
> **Looking for `ConversableAgent`, `GroupChat`, or `import autogen`? That's now [AG2 Classic](#ag2-classic-the-autogen-namespace).**
>
> As of AG2 v1.0, the protocol-driven framework is the top-level package, imported as `ag2`. The classic framework has moved to its own repository — [**ag2ai/ag2-classic**](https://github.com/ag2ai/ag2-classic), documented at [**classic.docs.ag2.ai**](https://classic.docs.ag2.ai). It is still maintained and installable; nothing you have built stops working.
>
> This repository (`pip install ag2`) no longer ships the `autogen` import name or the classic agent classes.

# AG2: Open-Source AgentOS for AI Agents

AG2 (formerly AutoGen) is an open-source programming framework for building AI agents and facilitating cooperation among multiple agents to solve tasks. AG2 aims to streamline the development and research of agentic AI. It offers features such as agents capable of interacting with each other, facilitates the use of various large language models (LLMs) and tool use support, autonomous and human-in-the-loop workflows, and multi-agent conversation patterns.

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
| **Multi-agent** | `GroupChat`, swarms, nested chats | [Network](https://docs.ag2.ai/latest/docs/user-guide/network/overview) (hub + channels) |

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
> AG2 v1.0 (`pip install ag2`) is **not** a drop-in upgrade from Classic. The agent model, orchestration, and imports all changed. Migrating is a rewrite, not a version bump — so upgrade deliberately, not by accident. See the [migration guide](https://classic.docs.ag2.ai) for what maps to what.

The rest of this README covers **AG2 v1.0** (`import ag2`).

## Getting started

For a step-by-step walk through of AG2 concepts and code, see the [Quick Start](https://docs.ag2.ai/latest/docs/user-guide/quick-start) in our documentation.

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

We maintain a dedicated repository with a wide range of applications to help you get started with various use cases, and a set of runnable code examples in the documentation.

- [Build with AG2](https://github.com/ag2ai/build-with-ag2)
- [Code Examples](https://docs.ag2.ai/latest/docs/user-guide/code_examples/code_examples)

## Introduction of different agent concepts

We have several agent concepts in AG2 to help you build your AI agents. We introduce the most common ones here.

- **Agents**: `Agent` is the core building block — it talks to a model provider, calls tools, and returns a reply.
- **Tools**: Plain Python functions, decorated with `@tool`, that the agent can invoke.
- **Human in the loop**: Pause a run to collect confirmation or missing information from a person.
- **Orchestrating multiple agents**: Coordinate several agents over a hub and typed channels using the **Network**.
- **Advanced Concepts**: Structured outputs, memory, middleware, observers, telemetry, evaluation, and more.

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

Here a `consulting` channel — strict one-question-one-reply, auto-closing on the answer — connects two agents:

```python
import asyncio

from ag2 import Agent
from ag2.config import OpenAIConfig
from ag2.knowledge import MemoryKnowledgeStore
from ag2.network import EV_CHANNEL_CLOSED, EV_TEXT, Hub


async def main() -> None:
    config = OpenAIConfig(model="gpt-4o-mini")
    hub = await Hub.open(MemoryKnowledgeStore(), ttl_sweep_interval=0)

    planner = await hub.register(
        Agent("planner", prompt="Ask one focused question about the lesson topic, then stop.", config=config),
    )
    reviewer = await hub.register(
        Agent("reviewer", prompt="Answer in one short, concrete sentence.", config=config),
    )

    channel = await planner.open(type="consulting", target="reviewer")
    await channel.send("What should a fourth-grade solar system lesson lead with?", audience=[reviewer.agent_id])

    # The reviewer's handler replies, then the consulting adapter closes the channel.
    await planner.wait_for_channel_event(
        channel_id=channel.channel_id,
        predicate=lambda e: e.event_type == EV_CHANNEL_CLOSED,
        timeout=60.0,
    )

    # Replay the exchange from the hub's write-ahead log.
    for envelope in await hub.read_wal(channel.channel_id):
        if envelope.event_type == EV_TEXT:
            print(envelope.event_data["text"])

    await hub.close()


asyncio.run(main())
```

Channels come in several shapes — `consulting` (1Q1R), `conversation` (free-form), `discussion` (round-robin), and `workflow` (a declarative `TransitionGraph` for conditional handoffs, which is the closest analogue to a classic `GroupChat`). See the [Network guide](https://docs.ag2.ai/latest/docs/user-guide/network/overview).

### Advanced agentic design patterns

AG2 supports more advanced concepts to help you build your AI agent workflows. You can find more information in the documentation.

- [Structured Output](https://docs.ag2.ai/latest/docs/user-guide/structured_output)
- [Multi-Agent Network](https://docs.ag2.ai/latest/docs/user-guide/network/overview)
- [Knowledge & Memory](https://docs.ag2.ai/latest/docs/user-guide/agents)
- [Middleware](https://docs.ag2.ai/latest/docs/user-guide/middleware)
- [Telemetry](https://docs.ag2.ai/latest/docs/user-guide/telemetry)
- [Evaluation](https://docs.ag2.ai/latest/docs/user-guide/evaluation)
- [Testing](https://docs.ag2.ai/latest/docs/user-guide/testing)

## Announcements

🔥 🎉 **Nov 11, 2024:** We are evolving AutoGen into **AG2**!
A new organization [AG2AI](https://github.com/ag2ai) is created to host the development of AG2 and related projects with open governance. Check [AG2's new look](https://ag2.ai/).

📄 **License:**
We adopt the Apache 2.0 license from v0.3. This enhances our commitment to open-source collaboration while providing additional protections for contributors and users alike.

🎉 May 29, 2024: DeepLearning.ai launched a new short course [AI Agentic Design Patterns with AutoGen](https://www.deeplearning.ai/short-courses/ai-agentic-design-patterns-with-autogen), made in collaboration with Microsoft and Penn State University, and taught by AutoGen creators [Chi Wang](https://github.com/sonichi) and [Qingyun Wu](https://github.com/qingyun-wu).

🎉 May 24, 2024: Foundation Capital published an article on [Forbes: The Promise of Multi-Agent AI](https://www.forbes.com/sites/joannechen/2024/05/24/the-promise-of-multi-agent-ai/?sh=2c1e4f454d97) and a video [AI in the Real World Episode 2: Exploring Multi-Agent AI and AutoGen with Chi Wang](https://www.youtube.com/watch?v=RLwyXRVvlNk).

🎉 Apr 17, 2024: Andrew Ng cited AutoGen in [The Batch newsletter](https://www.deeplearning.ai/the-batch/issue-245/) and [What's next for AI agentic workflows](https://youtu.be/sal78ACtGTc?si=JduUzN_1kDnMq0vF) at Sequoia Capital's AI Ascent (Mar 26).

[More Announcements](announcements.md)

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

## Related papers

- [AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation](https://arxiv.org/abs/2308.08155)

- [EcoOptiGen: Hyperparameter Optimization for Large Language Model Generation Inference](https://arxiv.org/abs/2303.04673)

- [MathChat: Converse to Tackle Challenging Math Problems with LLM Agents](https://arxiv.org/abs/2306.01337)

- [AgentOptimizer: Offline Training of Language Model Agents with Functions as Learnable Weights](https://arxiv.org/pdf/2402.11359)

- [StateFlow: Enhancing LLM Task-Solving through State-Driven Workflows](https://arxiv.org/abs/2403.11322)

## Contributors Wall

<a href="https://github.com/ag2ai/ag2/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=ag2ai/ag2&max=204" />
</a>

## Cite the project

```
@software{AG2_2024,
author = {Chi Wang and Qingyun Wu and the AG2 Community},
title = {AG2: Open-Source AgentOS for AI Agents},
year = {2024},
url = {https://github.com/ag2ai/ag2},
note = {Available at https://docs.ag2.ai/},
version = {latest}
}
```

## License

This project is licensed under the [Apache License, Version 2.0 (Apache-2.0)](./LICENSE).

This project is a spin-off of [AutoGen](https://github.com/microsoft/autogen) and contains code under two licenses:

- The original code from https://github.com/microsoft/autogen is licensed under the MIT License. See the [LICENSE_original_MIT](./license_original/LICENSE_original_MIT) file for details.

- Modifications and additions made in this fork are licensed under the Apache License, Version 2.0. See the [LICENSE](./LICENSE) file for the full license text.

We have documented these changes for clarity and to ensure transparency with our user and contributor community. For more details, please see the [NOTICE](./NOTICE.md) file.
