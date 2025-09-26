<a name="readme-top"></a>

<p align="center">
  <!-- The image URL points to the GitHub-hosted content, ensuring it displays correctly on the PyPI website.-->
  <img src="https://raw.githubusercontent.com/ag2ai/ag2/27b37494a6f72b1f8050f6bd7be9a7ff232cf749/website/static/img/ag2.svg" width="150" title="hover text">

  <br>
  <br>

  <a href="https://www.pepy.tech/projects/ag2">
    <img src="https://static.pepy.tech/personalized-badge/ag2?period=month&units=international_system&left_color=grey&right_color=green&left_text=downloads/month" alt="Downloads"/>
  </a>

  <a href="https://pypi.org/project/autogen/">
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
  <a href="https://discord.gg/pAbnFJrkgZ">💬 Join Discord</a>
</p>

<p align="center">
AG2 was evolved from AutoGen. Fully open-sourced. We invite collaborators from all organizations to contribute.
</p>

# AG2: Open-Source AgentOS for AI Agents

AG2 (formerly AutoGen) is an open-source programming framework for building AI agents and facilitating cooperation among multiple agents to solve tasks. AG2 aims to streamline the development and research of agentic AI. It offers features such as agents capable of interacting with each other, facilitates the use of various large language models (LLMs) and tool use support, autonomous and human-in-the-loop workflows, and multi-agent conversation patterns.

The project is currently maintained by a [dynamic group of volunteers](MAINTAINERS.md) from several organizations. Contact project administrators Chi Wang and Qingyun Wu via [support@ag2.ai](mailto:support@ag2.ai) if you are interested in becoming a maintainer.

## Table of contents

- [AG2: Open-Source AgentOS for AI Agents](#ag2-open-source-agentos-for-ai-agents)
  - [Table of contents](#table-of-contents)
  - [Getting started](#getting-started)
    - [Installation](#installation)
    - [Setup your API keys](#setup-your-api-keys)
    - [Run your first agent](#run-your-first-agent)
  - [Example applications](#example-applications)
  - [Introduction of different agent concepts](#introduction-of-different-agent-concepts)
    - [Conversable agent](#conversable-agent)
    - [Human in the loop](#human-in-the-loop)
    - [Orchestrating multiple agents](#orchestrating-multiple-agents)
    - [Tools](#tools)
    - [Advanced agentic design patterns](#advanced-agentic-design-patterns)
  - [Announcements](#announcements)
  - [Code style and linting](#code-style-and-linting)
  - [Related papers](#related-papers)
  - [Contributors Wall](#contributors-wall)
  - [Cite the project](#cite-the-project)
  - [License](#license)

## Getting started

For a step-by-step walk through of AG2 concepts and code, see [Basic Concepts](https://docs.ag2.ai/latest/docs/user-guide/basic-concepts/installing-ag2/) in our documentation.

### Installation

AG2 requires **Python version >= 3.10, < 3.14**. AG2 is available via `ag2` (or its alias `autogen`) on PyPI.

**Windows/Linux:**
```bash
pip install ag2[openai]
```

**Mac:**
```bash
pip install 'ag2[openai]'
```

Minimal dependencies are installed by default. You can install extra options based on the features you need.

### Setup your API keys

To keep your LLM dependencies neat and avoid accidentally checking in code with your API key, we recommend storing your keys in a configuration file.

In our examples, we use a file named **`OAI_CONFIG_LIST`** to store API keys. You can choose any filename, but make sure to add it to `.gitignore` so it will not be committed to source control.

You can use the following content as a template:

```json
[
  {
    "model": "gpt-5",
    "api_key": "<your OpenAI API key here>"
  }
]
```

### Run your first agent

Create a script or a Jupyter Notebook and run your first agent.

```python
from autogen import AssistantAgent, UserProxyAgent, LLMConfig

llm_config = LLMConfig.from_json(path="OAI_CONFIG_LIST")

assistant = AssistantAgent("assistant", llm_config=llm_config)

user_proxy = UserProxyAgent("user_proxy", code_execution_config={"work_dir": "coding", "use_docker": False})

user_proxy.run(assistant, message="Summarize the main differences between Python lists and tuples.").process()
```

## Example applications

We maintain a dedicated repository with a wide range of applications to help you get started with various use cases or check out our collection of jupyter notebooks as a starting point.

- [Build with AG2](https://github.com/ag2ai/build-with-ag2)
- [Jupyter Notebooks](notebook)

## Introduction of different agent concepts

We have several agent concepts in AG2 to help you build your AI agents. We introduce the most common ones here.

- **Conversable Agent**: Agents that are able to send messages, receive messages and generate replies using GenAI models, non-GenAI tools, or human inputs.
- **Human in the loop**: Add human input to the conversation
- **Orchestrating multiple agents**: Users can orchestrate multiple agents with built-in conversation patterns such as swarms, group chats, nested chats, sequential chats or customize the orchestration by registering custom reply methods.
- **Tools**: Programs that can be registered, invoked and executed by agents
- **Advanced Concepts**: AG2 supports more concepts such as structured outputs, rag, code execution, etc.

### Conversable agent

The [ConversableAgent](https://docs.ag2.ai/latest/docs/api-reference/autogen/ConversableAgent) is the fundamental building block of AG2, designed to enable seamless communication between AI entities. This core agent type handles message exchange and response generation, serving as the base class for all agents in the framework.

Let's begin with a simple example where two agents collaborate:
- A **coder agent** that writes Python code.
- A **reviewer agent** that critiques the code without rewriting it.

```python
import logging
from autogen import ConversableAgent, LLMConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load LLM configuration
llm_config = LLMConfig.from_json(path="OAI_CONFIG_LIST")

# Define agents
coder = ConversableAgent(
    name="coder",
    system_message="You are a Python developer. Write short Python scripts.",
    llm_config=llm_config,
)

reviewer = ConversableAgent(
    name="reviewer",
    system_message="You are a code reviewer. Analyze provided code and suggest improvements. "
                   "Do not generate code, only suggest improvements.",
    llm_config=llm_config,
)

# Start a conversation
response = reviewer.run(
            recipient=coder,
            message="Write a Python function that computes Fibonacci numbers.",
            max_turns=10
        )

response.process()

logger.info("Final output:\n%s", response.summary)
```

---
### Orchestrating Multiple Agents

AG2 enables sophisticated multi-agent collaboration through flexible orchestration patterns, allowing you to create dynamic systems where specialized agents work together to solve complex problems.

Here’s how to build a team of **teacher**, **lesson planner**, and **reviewer** agents working together to design a lesson plan:

```python
import logging
from autogen import ConversableAgent, LLMConfig
from autogen.agentchat import run_group_chat
from autogen.agentchat.group.patterns import AutoPattern

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

llm_config = LLMConfig.from_json(path="OAI_CONFIG_LIST")

# Define lesson planner and reviewer
planner_message = "You are a classroom lesson planner. Given a topic, write a lesson plan for a fourth grade class."
reviewer_message = "You are a classroom lesson reviewer. Compare the plan to the curriculum and suggest up to 3 improvements."

lesson_planner = ConversableAgent(
    name="planner_agent",
    system_message=planner_message,
    description="Creates or revises lesson plans.",
    llm_config=llm_config,
)

lesson_reviewer = ConversableAgent(
    name="reviewer_agent",
    system_message=reviewer_message,
    description="Provides one round of feedback to lesson plans.",
    llm_config=llm_config,
)

teacher_message = "You are a classroom teacher. You decide topics and collaborate with planner and reviewer to finalize lesson plans. When satisfied, output DONE!"

teacher = ConversableAgent(
    name="teacher_agent",
    system_message=teacher_message,
    is_termination_msg=lambda x: "DONE!" in (x.get("content", "") or "").upper(),
    llm_config=llm_config,
)

auto_selection = AutoPattern(
    agents=[teacher, lesson_planner, lesson_reviewer],
    initial_agent=lesson_planner,
    group_manager_args={"name": "group_manager", "llm_config": llm_config},
)

response = run_group_chat(
    pattern=auto_selection,
    messages="Let's introduce our kids to the solar system.",
    max_rounds=20,
)

response.process()

logger.info("Final output:\n%s", response.summary)
```

---

### Human in the Loop

Human oversight is often essential for validating or guiding AI outputs.
AG2 provides the `UserProxyAgent` for seamless integration of human feedback.

Here we extend the **teacher–planner–reviewer** example by introducing a **human agent** who validates the final lesson:

```python
import logging
from autogen import ConversableAgent, LLMConfig, UserProxyAgent
from autogen.agentchat import run_group_chat
from autogen.agentchat.group.patterns import AutoPattern

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

llm_config = LLMConfig.from_json(path="OAI_CONFIG_LIST")

# Same agents as before, but now the human validator will pass to the planner who will check for "APPROVED" and terminate
planner_message = "You are a classroom lesson planner. Given a topic, write a lesson plan for a fourth grade class."
reviewer_message = "You are a classroom lesson reviewer. Compare the plan to the curriculum and suggest up to 3 improvements."
teacher_message = "You are an experienced classroom teacher. You don't prepare plans, you provide simple guidance to the planner to prepare a lesson plan on the key topic."

lesson_planner = ConversableAgent(
    name="planner_agent",
    system_message=planner_message,
    description="Creates or revises lesson plans before having them reviewed.",
    is_termination_msg=lambda x: "APPROVED" in (x.get("content", "") or "").upper(),
    human_input_mode="NEVER",
    llm_config=llm_config,
)

lesson_reviewer = ConversableAgent(
    name="reviewer_agent",
    system_message=reviewer_message,
    description="Provides one round of feedback to lesson plans back to the lesson planner before requiring the human validator.",
    llm_config=llm_config,
)

teacher = ConversableAgent(
    name="teacher_agent",
    system_message=teacher_message,
    description="Provides guidance on the topic and content, if required.",
    llm_config=llm_config,
)

human_validator = UserProxyAgent(
    name="human_validator",
    system_message="You are a human educator who provides final approval for lesson plans.",
    description="Evaluates the proposed lesson plan and either approves it or requests revisions, before returning to the planner.",
)

auto_selection = AutoPattern(
    agents=[teacher, lesson_planner, lesson_reviewer],
    initial_agent=teacher,
    user_agent=human_validator,
    group_manager_args={"name": "group_manager", "llm_config": llm_config},
)

response = run_group_chat(
    pattern=auto_selection,
    messages="Let's introduce our kids to the solar system.",
    max_rounds=20,
)

response.process()

logger.info("Final output:\n%s", response.summary)
```

---

### Tools

Agents gain significant utility through **tools**, which extend their capabilities with external data, APIs, or functions.

```python
import logging
from datetime import datetime
from typing import Annotated
from autogen import ConversableAgent, register_function, LLMConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

llm_config = LLMConfig.from_json(path="OAI_CONFIG_LIST")

# Tool: returns weekday for a given date
def get_weekday(date_string: Annotated[str, "Format: YYYY-MM-DD"]) -> str:
    date = datetime.strptime(date_string, "%Y-%m-%d")
    return date.strftime("%A")

date_agent = ConversableAgent(
    name="date_agent",
    system_message="You find the day of the week for a given date.",
    llm_config=llm_config,
)

executor_agent = ConversableAgent(
    name="executor_agent",
    human_input_mode="NEVER",
    llm_config=llm_config,
)

# Register tool
register_function(
    get_weekday,
    caller=date_agent,
    executor=executor_agent,
    description="Get the day of the week for a given date",
)

# Use tool in chat
chat_result = executor_agent.initiate_chat(
    recipient=date_agent,
    message="I was born on 1995-03-25, what day was it?",
    max_turns=2,
)

logger.info("Final output:\n%s", chat_result.chat_history[-1]["content"])
```

### Advanced agentic design patterns

AG2 supports more advanced concepts to help you build your AI agent workflows. You can find more information in the documentation.

- [Structured Output](https://docs.ag2.ai/latest/docs/user-guide/basic-concepts/structured-outputs)
- [Ending a conversation](https://docs.ag2.ai/latest/docs/user-guide/advanced-concepts/orchestration/ending-a-chat/)
- [Retrieval Augmented Generation (RAG)](https://docs.ag2.ai/latest/docs/user-guide/advanced-concepts/rag/)
- [Code Execution](https://docs.ag2.ai/latest/docs/user-guide/advanced-concepts/code-execution)
- [Tools with Secrets](https://docs.ag2.ai/latest/docs/user-guide/advanced-concepts/tools/tools-with-secrets/)
- [Pattern Cookbook (9 group orchestrations)](https://docs.ag2.ai/latest/docs/user-guide/advanced-concepts/pattern-cookbook/overview/)

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

This project uses pre-commit hooks to maintain code quality. Before contributing:

1. Install pre-commit:

```bash
pip install pre-commit
pre-commit install
```

2. The hooks will run automatically on commit, or you can run them manually:

```bash
pre-commit run --all-files
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
