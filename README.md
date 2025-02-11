<a name="readme-top"></a>

<p align="center">
  <img src="assets/ag2-logo.png" width="100" title="hover text">
  <br>
  <br>
  <img src="https://img.shields.io/pypi/dm/pyautogen?label=PyPI%20downloads">
  <a href="https://badge.fury.io/py/autogen"><img src="https://badge.fury.io/py/autogen.svg"></a>
  <a href="https://github.com/ag2ai/ag2/actions/workflows/python-package.yml">
    <img src="https://github.com/ag2ai/ag2/actions/workflows/python-package.yml/badge.svg">
  </a>
  <img src="https://img.shields.io/badge/3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue">
  <a href="https://discord.gg/pAbnFJrkgZ">
    <img src="https://img.shields.io/discord/1153072414184452236?logo=discord&style=flat">
  </a>
  <a href="https://x.com/Chi_Wang_">
    <img src="https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Follow%20%40ag2ai">
  </a>
</p>

<p align="center">
  <a href="https://docs.ag2.ai/">📚 Documentation</a> |
  <a href="https://github.com/ag2ai/build-with-ag2">💡 Examples</a> |
  <a href="https://docs.ag2.ai/docs/contributor-guide/contributing">🤝 Contributing</a> |
  <a href="#related-papers">📚 Cite paper</a> |
  <a href="https://discord.gg/pAbnFJrkgZ">💬 Join Discord</a>
</p>

<p align="center">
AG2 was evolved from AutoGen. Fully open-sourced we invite collaborators from all organizations to contribute.
</p>

# AG2: Open-Source AgentOS for AI Agents

AG2 (formerly AutoGen) is an open-source programming framework for building AI agents and facilitating cooperation among multiple agents to solve tasks. AG2 aims to streamline the development and research of agentic AI. It offers features such as agents capable of interacting with each other, facilitates the use of various large language models (LLMs) and tool use support, autonomous and human-in-the-loop workflows, and multi-agent conversation patterns.

The project is currently maintained by a [dynamic group of volunteers](MAINTAINERS.md) from several organizations. Contact project administrators Chi Wang and Qingyun Wu via [support@ag2.ai](mailto:support@ag2.ai) if you are interested in becoming a maintainer.

## Installation

AG2 requires **Python version >= 3.9, < 3.14**. AG2 is available via `pyautogen` (or its alias `autogen` or `ag2`) on PyPI!

```bash
pip install ag2
```

Minimal dependencies are installed without extra options. You can install extra options based on the feature you need.

## Getting Started

### 1. Set up your API keys

To keep your LLM dependencies neat we recommend using the `OAI_CONFIG_LIST` file to store your API keys.

You can use the sample file `OAI_CONFIG_LIST_sample` as a template.

```json
[
  {
    "model": "gpt-4o",
    "api_key": "<your OpenAI API key here>"
  }
]
```

### Run your first agent

```python
from autogen import AssistantAgent, UserProxyAgent, config_list_from_json

llm_config = {
    "config_list": config_list_from_json(env_or_file="OAI_CONFIG_LIST")
}

assistant = AssistantAgent("assistant", llm_config=llm_config)
user_proxy = UserProxyAgent("user_proxy", code_execution_config={"work_dir": "coding", "use_docker": False})
user_proxy.initiate_chat(assistant, message="Plot a chart of NVDA and TESLA stock price change YTD.")
# This initiates an automated chat between the two agents to solve the task
```

## Example applications

We maintain a dedicated repository with a wide range of applications to help you get started with various use cases or check out some juypter notebooks as a starting point.

- [Build with AG2](https://github.com/ag2ai/build-with-ag2)
- [Jutpyer Notebooks](notebook)

## Introduction of different agent concepts

We have several agent concepts in AG2 to help you build your AI agents. We introduce the most common ones here.

- **Conversable Agent**: The conversable agent
- **Human in the loop**: Conversable agent with human_input_mode set to ALWAYS
- **Orchestrating multiple agents**: GroupChat and Swarm
- **Tools**: Attach functionalities to agents
- **Structured Output**: Structured output from agents

### Conversable agent

The conversable agent is the most used agent and is created for generating conversations among agents.

```python
from autogen import ConversableAgent

# Create an AI agent
assistant = ConversableAgent(
    name="assistant",
    system_message="You are an assistant that responds concisely.",
    llm_config=llm_config
)

# Create another AI agent
fact_checker = ConversableAgent(
    name="fact_checker",
    system_message="You are a fact-checking assistant.",
    llm_config=llm_config
)

# Start the conversation
assistant.initiate_chat(
    recipient=fact_checker,
    message="What is ag2?",
    max_turns=2
)
```

### Human in the loop

If you set the setting `human_input_mode` to `ALWAYS` on the Conversable Agent you can give human input to the conversation.

There are three modes for `human_input_mode`: `ALWAYS`, `NEVER`, `TERMINATE`.

We created a class which sets the `human_input_mode` to `ALWAYS` for you. Its called `UserProxyAgent`.

```python
from autogen import ConversableAgent

# Create an AI agent
assistant = ConversableAgent(
    name="assistant",
    system_message="You are a helpful assistant.",
    llm_config=llm_config
)

# Create a human agent with manual input mode
human = ConversableAgent(
    name="human",
    human_input_mode="ALWAYS"
)
# or
human = UserProxyAgent(name="human", code_execution_config={"work_dir": "coding", "use_docker": False})

# Start the chat
human.initiate_chat(
    recipient=assistant,
    message="Hello! What's 2 + 2?"
)

```

### Orchestrating multiple agents

At ag2 we have two concepts to orchestrate multiple agents. `Group Chat` or `Swarm`.
Both concepts are used to orchestrate multiple agents to solve a task.

The group chat works like a chat where each registered agent can participate in the conversation.

```python
from autogen import ConversableAgent, GroupChat, GroupChatManager

# Create AI agents
teacher = ConversableAgent(name="teacher", system_message="You suggest lesson topics.")
planner = ConversableAgent(name="planner", system_message="You create lesson plans.")
reviewer = ConversableAgent(name="reviewer", system_message="You review lesson plans.")

# Create GroupChat
groupchat = GroupChat(agents=[teacher, planner, reviewer], speaker_selection_method="auto")

# Create GroupChatManager
manager = GroupChatManager(name="manager", groupchat=groupchat)

# Start the conversation
teacher.initiate_chat(manager, "Create a lesson on photosynthesis.")
```

The swarm requires a more rigid structure and the flow needs to be defined with hand-off, post-tool, and post-work transitions from an agent to another agent.

Read more about it in the [documentation](https://docs.ag2.ai/docs/user-guide/basic-concepts/orchestrations)

### Tools

Agents gain significant utility through tools as they provide access to external data, APIs, and functionality.

```python
from datetime import datetime
from typing import Annotated

from autogen import ConversableAgent, register_function

# 1. Our tool, returns the day of the week for a given date
def get_weekday(date_string: Annotated[str, "Format: YYYY-MM-DD"]) -> str:
    date = datetime.strptime(date_string, "%Y-%m-%d")
    return date.strftime("%A")

# 2. Agent for determining whether to run the tool
date_agent = ConversableAgent(
    name="date_agent",
    system_message="You get the day of the week for a given date.",
    llm_config=llm_config,
)

# 3. And an agent for executing the tool
executor_agent = ConversableAgent(
    name="executor_agent",
    human_input_mode="NEVER",
)

# 4. Registers the tool with the agents, the description will be used by the LLM
register_function(
    get_weekday,
    caller=date_agent,
    executor=executor_agent,
    description="Get the day of the week for a given date",
)

# 5. Two-way chat ensures the executor agent follows the suggesting agent
chat_result = executor_agent.initiate_chat(
    recipient=date_agent,
    message="I was born on the 25th of March 1995, what day was it?",
    max_turns=1,
)
```

### Structured Output

Structured output is a way to get structured data from the agents. Structured output are really helpful when you need to do an action based on the output from the agent, e.g. upload to a database, create a file, etc.

```python
import json
from pydantic import BaseModel
from autogen import ConversableAgent

# Define a structured response model
class LessonPlan(BaseModel):
    title: str
    script: str

llm_config_list = config_list_from_json(
    env_or_file="OAI_CONFIG_LIST",
    filter_dict={
        "model": ["gpt-4o"],
    },
)

for config in llm_config_list:
    config["response_format"] = LessonPlan

llm_config = {
    "config_list": llm_config_list,
}

# Configure the AI agent
lesson_agent = ConversableAgent(
    name="lesson_agent",
    llm_config=llm_config,
    system_message="You create simple lesson plans."
)

# Human agent
human = ConversableAgent(name="human", human_input_mode="NEVER")

# Start chat
result = human.initiate_chat(recipient=lesson_agent, message="Create a lesson on gravity.", max_turns=1)

```

## Announcements

📄 **License:**
We adopt the Apache 2.0 license from v0.3. This enhances our commitment to open-source collaboration while providing additional protections for contributors and users alike.

🎉 May 29, 2024: DeepLearning.ai launched a new short course [AI Agentic Design Patterns with AutoGen](https://www.deeplearning.ai/short-courses/ai-agentic-design-patterns-with-autogen), made in collaboration with Microsoft and Penn State University, and taught by AutoGen creators [Chi Wang](https://github.com/sonichi) and [Qingyun Wu](https://github.com/qingyun-wu).

🎉 May 24, 2024: Foundation Capital published an article on [Forbes: The Promise of Multi-Agent AI](https://www.forbes.com/sites/joannechen/2024/05/24/the-promise-of-multi-agent-ai/?sh=2c1e4f454d97) and a video [AI in the Real World Episode 2: Exploring Multi-Agent AI and AutoGen with Chi Wang](https://www.youtube.com/watch?v=RLwyXRVvlNk).

🎉 May 13, 2024: [The Economist](https://www.economist.com/science-and-technology/2024/05/13/todays-ai-models-are-impressive-teams-of-them-will-be-formidable) published an article about multi-agent systems (MAS) following a January 2024 interview with [Chi Wang](https://github.com/sonichi).

🎉 May 11, 2024: [AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation](https://openreview.net/pdf?id=uAjxFFing2) received the best paper award at the [ICLR 2024 LLM Agents Workshop](https://llmagents.github.io/).

🎉 Apr 26, 2024: [AutoGen.NET](https://docs.ag2.ai/ag2-for-net/) is available for .NET developers

🎉 Apr 17, 2024: Andrew Ng cited AutoGen in [The Batch newsletter](https://www.deeplearning.ai/the-batch/issue-245/) and [What's next for AI agentic workflows](https://youtu.be/sal78ACtGTc?si=JduUzN_1kDnMq0vF) at Sequoia Capital's AI Ascent (Mar 26).

🎉 Mar 3, 2024: What's new in AutoGen? 📰[Blog](https://docs.ag2.ai/blog/2024-03-03-AutoGen-Update); 📺[Youtube](https://www.youtube.com/watch?v=j_mtwQiaLGU).

🎉 Mar 1, 2024: the first AutoGen multi-agent experiment on the challenging [GAIA](https://huggingface.co/spaces/gaia-benchmark/leaderboard) benchmark achieved the No. 1 accuracy in all the three levels.

🎉 Jan 30, 2024: AutoGen is highlighted by Peter Lee in Microsoft Research Forum [Keynote](https://t.co/nUBSjPDjqD).

🎉 Dec 31, 2023: [AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation Framework](https://arxiv.org/abs/2308.08155) is selected by [TheSequence: My Five Favorite AI Papers of 2023](https://thesequence.substack.com/p/my-five-favorite-ai-papers-of-2023).

🔥 Nov 24: pyautogen [v0.2](https://github.com/ag2ai/ag2/releases/tag/v0.2.0) is released with many updates and new features compared to v0.1.1. It switches to using openai-python v1. Please read the [migration guide](https://docs.ag2.ai/docs/installation/Installation).

🔥 Nov 11: OpenAI's Assistants are available in AutoGen and interoperatable with other AutoGen agents! Checkout our [blogpost](https://docs.ag2.ai/blog/2023-11-13-OAI-assistants) for details and examples.

🎉 Nov 8, 2023: AutoGen is selected into [Open100: Top 100 Open Source achievements](https://www.benchcouncil.org/evaluation/opencs/annual.html) 35 days after spinoff from [FLAML](https://github.com/microsoft/FLAML).

🎉 Nov 6, 2023: AutoGen is mentioned by Satya Nadella in a [fireside chat](https://youtu.be/0pLBvgYtv6U).

🎉 Nov 1, 2023: AutoGen is the top trending repo on GitHub in October 2023.

🎉 Oct 03, 2023: AutoGen spins off from [FLAML](https://github.com/microsoft/FLAML) on GitHub.

🎉 Aug 16: Paper about AutoGen on [arxiv](https://arxiv.org/abs/2308.08155).

🎉 Mar 29, 2023: AutoGen is first created in [FLAML](https://github.com/microsoft/FLAML).

🔥 FLAML is highlighted in OpenAI's [cookbook](https://github.com/openai/openai-cookbook#related-resources-from-around-the-web).

🔥 [autogen](https://docs.ag2.ai/) is released with support for ChatGPT and GPT-4, based on [Cost-Effective Hyperparameter Optimization for Large Language Model Generation Inference](https://arxiv.org/abs/2303.04673).

🔥 FLAML supports Code-First AutoML & Tuning – Private Preview in [Microsoft Fabric Data Science](https://learn.microsoft.com/en-us/fabric/data-science/).

## Contributors Wall

<a href="https://github.com/ag2ai/ag2/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=ag2ai/ag2&max=204" />
</a>

## Related Papers

[AutoGen](https://arxiv.org/abs/2308.08155)
[EcoOptiGen](https://arxiv.org/abs/2303.04673)
[MathChat](https://arxiv.org/abs/2306.01337)
[AgentOptimizer](https://arxiv.org/pdf/2402.11359)
[StateFlow](https://arxiv.org/abs/2403.11322)

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

<p align="right" style="font-size: 14px; color: #555; margin-top: 20px;">
  <a href="#readme-top" style="text-decoration: none; color: blue; font-weight: bold;">
    ↑ Back to Top ↑
  </a>
</p>
