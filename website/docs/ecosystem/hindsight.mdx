---
title: "Hindsight: Long-Term Memory for AG2 Agents"
---

[Hindsight](https://github.com/vectorize-io/hindsight) is an open-source (MIT) long-term memory engine for AI agents. It automatically extracts facts from conversations, builds entity graphs, and retrieves relevant context using four parallel strategies (semantic, BM25, graph traversal, temporal).

Hindsight runs locally via Docker, as a pip-installable server, or embedded in your Python process.

|                                          |                                                                   |
| ---------------------------------------- | ----------------------------------------------------------------- |
| 🧠 **Multi-Strategy Retrieval**          | Four parallel recall strategies: semantic, BM25, graph, temporal  |
| 📝 **Automatic Fact Extraction**         | Extracts structured facts and entities from raw conversations     |
| 🔗 **Entity Graph**                      | Builds and traverses entity relationship graphs across memories   |
| 🪞 **Reflect**                           | LLM-powered synthesis of coherent answers from stored memories    |
| 🏠 **Self-Hosted**                       | Run locally with Docker or pip, no cloud account required         |

## Installation

1. **Install the integration package:**

```bash
pip install hindsight-ag2
```

2. **Start Hindsight locally:**

```bash
docker run --rm -it --pull always -p 8888:8888 \
  -e HINDSIGHT_API_LLM_API_KEY=$OPENAI_API_KEY \
  -v $HOME/.hindsight-docker:/home/hindsight/.pg0 \
  ghcr.io/vectorize-io/hindsight:latest
```

See the [Hindsight quick start](https://github.com/vectorize-io/hindsight#quick-start) for more deployment options.

## AG2 + Hindsight Example

This example shows how to give AG2 agents persistent long-term memory using `register_hindsight_tools`.

```python
import os
from autogen import AssistantAgent, UserProxyAgent, LLMConfig
from hindsight_ag2 import register_hindsight_tools

os.environ["OPENAI_API_KEY"] = "your_openai_api_key"

llm_config = LLMConfig(api_type="openai", model="gpt-4o")

with llm_config:
    assistant = AssistantAgent(
        name="assistant",
        system_message="You are a helpful assistant with long-term memory.",
    )
    user_proxy = UserProxyAgent(
        name="user",
        human_input_mode="NEVER",
    )

# Register Hindsight memory tools on both agents
register_hindsight_tools(
    assistant,
    user_proxy,
    bank_id="my-memory-bank",
    hindsight_api_url="http://localhost:8888",
)

# The assistant can now use hindsight_retain, hindsight_recall, hindsight_reflect
result = user_proxy.initiate_chat(
    assistant,
    message="Remember that I prefer Python over JavaScript and use VS Code.",
)
```

The agent gets three tools:

| Tool | Description |
| :--- | :--- |
| `hindsight_retain` | Store facts, preferences, or decisions to long-term memory |
| `hindsight_recall` | Search memories using multi-strategy retrieval |
| `hindsight_reflect` | Synthesize a reasoned answer from stored memories |

### GroupChat with Shared Memory

Multiple agents can share the same memory bank:

```python
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager, LLMConfig
from hindsight_ag2 import register_hindsight_tools

llm_config = LLMConfig(api_type="openai", model="gpt-4o")

with llm_config:
    researcher = AssistantAgent(name="researcher", system_message="You research topics.")
    writer = AssistantAgent(name="writer", system_message="You write content.")
    executor = UserProxyAgent(name="executor", human_input_mode="NEVER")

# All agents share the same memory bank
for agent in [researcher, writer]:
    register_hindsight_tools(agent, executor, bank_id="team-memory")

group_chat = GroupChat(agents=[researcher, writer, executor], messages=[])
manager = GroupChatManager(groupchat=group_chat)
```

## Resources

- [Hindsight GitHub](https://github.com/vectorize-io/hindsight)
- [hindsight-ag2 on PyPI](https://pypi.org/project/hindsight-ag2/)
- [Integration documentation](https://github.com/vectorize-io/hindsight/tree/main/hindsight-integrations/ag2)
