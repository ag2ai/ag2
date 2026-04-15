---
name: build-research-agent
description: Build a complete web research agent team using AG2 with search tools, web crawling, and structured output. Use when the user wants a practical research or information-gathering workflow.
---

# Build Research Agent Team

You are an expert at building AG2 research workflows. When the user wants to build a research agent:

## 1. Choose the Right Approach

Ask the user:
- Simple single-query search? → Two agents + DuckDuckGoSearchTool
- Multi-query parallel research? → Two agents + QuickResearchTool
- Deep multi-step research? → Multi-agent team with AutoPattern

## 2. Simple Research Agent (Two Agents + Search Tool)

```python
import os
from typing import Annotated
from autogen import ConversableAgent, LLMConfig
from autogen.tools.experimental import DuckDuckGoSearchTool

llm_config = LLMConfig(
    {"model": "gpt-4o-mini", "api_key": os.environ["OPENAI_API_KEY"]}
)

assistant = ConversableAgent(
    name="researcher",
    system_message="You are a research assistant. Use the search tool to find information, then synthesize a clear answer.",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

user = ConversableAgent(
    name="user",
    llm_config=False,
    human_input_mode="NEVER",
    is_termination_msg=lambda msg: True,  # End after first response
)

# Register pre-built search tool
search = DuckDuckGoSearchTool()
search.register_for_llm(assistant)
search.register_for_execution(user)

result = await user.a_run(assistant, message="What are the latest trends in AI agents?")
await result.process()
```

Requires: `pip install ag2[openai,duckduckgo_search]`

## 3. Parallel Research Agent (QuickResearchTool)

```python
import os
from autogen import ConversableAgent, LLMConfig
from autogen.tools.experimental import QuickResearchTool

llm_config = LLMConfig(
    {"model": "gpt-4o-mini", "api_key": os.environ["OPENAI_API_KEY"]}
)

assistant = ConversableAgent(
    name="researcher",
    system_message="You are a research assistant. Break down the user's question into multiple search queries and use the research tool to find comprehensive answers.",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

user = ConversableAgent(
    name="user",
    llm_config=False,
    human_input_mode="NEVER",
    is_termination_msg=lambda msg: True,
)

# QuickResearchTool searches multiple queries in parallel using Tavily + crawl4ai
research = QuickResearchTool(
    llm_config=llm_config,
    tavily_api_key=os.environ["TAVILY_API_KEY"],
    num_results_per_query=3,
)
research.register_for_llm(assistant)
research.register_for_execution(user)

result = await user.a_run(assistant, message="Compare React, Vue, and Svelte for 2025")
await result.process()
```

Requires: `pip install ag2[openai,quick-research]`

## 4. Multi-Agent Research Team (Group Chat)

```python
import os
from typing import Annotated
from autogen import ConversableAgent, LLMConfig
from autogen.agentchat import run_group_chat
from autogen.agentchat.group.patterns import AutoPattern
from autogen.tools.experimental import DuckDuckGoSearchTool

llm_config = LLMConfig(
    {"model": "gpt-4o-mini", "api_key": os.environ["OPENAI_API_KEY"]}
)

planner = ConversableAgent(
    name="planner",
    system_message="You break research tasks into specific questions. Output a numbered list of questions to investigate.",
    llm_config=llm_config,
    human_input_mode="NEVER",
    description="Plans research by breaking down topics into questions. Call first.",
)

researcher = ConversableAgent(
    name="researcher",
    system_message="You search for information to answer specific questions. Use the search tool for each question.",
    llm_config=llm_config,
    human_input_mode="NEVER",
    description="Searches the web for answers. Call after planner has listed questions.",
)

writer = ConversableAgent(
    name="writer",
    system_message="You synthesize research findings into a clear, well-structured report. When done, end with TERMINATE.",
    llm_config=llm_config,
    human_input_mode="NEVER",
    description="Writes the final report from gathered research. Call after researcher has findings.",
    is_termination_msg=lambda msg: "TERMINATE" in msg.get("content", ""),
)

user = ConversableAgent(name="user", human_input_mode="NEVER", llm_config=False)

# Register search on the researcher
search = DuckDuckGoSearchTool()
search.register_for_llm(researcher)
search.register_for_execution(researcher)

result = run_group_chat(
    pattern=AutoPattern(
        initial_agent=planner,
        agents=[planner, researcher, writer],
        user_agent=user,
        group_manager_args={"llm_config": llm_config},
    ),
    messages="Research the current state of quantum computing.",
    max_rounds=15,
)
```

## 5. Rules

- Always install the right extras: `ag2[duckduckgo_search]`, `ag2[quick-research]`, `ag2[tavily]`
- DuckDuckGoSearchTool needs no API key — best for getting started
- QuickResearchTool needs a Tavily API key but gives much richer results
- For group chat research teams, register search on the researcher agent specifically
- Set `max_rounds` to prevent infinite loops
- Include a termination condition in the writer/final agent
