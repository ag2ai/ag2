---
title: Overview
sidebarTitle: Overview
---

### Understanding Tool Usage in AG2

Agents significantly enhance their capabilities by leveraging tools, which provide access to external data, APIs, and additional functionality.

In **AG2**, tool usage happens in two stages:

- An agent suggests which tool to use (via its LLM).
- Another agent executes the selected tool.

Typically, you'll create two agents. One to determine the appropriate tool and another to carry out the execution.

!!! note
    In a conversation, the executor agent must always follow the agent that suggests a tool.

### Example: Implementing a Date Tool

```python
from datetime import datetime
from typing import Annotated

from autogen import ConversableAgent, register_function, LLMConfig
from dotenv import load_dotenv
import os
load_dotenv()

# Put your key in the OPENAI_API_KEY environment variable
llm_config = LLMConfig(config_list={"api_type": "openai", "model": "gpt-5-nano","api_key":os.getenv("OPENAI_API_KEY")})

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
    max_turns=2,
)

print(chat_result.chat_history[-1]["content"])
```

<Example/>

1. We define a tool, a function that will be attached to our agents. The `Annotated` parameter is included in the LLM call to ensure it understands the purpose of `date_string`.

2. The `date_agent` decides whether to use the tool based on its LLM reasoning.

3. The `executor_agent` executes the tool and returns the output as its response.

4. We register the tool with the agents and provide a description to help the LLM determine when to use it.

5. Since this is a two-way conversation, the `executor_agent` follows the `date_agent`. If the `date_agent` suggests using the tool, the `executor_agent` executes it accordingly.

    ```console
    executor_agent (to date_agent):

    I was born on the 25th of March 1995, what day was it?

    --------------------------------------------------------------------------------

    >>>>>>>> USING AUTO REPLY...
    date_agent (to executor_agent):

    ***** Suggested tool call (call_iOOZMTCoIVVwMkkSVu04Krj8): get_weekday *****
    Arguments:
    {"date_string":"1995-03-25"}
    ****************************************************************************

    --------------------------------------------------------------------------------

    >>>>>>>> EXECUTING FUNCTION get_weekday...
    Call ID: call_iOOZMTCoIVVwMkkSVu04Krj8
    Input arguments: {'date_string': '1995-03-25'}
    executor_agent (to date_agent):

    ***** Response from calling tool (call_iOOZMTCoIVVwMkkSVu04Krj8) *****
    Saturday
    **********************************************************************

    --------------------------------------------------------------------------------

    >>>>>>>> USING AUTO REPLY...
    date_agent (to executor_agent):

    It was a Saturday.

    --------------------------------------------------------------------------------
    ```

### Alternative Registration Methods

Alternatively, you can use decorators `register_for_execution` and `register_for_llm` to register a tool. So, instead of using `register_function`, you can register them with the function definition.
```python
@date_agent.register_for_llm(description="Get the day of the week for a given date")
@executor_agent.register_for_execution()
def get_weekday(date_string: Annotated[str, "Format: YYYY-MM-DD"]) -> str:
    date = datetime.strptime(date_string, '%Y-%m-%d')
    return date.strftime('%A')
```

### Client-specific built-in tools

AG2's OpenAI Responses client offers support for the following tools through the `built_in_tools` parameter in `LLMConfig`:

- web_search
- image_generation
- apply_patch

An example of how to use built_in_tools:

```python
import os
from pathlib import Path

from dotenv import load_dotenv

from autogen import ConversableAgent, LLMConfig

load_dotenv()

# configure LLMConfig to use built_in_tools
llm_config = LLMConfig(
    config_list={
        "api_type": "responses_v2",
        "model": "gpt-5.1",
        "api_key": os.getenv("OPENAI_API_KEY"),
        "built_in_tools": ["apply_patch"],
        "workspace_dir": "./my_project_folder",
    },
)

coding_agent = ConversableAgent(
    name="coding_assistant",
    llm_config=llm_config,
    system_message="""You are a helpful coding assistant...""",
)

# Tool is automatically registered! Just use it:
result = coding_agent.initiate_chat(
    recipient=coding_agent,
    message="""
    Create app.py in the workspace directory,
    create a app.yaml file,
    create a app.sh file,
    create a folder /tests,
    create tests/test_app.py
    """,
    max_turns=2,
)
```
