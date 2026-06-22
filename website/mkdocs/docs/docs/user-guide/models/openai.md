---
title: OpenAI API Integration Guide | Multi-Agent Framework
sidebarTitle: OpenAI
description: "How to use OpenAI models with AG2. Configuration, API setup, and code examples for building multi-agent systems with GPT-4o, o1, and other OpenAI models."
---

[OpenAI](https://openai.com/)'s API and models are fully supported in AG2. You will need an OpenAI account and to create an API key. [See their website for further details](https://openai.com/).

## Installation

!!! warning
    **From version 0.8**: The OpenAI package, `openai`, is not installed by default, install it with the `openai` extra as shown below.

``` bash
pip install ag2[openai]
```

!!! tip
    If you have been using `autogen` or `ag2`, all you need to do is upgrade it using:
    ```bash
    pip install -U autogen[openai]
    ```
    or
    ```bash
    pip install -U ag2[openai]
    ```
    as `autogen` and `ag2` are aliases for the same PyPI package.

## LLM Configurations

See the [LLM Configuration](../basic-concepts/llm-configuration.md) for further details on setting an creating an LLM configuration.

The below example of an `OAI_CONFIG_LIST` shows how the OpenAI client class is used.
By specifying the `api_type` as `openai`, the client will use the OpenAI ChatCompletion API.

```python
[
    {
        "api_type": "openai",
        "model": "o1",
        "api_key": "your OpenAI Key goes here",
    },
    {
        "api_type": "openai", # use OpenAI ChatCompletion API
        "model": "gpt-5-nano",
        "api_key": "your OpenAI Key goes here",
    }
]
```

As an alternative to the `api_key` key and value in the config, you can set the environment variable `OPENAI_API_KEY` to your OpenAI key.

Linux/Mac:
``` bash
export OPENAI_API_KEY="your_openai_api_key_here"
```

Windows:
``` bash
set OPENAI_API_KEY=your_openai_api_key_here
```

!!! tip
    Interested in trying out image generation with the new OpenAI Responses API? Check out the [OpenAPI Responses client](openai_responses.md).

## Extra Headers

You can pass custom HTTP headers on each API request using the `extra_headers` parameter. This is useful when working through API gateways, proxies, or OpenAI-compatible servers that require additional headers for authentication, routing, or observability.

```python
from autogen import LLMConfig

llm_config = LLMConfig(
    api_type="openai",
    model="gpt-4o-mini",
    api_key="your OpenAI Key goes here",
    extra_headers={
        "X-Custom-Header": "my-value",
    },
)
```

!!! note
    `extra_headers` is a per-request parameter passed on each call. This is different from `default_headers`, which is set once on the client constructor and applies to all requests made by that client.

## Examples

See the [ConversableAgent example](../basic-concepts/conversable-agent.md) in our [Basic Concepts](../basic-concepts/installing-ag2.md) documentation to see OpenAI's LLMs in use.
