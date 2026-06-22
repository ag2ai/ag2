---
title: Quick Start
sidebarTitle: Quick Start
---

Get up and running with AG2 in just **3 minutes**! This guide will help you set up your environment and build your very first agent workflow.

!!! tip "Prerequisites"
    Make sure you've [installed AG2](user-guide/basic-concepts/installing-ag2.md) before continuing. Need just 30 seconds for your provider of choice:

    === "OpenAI"

        ```
        pip install "ag2[openai]"
        ```

    === "Gemini"

        ```
        pip install "ag2[gemini]"
        ```

    === "Azure OpenAI"

        ```
        pip install "ag2[openai]"
        ```

### Set Up Your Environment

**Python Version**

AG2 requires **Python version >= 3.10, < 3.14**. We recommend using a virtual environment for your project to keep your packages contained. See <a href="https://docs.python.org/3/library/venv.html" target="_blank">venv</a>.

**Set Your API Key**

Make sure to set your LLM API key (and any provider-specific endpoint values) as environment variables. Pick the provider you plan to use:

=== "OpenAI"

    === "macOS / Linux"

        ```
        export OPENAI_API_KEY="YOUR_API_KEY"
        ```

    === "Windows"

        ```
        setx OPENAI_API_KEY "YOUR_API_KEY"
        ```

=== "Gemini"

    === "macOS / Linux"

        ```
        export GEMINI_API_KEY="YOUR_API_KEY"
        ```

    === "Windows"

        ```
        setx GEMINI_API_KEY "YOUR_API_KEY"
        ```

=== "Azure OpenAI"

    === "macOS / Linux"

        ```
        export AZURE_OPENAI_API_KEY="YOUR_API_KEY"
        export AZURE_OPENAI_ENDPOINT="https://YOUR-RESOURCE.openai.azure.com/"
        ```

    === "Windows"

        ```
        setx AZURE_OPENAI_API_KEY "YOUR_API_KEY"
        setx AZURE_OPENAI_ENDPOINT "https://YOUR-RESOURCE.openai.azure.com/"
        ```

### Build Your First Agent Workflow

Let's build a poetic AI assistant that responds in rhymes using AG2 and your model provider of choice.

Create a Python script called `first_agent.py`, and paste the code for your provider into it. The agent behaviour is identical across providers — only the `LLMConfig` differs.

=== "OpenAI"

    ```python
    import os

    from autogen import ConversableAgent, LLMConfig

    # 1. Define our LLM configuration for OpenAI's gpt-5-nano
    #    uses the OPENAI_API_KEY environment variable
    llm_config = LLMConfig(
        {"api_type": "openai", "model": "gpt-5-nano", "api_key": os.environ["OPENAI_API_KEY"]}
    )

    # 2. Create our LLM agent
    my_agent = ConversableAgent(
        name="helpful_agent",
        system_message="You are a poetic AI assistant, respond in rhyme.",
        llm_config=llm_config,
    )

    # 3. Run the agent with a prompt and process the response
    response = my_agent.run(
        message="In one sentence, what's the big deal about AI?",
        max_turns=3,
        user_input=True,
    )
    response.process()
    ```

=== "Gemini"

    ```python
    import os

    from autogen import ConversableAgent, LLMConfig

    # 1. Define our LLM configuration for Google Gemini
    #    uses the GEMINI_API_KEY environment variable
    llm_config = LLMConfig(
        {"api_type": "google", "model": "gemini-2.5-flash", "api_key": os.environ["GEMINI_API_KEY"]}
    )

    # 2. Create our LLM agent
    my_agent = ConversableAgent(
        name="helpful_agent",
        system_message="You are a poetic AI assistant, respond in rhyme.",
        llm_config=llm_config,
    )

    # 3. Run the agent with a prompt and process the response
    response = my_agent.run(
        message="In one sentence, what's the big deal about AI?",
        max_turns=3,
        user_input=True,
    )
    response.process()
    ```

=== "Azure OpenAI"

    ```python
    import os

    from autogen import ConversableAgent, LLMConfig

    # 1. Define our LLM configuration for Azure OpenAI
    #    "model" must match your Azure *deployment name*, not the underlying model name
    llm_config = LLMConfig(
        {
            "api_type": "azure",
            "model": "my-gpt-deployment",
            "api_key": os.environ["AZURE_OPENAI_API_KEY"],
            "base_url": os.environ["AZURE_OPENAI_ENDPOINT"],
            "api_version": "2025-01-01",
        }
    )

    # 2. Create our LLM agent
    my_agent = ConversableAgent(
        name="helpful_agent",
        system_message="You are a poetic AI assistant, respond in rhyme.",
        llm_config=llm_config,
    )

    # 3. Run the agent with a prompt and process the response
    response = my_agent.run(
        message="In one sentence, what's the big deal about AI?",
        max_turns=3,
        user_input=True,
    )
    response.process()
    ```

!!! note "Why `run()` then `process()`?"
    `run()` prepares the conversation, `process()` executes it. This two-step pattern gives you a chance to inspect or modify the workflow before it runs.

### Run Your Example

In your terminal, run:

```bash
python first_agent.py
```

If everything is set up correctly, the agent will reply to your initial message in rhyme, then prompt you for a response.
You can either:

- **Type a reply** — and the agent will respond in rhyme to your message
- **Press Enter** — to send an empty message to the agent, and see how it creatively responds
- **Type exit** — to end the conversation

The interaction continues for up to 3 turns (or until you exit).

#### Example Output

```console
user (to helpful_agent):

In one sentence, what's the big deal about AI?

--------------------------------------------------------------------------------

>>>>>>>> USING AUTO REPLY...
helpful_agent (to user):

AI transforms our world, enhancing life's parade,
With insights and solutions, it helps plans cascade.

--------------------------------------------------------------------------------
Replying as user. Provide feedback to helpful_agent. Press enter to skip and use auto-reply, or type 'exit' to end the conversation:
```

!!! example "Try it without installing"
    Want to see AG2 in action right away? The [AG2 Playground](https://playground.ag2.ai){:target="_blank"} lets you run multi-agent workflows live in your browser — no setup required.

### What's Next?

Now that you've built your first agent, here's where to go:

- **[Overview](user-guide/basic-concepts/overview.md)** — Understand the core building blocks of AG2
- **[LLM Configuration](user-guide/basic-concepts/llm-configuration.md)** — Configure different model providers (Gemini, Anthropic, etc.)
- **[Introducing Group Chat](user-guide/basic-concepts/introducing-group-chat.md)** — Make multiple agents collaborate together
