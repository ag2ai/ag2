---
title: "Quick Start"
---

AG2 (formerly AutoGen) is an open-source programming framework for building AI agents and facilitating their cooperation to solve tasks. AG2 supports tool use, autonomous and human-in-the-loop workflows, and multi-agent conversation patterns.

### Let's go

```sh
pip install ag2
```
!!! note
    If you have been using `autogen` or `pyautogen`, all you need to do is upgrade it using:
    ```bash
    pip install -U autogen
    ```
    or
    ```bash
    pip install -U pyautogen
    ```
    as `pyautogen`, `autogen`, and `ag2` are aliases for the same PyPI package.

=== "Chat with an agent"

    ```python
    # 1. Import our agent class
    from autogen import ConversableAgent

    # 2. Define our LLM configuration for OpenAI's GPT-4o mini
    #    uses the OPENAI_API_KEY environment variable
    llm_config = {"api_type": "openai", "model": "gpt-4o-mini"}

    # 3. Create our LLM agent
    my_agent = ConversableAgent(
        name="helpful_agent",
        llm_config=llm_config,
        system_message="You are a poetic AI assistant, respond in rhyme.",
    )

    # 4. Run the agent with a prompt
    chat_result = my_agent.run("In one sentence, what's the big deal about AI?")

    # 5. Print the chat
    print(chat_result.chat_history)
    ```

=== "Two agent chat"

    ```python
    # Chat between two comedian agents

    # 1. Import our agent class
    from autogen import ConversableAgent

    # 2. Define our LLM configuration for OpenAI's GPT-4o mini,
    #    uses the OPENAI_API_KEY environment variable
    llm_config = {"api_type": "openai", "model": "gpt-4o-mini"}

    # 3. Create our agents who will tell each other jokes,
    #    with Jack ending the chat when Emma says FINISH
    jack = ConversableAgent(
        "Jack",
        llm_config=llm_config,
        system_message=(
        "Your name is Jack and you are a comedian "
        "in a two-person comedy show."
        ),
        is_termination_msg=lambda x: True if "FINISH" in x["content"] else False
    )
    emma = ConversableAgent(
        "Emma",
        llm_config=llm_config,
        system_message=(
        "Your name is Emma and you are a comedian "
        "in a two-person comedy show. Say the word FINISH "
        "ONLY AFTER you've heard 2 of Jack's jokes."
        ),
    )

    # 4. Run the chat
    chat_result = jack.initiate_chat(
        emma,
        message="Emma, tell me a joke about goldfish and peanut butter.",
    )

    # 5. Print the chat
    print(chat_result.chat_history)
    ```

=== "Group chat"

    ```python
    # Group chat amongst agents to create a 4th grade lesson plan
    # Flow determined by Group Chat Manager automatically, and
    # should be Teacher > Planner > Reviewer > Teacher (repeats if necessary)

    # 1. Import our agent and group chat classes
    from autogen import ConversableAgent, GroupChat, GroupChatManager

    # Define our LLM configuration for OpenAI's GPT-4o mini
    # uses the OPENAI_API_KEY environment variable
    llm_config = {"api_type": "openai", "model": "gpt-4o-mini"}

    # Planner agent setup
    planner_message = "Create lesson plans for 4th grade. Use format: <title>, <learning_objectives>, <script>"
    planner = ConversableAgent(
        name="planner_agent",
        llm_config=llm_config,
        system_message=planner_message,
        description="Creates lesson plans"
    )

    # Reviewer agent setup
    reviewer_message = "Review lesson plans against 4th grade curriculum. Provide max 3 changes."
    reviewer = ConversableAgent(
        name="reviewer_agent",
        llm_config=llm_config,
        system_message=reviewer_message,
        description="Reviews lesson plans"
    )

    # Teacher agent setup
    teacher_message = "Choose topics and work with planner and reviewer. Say DONE! when finished."
    teacher = ConversableAgent(
        name="teacher_agent",
        llm_config=llm_config,
        system_message=teacher_message,
    )

    # Setup group chat
    groupchat = GroupChat(
        agents=[teacher, planner, reviewer],
        speaker_selection_method="auto",
        messages=[]
    )

    # Create manager
    # At each turn, the manager will check if the message contains DONE! and end the chat if so
    # Otherwise, it will select the next appropriate agent using its LLM
    manager = GroupChatManager(
        name="group_manager",
        groupchat=groupchat,
        llm_config=llm_config,
        is_termination_msg=lambda x: "DONE!" in (x.get("content", "") or "").upper()
    )

    # Start the conversation
    chat_result = teacher.initiate_chat(
        recipient=manager,
        message="Let's teach the kids about the solar system."
    )

    # Print the chat
    print(chat_result.chat_history)
    ```

!!! tip
    Learn more about configuring LLMs for agents
    [here](/docs/user-guide/basic-concepts/llm-configuration).


### Where to Go Next?

- Go through the [basic concepts](/docs/user-guide/basic-concepts/installing-ag2) to get started
- Once you're ready, hit the [advanced concepts](/docs/user-guide/advanced-concepts/rag)
- Explore the [API Reference](/docs/api-reference/autogen/overview)
- Chat on [Discord](https://discord.gg/pAbnFJrkgZ)
- Follow on [X](https://x.com/ag2oss)

If you like our project, please give it a [star](https://github.com/ag2ai/ag2) on GitHub. If you are interested in contributing, please read [Contributor's Guide](/contributor-guide/contributing).

<iframe
  src="https://ghbtns.com/github-btn.html?user=ag2ai&amp;repo=ag2&amp;type=star&amp;count=true&amp;size=large"
  frameborder="0"
  scrolling="0"
  width="170"
  height="30"
  title="GitHub"
></iframe>
