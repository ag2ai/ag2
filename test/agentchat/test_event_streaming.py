# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
import pytest

from autogen.agentchat.contrib.swarm_agent import AfterWorkOption, a_run_swarm, run_swarm
from autogen.agentchat.conversable_agent import ConversableAgent
from autogen.agentchat.groupchat import GroupChat, GroupChatManager
from autogen.agentchat.user_proxy_agent import UserProxyAgent
from autogen.import_utils import run_for_optional_imports
from autogen.io.run_response import AsyncRunResponseProtocol
from autogen.llm_config import LLMConfig
from test.conftest import Credentials


@run_for_optional_imports("openai", "openai")
def test_single_agent_sync(credentials_gpt_4o_mini: Credentials):
    llm_config = LLMConfig(**credentials_gpt_4o_mini.llm_config)

    with llm_config:
        my_agent = ConversableAgent(
            name="helpful_agent",
            system_message="You are a poetic AI assistant, respond in rhyme.",
        )

    # 2. Run the agent with a prompt
    response = my_agent.run(message="In one sentence, what's the big deal about AI?", max_turns=1)

    for event in response.events:
        if event.type == "input_request":
            event.content.respond("exit")

    assert response.summary is not None, "Summary should not be None"
    assert len(response.messages) == 2, "Messages should not be empty"
    assert isinstance(response.last_speaker, ConversableAgent), "Last speaker should be an agent"


@pytest.mark.asyncio
@run_for_optional_imports("openai", "openai")
async def test_single_agent_async(credentials_gpt_4o_mini: Credentials):
    llm_config = LLMConfig(**credentials_gpt_4o_mini.llm_config)

    with llm_config:
        my_agent = ConversableAgent(
            name="helpful_agent",
            system_message="You are a poetic AI assistant, respond in rhyme.",
        )

    # 2. Run the agent with a prompt
    response = await my_agent.a_run(message="In one sentence, what's the big deal about AI?", max_turns=1)

    async for event in response.events:
        if event.type == "input_request":
            await event.content.respond("exit")

    assert await response.summary is not None, "Summary should not be None"
    assert len(await response.messages) == 2, "Messages should not be empty"
    assert isinstance(await response.last_speaker, ConversableAgent), "Last speaker should be an agent"


@run_for_optional_imports("openai", "openai")
def test_two_agents_sync(credentials_gpt_4o_mini: Credentials):
    llm_config = LLMConfig(**credentials_gpt_4o_mini.llm_config)

    with llm_config:
        jack = ConversableAgent(
            "Jack",
            system_message=("Your name is Jack and you are a comedian in a two-person comedy show."),
            is_termination_msg=lambda x: "FINISH" in x["content"],
        )
        emma = ConversableAgent(
            "Emma",
            system_message=(
                "Your name is Emma and you are a comedian "
                "in a two-person comedy show. Say the word FINISH "
                "ONLY AFTER you've heard 2 of Jack's jokes."
            ),
        )

    # 3. Run the chat
    response = jack.run(emma, message="Emma, tell me a joke about goldfish and peanut butter.")

    for event in response.events:
        if event.type == "input_request":
            event.content.respond("exit")

    assert response.last_speaker in [jack, emma], "Last speaker should be one of the agents"
    assert response.summary is not None, "Summary should not be None"
    assert len(response.messages) > 0, "Messages should not be empty"


@pytest.mark.asyncio
@run_for_optional_imports("openai", "openai")
async def test_two_agents_async(credentials_gpt_4o_mini: Credentials):
    llm_config = LLMConfig(**credentials_gpt_4o_mini.llm_config)

    with llm_config:
        jack = ConversableAgent(
            "Jack",
            system_message=("Your name is Jack and you are a comedian in a two-person comedy show."),
            is_termination_msg=lambda x: "FINISH" in x["content"],
            human_input_mode="NEVER",
        )
        emma = ConversableAgent(
            "Emma",
            system_message=(
                "Your name is Emma and you are a comedian "
                "in a two-person comedy show. Say the word FINISH "
                "ONLY AFTER you've heard 2 of Jack's jokes."
            ),
            human_input_mode="NEVER",
        )

    # 4. Run the chat
    response: AsyncRunResponseProtocol = await jack.a_run(
        emma, message="Emma, tell me a joke about goldfish and peanut butter."
    )

    async for event in response.events:
        if event.type == "input_request":
            await event.content.respond("exit")

    assert await response.last_speaker in [jack, emma], "Last speaker should be one of the agents"
    assert await response.summary is not None, "Summary should not be None"
    assert len(await response.messages) > 0, "Messages should not be empty"


@run_for_optional_imports("openai", "openai")
def test_group_chat_sync(credentials_gpt_4o_mini: Credentials):
    llm_config = LLMConfig(**credentials_gpt_4o_mini.llm_config)

    with llm_config:
        # Planner agent setup
        planner_message = "Create lesson plans for 4th grade. Use format: <title>, <learning_objectives>, <script>"
        planner = ConversableAgent(
            name="planner_agent", system_message=planner_message, description="Creates lesson plans"
        )

        # Reviewer agent setup
        reviewer_message = "Review lesson plans against 4th grade curriculum. Provide max 3 changes."
        reviewer = ConversableAgent(
            name="reviewer_agent", system_message=reviewer_message, description="Reviews lesson plans"
        )

        # Teacher agent setup
        teacher_message = "Choose topics and work with planner and reviewer. Say DONE! when finished."
        teacher = ConversableAgent(
            name="teacher_agent",
            system_message=teacher_message,
        )

    # Setup group chat
    groupchat = GroupChat(agents=[teacher, planner, reviewer], speaker_selection_method="auto", messages=[])

    # Create manager
    # At each turn, the manager will check if the message contains DONE! and end the chat if so
    # Otherwise, it will select the next appropriate agent using its LLM
    manager = GroupChatManager(
        name="group_manager",
        groupchat=groupchat,
        llm_config=llm_config,
        is_termination_msg=lambda x: "DONE!" in (x.get("content", "") or "").upper(),
    )

    # Start the conversation
    response = teacher.run(recipient=manager, message="Let's teach the kids about the solar system.")

    for event in response.events:
        if event.type == "input_request":
            event.content.respond("exit")

    assert response.summary is not None, "Summary should not be None"
    assert len(response.messages) > 0, "Messages should not be empty"
    assert isinstance(response.last_speaker, ConversableAgent), "Last speaker should be an agent"


@pytest.mark.asyncio
@run_for_optional_imports("openai", "openai")
async def test_group_chat_async(credentials_gpt_4o_mini: Credentials):
    llm_config = LLMConfig(**credentials_gpt_4o_mini.llm_config)

    with llm_config:
        # Planner agent setup
        planner_message = "Create lesson plans for 4th grade. Use format: <title>, <learning_objectives>, <script>"
        planner = ConversableAgent(
            name="planner_agent", system_message=planner_message, description="Creates lesson plans"
        )

        # Reviewer agent setup
        reviewer_message = "Review lesson plans against 4th grade curriculum. Provide max 3 changes."
        reviewer = ConversableAgent(
            name="reviewer_agent", system_message=reviewer_message, description="Reviews lesson plans"
        )

        # Teacher agent setup
        teacher_message = "Choose topics and work with planner and reviewer. Say DONE! when finished."
        teacher = ConversableAgent(
            name="teacher_agent",
            system_message=teacher_message,
        )

    # Setup group chat
    groupchat = GroupChat(agents=[teacher, planner, reviewer], speaker_selection_method="auto", messages=[])

    # Create manager
    # At each turn, the manager will check if the message contains DONE! and end the chat if so
    # Otherwise, it will select the next appropriate agent using its LLM
    manager = GroupChatManager(
        name="group_manager",
        groupchat=groupchat,
        llm_config=llm_config,
        is_termination_msg=lambda x: "DONE!" in (x.get("content", "") or "").upper(),
    )

    # Start the conversation
    response = await teacher.a_run(recipient=manager, message="Let's teach the kids about the solar system.")

    async for event in response.events:
        if event.type == "input_request":
            await event.content.respond("exit")

    assert await response.summary is not None, "Summary should not be None"
    assert len(await response.messages) > 0, "Messages should not be empty"
    assert isinstance(await response.last_speaker, ConversableAgent), "Last speaker should be one of the agents"


@run_for_optional_imports("openai", "openai")
def test_swarm_sync(credentials_gpt_4o_mini: Credentials):
    llm_config = LLMConfig(**credentials_gpt_4o_mini.llm_config)

    planner_message = """You are a classroom lesson planner.
    Given a topic, write a lesson plan for a fourth grade class.
    If you are given revision feedback, update your lesson plan and record it.
    Use the following format:
    <title>Lesson plan title</title>
    <learning_objectives>Key learning objectives</learning_objectives>
    <script>How to introduce the topic to the kids</script>
    """

    reviewer_message = """You are a classroom lesson reviewer.
    You compare the lesson plan to the fourth grade curriculum
    and provide a maximum of 3 recommended changes for each review.
    Make sure you provide recommendations each time the plan is updated.
    """

    teacher_message = """You are a classroom teacher.
    You decide topics for lessons and work with a lesson planner.
    and reviewer to create and finalise lesson plans.
    """

    with llm_config:
        lesson_planner = ConversableAgent(name="planner_agent", system_message=planner_message)

        lesson_reviewer = ConversableAgent(name="reviewer_agent", system_message=reviewer_message)

        teacher = ConversableAgent(
            name="teacher_agent",
            system_message=teacher_message,
        )

    # 2. Initiate the swarm chat using a swarm manager who will
    # select agents automatically
    response = run_swarm(
        initial_agent=teacher,
        agents=[lesson_planner, lesson_reviewer, teacher],
        messages="Today, let's introduce our kids to the solar system.",
        max_rounds=10,
        swarm_manager_args={"llm_config": llm_config},
        after_work=AfterWorkOption.SWARM_MANAGER,
    )

    for events in response.events:
        if events.type == "input_request":
            events.content.respond("exit")

    assert response.summary is not None, "Summary should not be None"
    assert len(response.messages) > 0, "Messages should not be empty"
    assert isinstance(response.last_speaker, ConversableAgent), "Last speaker should be an agent"


@pytest.mark.asyncio
@run_for_optional_imports("openai", "openai")
async def test_swarm_async(credentials_gpt_4o_mini: Credentials):
    llm_config = LLMConfig(**credentials_gpt_4o_mini.llm_config)

    # 1. Create our agents
    planner_message = """You are a classroom lesson planner.
    Given a topic, write a lesson plan for a fourth grade class.
    If you are given revision feedback, update your lesson plan and record it.
    Use the following format:
    <title>Lesson plan title</title>
    <learning_objectives>Key learning objectives</learning_objectives>
    <script>How to introduce the topic to the kids</script>
    """

    reviewer_message = """You are a classroom lesson reviewer.
    You compare the lesson plan to the fourth grade curriculum
    and provide a maximum of 3 recommended changes for each review.
    Make sure you provide recommendations each time the plan is updated.
    """

    teacher_message = """You are a classroom teacher.
    You decide topics for lessons and work with a lesson planner.
    and reviewer to create and finalise lesson plans.
    """

    with llm_config:
        lesson_planner = ConversableAgent(name="planner_agent", system_message=planner_message)

        lesson_reviewer = ConversableAgent(name="reviewer_agent", system_message=reviewer_message)

        teacher = ConversableAgent(
            name="teacher_agent",
            system_message=teacher_message,
        )

    # 2. Initiate the swarm chat using a swarm manager who will
    # select agents automatically
    response = await a_run_swarm(
        initial_agent=teacher,
        agents=[lesson_planner, lesson_reviewer, teacher],
        messages="Today, let's introduce our kids to the solar system.",
        max_rounds=10,
        swarm_manager_args={"llm_config": llm_config},
        after_work=AfterWorkOption.SWARM_MANAGER,
    )

    async for event in response.events:
        if event.type == "input_request":
            await event.content.respond("exit")

    assert await response.summary is not None, "Summary should not be None"
    assert len(await response.messages) > 0, "Messages should not be empty"
    assert isinstance(await response.last_speaker, ConversableAgent), "Last speaker should be one of the agents"


@run_for_optional_imports("openai", "openai")
def test_sequential_sync(credentials_gpt_4o_mini: Credentials):
    llm_config = LLMConfig(**credentials_gpt_4o_mini.llm_config)

    financial_tasks = [
        """What are the current stock prices of NVDA and TESLA, and how is the performance over the past month in terms of percentage change?""",
        """Investigate possible reasons of the stock performance.""",
    ]

    writing_tasks = ["""Develop an engaging blog post using any information provided."""]

    with llm_config:
        financial_assistant = ConversableAgent(
            name="Financial_assistant",
            system_message="You are a financial assistant, helping with stock market analysis. Reply 'TERMINATE' when financial tasks are done.",
            llm_config=llm_config,
        )
        research_assistant = ConversableAgent(
            name="Researcher",
            system_message="You are a research assistant, helping with stock market analysis. Reply 'TERMINATE' when research tasks are done.",
            llm_config=llm_config,
        )
        writer = ConversableAgent(
            name="writer",
            llm_config=llm_config,
            system_message="""
                You are a professional writer, known for
                your insightful and engaging articles.
                You transform complex concepts into compelling narratives.
                Reply "TERMINATE" in the end when everything is done.
                """,
        )

        user = UserProxyAgent(
            name="User",
            human_input_mode="NEVER",
            is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
            code_execution_config={
                "last_n_messages": 1,
                "work_dir": "tasks",
                "use_docker": False,
            },
        )

    responses = user.sequential_run([
        {
            "chat_id": 1,
            "recipient": financial_assistant,
            "message": financial_tasks[0],
            "silent": False,
            "summary_method": "reflection_with_llm",
        },
        {
            "chat_id": 2,
            "prerequisites": [1],
            "recipient": research_assistant,
            "message": financial_tasks[1],
            "silent": False,
            "summary_method": "reflection_with_llm",
        },
        {"chat_id": 3, "prerequisites": [1, 2], "recipient": writer, "silent": False, "message": writing_tasks[0]},
    ])

    for response in responses:
        for event in response.events:
            if event.type == "input_request":
                event.content.respond("exit")

    for response in responses:
        assert len(response.messages) > 0, "Messages should not be empty"
        assert isinstance(response.last_speaker, ConversableAgent), "Last speaker should be an agent"
        assert response.summary is not None, "Summary should not be None"


@pytest.mark.asyncio
@run_for_optional_imports("openai", "openai")
async def test_sequential_async(credentials_gpt_4o_mini: Credentials):
    llm_config = LLMConfig(**credentials_gpt_4o_mini.llm_config)

    financial_tasks = [
        """What are the current stock prices of NVDA and TESLA, and how is the performance over the past month in terms of percentage change?""",
        """Investigate possible reasons of the stock performance.""",
    ]

    writing_tasks = ["""Develop an engaging blog post using any information provided."""]

    with llm_config:
        financial_assistant = ConversableAgent(
            name="Financial_assistant",
            system_message="You are a financial assistant, helping with stock market analysis. Reply 'TERMINATE' when financial tasks are done.",
            llm_config=llm_config,
        )
        research_assistant = ConversableAgent(
            name="Researcher",
            system_message="You are a research assistant, helping with stock market analysis. Reply 'TERMINATE' when research tasks are done.",
            llm_config=llm_config,
        )
        writer = ConversableAgent(
            name="writer",
            llm_config=llm_config,
            system_message="""
                You are a professional writer, known for
                your insightful and engaging articles.
                You transform complex concepts into compelling narratives.
                Reply "TERMINATE" in the end when everything is done.
                """,
        )

        user = UserProxyAgent(
            name="User",
            human_input_mode="NEVER",
            is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
            code_execution_config={
                "last_n_messages": 1,
                "work_dir": "tasks",
                "use_docker": False,
            },
        )

    responses = await user.a_sequential_run([
        {
            "chat_id": 1,
            "recipient": financial_assistant,
            "message": financial_tasks[0],
            "silent": False,
            "summary_method": "reflection_with_llm",
        },
        {
            "chat_id": 2,
            "prerequisites": [1],
            "recipient": research_assistant,
            "message": financial_tasks[1],
            "silent": False,
            "summary_method": "reflection_with_llm",
        },
        {"chat_id": 3, "prerequisites": [1, 2], "recipient": writer, "silent": False, "message": writing_tasks[0]},
    ])

    for response in responses:
        async for event in response.events:
            if event.type == "input_request":
                await event.content.respond("exit")

        assert len(await response.messages) > 0, "Messages should not be empty"
        assert isinstance(await response.last_speaker, ConversableAgent), "Last speaker should be one of the agents"
        assert await response.summary is not None, "Summary should not be None"
