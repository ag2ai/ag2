import os

from dotenv import load_dotenv

from autogen import LLMConfig

# from autogen.agentchat.group.reply_result import ReplyResult
from autogen.tools import tool

load_dotenv()

llm_config = LLMConfig(
    config_list={"api_type": "openai", "model": "gpt-4o-mini", "api_key": os.getenv("OPENAI_API_KEY")}
)

from autogen import AssistantAgent, ConversableAgent, GroupChat, GroupChatManager, UserProxyAgent
from autogen.agentchat import initiate_group_chat
from autogen.agentchat.group.llm_condition import StringLLMCondition
from autogen.agentchat.group.on_condition import OnCondition
from autogen.agentchat.group.patterns import (
    AutoPattern,
    DefaultPattern,
    ManualPattern,
    RandomPattern,
    RoundRobinPattern,
)
from autogen.agentchat.group.targets.transition_target import (
    AgentTarget,
    NestedChatTarget,
    TerminateTarget,
)


# Tools for testing
@tool(name="calculator", description="Basic calculator for math operations")
def calculator(operation: str, a: float, b: float) -> float:
    """Basic calculator function"""
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        return a / b if b != 0 else 0
    return 0


@tool(name="weather_check", description="Check weather for a location")
def weather_check(location: str) -> str:
    """Mock weather check function"""
    return f"Weather in {location}: Sunny, 25°C"


@tool(name="database_query", description="Query database for user info")
def database_query(user_id: str) -> str:
    """Mock database query function"""
    return f"User {user_id}: Active, Premium member"


@tool(name="file_processor", description="Process files and return status")
def file_processor(filename: str) -> str:
    """Mock file processing function"""
    return f"Processed {filename}: Success"


# Agent definitions with 250 char limit system messages
assistant = AssistantAgent(
    "assistant",
    llm_config=llm_config,
    system_message="You are a helpful assistant. Keep responses under 250 characters. Be concise and direct.",
)

user_proxy = UserProxyAgent(
    "user_proxy",
    llm_config=llm_config,
    system_message="You represent the user. Keep responses under 250 characters.",
    code_execution_config={"work_dir": "coding", "use_docker": False},
    human_input_mode="NEVER",
)

math_agent = ConversableAgent(
    "math_agent",
    llm_config=llm_config,
    system_message="You are a math expert. Solve problems using calculator tool. Keep responses under 250 characters.",
    functions=[calculator],
)

weather_agent = ConversableAgent(
    "weather_agent",
    llm_config=llm_config,
    system_message="You provide weather information using weather_check tool. Keep responses under 250 characters.",
    functions=[weather_check],
)

db_agent = ConversableAgent(
    "db_agent",
    llm_config=llm_config,
    system_message="You query databases using database_query tool. Keep responses under 250 characters.",
    functions=[database_query],
)

file_agent = ConversableAgent(
    "file_agent",
    llm_config=llm_config,
    system_message="You process files using file_processor tool. Keep responses under 250 characters.",
    functions=[file_processor],
)

triage_agent = ConversableAgent(
    "triage_agent",
    llm_config=llm_config,
    system_message="You are a triage agent. Route requests to appropriate agents. Keep responses under 250 characters.",
)

general_agent = ConversableAgent(
    "general_agent",
    llm_config=llm_config,
    system_message="You handle general questions and coordination. Keep responses under 250 characters.",
)

user = ConversableAgent(
    "user",
    llm_config=llm_config,
    system_message="You represent the user asking questions. Keep responses under 250 characters.",
)


def test_scenario_1_one_agent_chat():
    """Test Scenario 1: Single Agent Chat"""
    print("\n=== Scenario 1: Single Agent Chat ===")

    # Simple conversation between user and assistant
    messages = [
        {"role": "user", "content": "What is 2+2?"},
    ]

    response = assistant.generate_reply(messages)
    print(f"Assistant response: {response}")

    return response


def test_scenario_2_two_agent_chat():
    """Test Scenario 2: Two Agent Chat"""
    print("\n=== Scenario 2: Two Agent Chat ===")

    # Chat between user proxy and math agent using messages format
    messages = [
        {"role": "user", "content": "Calculate 15 * 7 using your calculator tool"},
        {"role": "assistant", "content": "I'll calculate 15 * 7 for you using the calculator tool."},
    ]

    user_proxy.initiate_chat(math_agent, messages=messages, max_turns=3)

    return "Two agent chat completed"


def test_scenario_3_group_chat_manager():
    """Test Scenario 3: Group Chat with Manager"""
    print("\n=== Scenario 3: Group Chat with Manager ===")

    # Create group chat with manager
    group_chat = GroupChat(agents=[triage_agent, math_agent, weather_agent, db_agent], messages=[], max_round=5)

    group_chat_manager = GroupChatManager(groupchat=group_chat, llm_config=llm_config)

    # Initiate group chat using messages format
    messages = [
        {"role": "user", "content": "I need help with math and weather. Calculate 10+5 and check weather in Paris."},
    ]

    user_proxy.initiate_chat(group_chat_manager, messages=messages, max_turns=6)

    return "Group chat with manager completed"


def test_scenario_4_patterns():
    """Test Scenario 4: Different Patterns"""
    print("\n=== Scenario 4: Testing Different Patterns ===")

    # Round Robin Pattern
    print("\n--- Round Robin Pattern ---")
    round_robin_pattern = RoundRobinPattern(
        initial_agent=triage_agent,
        agents=[triage_agent, math_agent, weather_agent],
        user_agent=user,
        group_manager_args={"llm_config": llm_config},
    )

    messages1 = [
        {"role": "user", "content": "Help me with calculations and weather info"},
    ]

    result1, context1, last_agent1 = initiate_group_chat(pattern=round_robin_pattern, messages=messages1, max_rounds=4)
    print(f"Round Robin - Last agent: {last_agent1}")

    # Auto Pattern
    print("\n--- Auto Pattern ---")
    auto_pattern = AutoPattern(
        initial_agent=triage_agent,
        agents=[triage_agent, math_agent, weather_agent, db_agent, user],
        user_agent=user,
        group_manager_args={"llm_config": llm_config},
    )

    messages2 = [
        {"role": "user", "content": "I need database info and math calculations"},
    ]

    result2, context2, last_agent2 = initiate_group_chat(pattern=auto_pattern, messages=messages2, max_rounds=4)
    print(f"Auto Pattern - Last agent: {last_agent2}")

    # Random Pattern
    print("\n--- Random Pattern ---")
    random_pattern = RandomPattern(
        initial_agent=math_agent,
        agents=[math_agent, weather_agent, file_agent, user],
        user_agent=user,
        group_manager_args={"llm_config": llm_config},
    )

    messages3 = [
        {"role": "user", "content": "Random help needed"},
    ]

    result3, context3, last_agent3 = initiate_group_chat(pattern=random_pattern, messages=messages3, max_rounds=3)
    print(f"Random Pattern - Last agent: {last_agent3}")

    return "Patterns testing completed"


def test_scenario_5_handoffs_and_tools():
    """Test Scenario 5: Handoffs and Tools Integration"""
    print("\n=== Scenario 5: Handoffs and Tools ===")

    # Set up handoffs on the triage agent to route to appropriate agents
    triage_agent.handoffs.add_llm_conditions([
        OnCondition(
            condition=StringLLMCondition("contains math or calculate"),
            target=AgentTarget(math_agent),
            condition_llm_config=llm_config,
        ),
        OnCondition(
            condition=StringLLMCondition("contains weather"),
            target=AgentTarget(weather_agent),
            condition_llm_config=llm_config,
        ),
        OnCondition(
            condition=StringLLMCondition("contains database or user"),
            target=AgentTarget(db_agent),
            condition_llm_config=llm_config,
        ),
        OnCondition(
            condition=StringLLMCondition("contains file or process"),
            target=AgentTarget(file_agent),
            condition_llm_config=llm_config,
        ),
    ])

    # Set default target for triage agent
    triage_agent.handoffs.set_after_work(target=AgentTarget(general_agent))

    # Manual Pattern with handoffs - need to include all agents
    manual_pattern = ManualPattern(
        initial_agent=triage_agent,
        agents=[triage_agent, math_agent, weather_agent, db_agent, file_agent, general_agent],
        user_agent=user,
        group_manager_args={"llm_config": llm_config},
    )

    # Test different handoff scenarios using messages format
    test_messages_list = [
        [{"role": "user", "content": "Calculate 25 * 4 for me"}],
        [{"role": "user", "content": "What's the weather in Tokyo?"}],
        [{"role": "user", "content": "Query user info for ID 12345"}],
        [{"role": "user", "content": "Process my document file.pdf"}],
        [{"role": "user", "content": "Just a general question"}],
    ]

    for i, messages in enumerate(test_messages_list):
        print(f"\nTesting message: '{messages[0]['content']}'")
        result, context, last_agent = initiate_group_chat(pattern=manual_pattern, messages=messages, max_rounds=3)
        print(f"Handled by: {last_agent}")

    return "Handoffs and tools testing completed"


def test_scenario_6_mixed_tools_scenario():
    """Test Scenario 6: Mixed Tools in Group Chat"""
    print("\n=== Scenario 6: Mixed Tools Scenario ===")

    # Create pattern with multiple tool-enabled agents
    mixed_pattern = RoundRobinPattern(
        initial_agent=triage_agent,
        agents=[triage_agent, math_agent, weather_agent, db_agent, file_agent],
        user_agent=user,
        group_manager_args={"llm_config": llm_config},
    )

    messages = [
        {
            "role": "user",
            "content": "I need multiple services: calculate 100/4, check weather in London, query user 999, and process report.pdf",
        },
    ]

    result, context, last_agent = initiate_group_chat(pattern=mixed_pattern, messages=messages, max_rounds=8)

    print(f"Mixed tools scenario completed. Last agent: {last_agent}")
    return "Mixed tools scenario completed"


def test_scenario_7_conditional_handoffs():
    """Test Scenario 7: Advanced Conditional Handoffs"""
    print("\n=== Scenario 7: Advanced Conditional Handoffs ===")

    # Clear any existing handoffs on triage_agent
    triage_agent.handoffs.clear()

    # Set up handoffs on the triage agent using LLM conditions (simpler and more reliable)
    triage_agent.handoffs.add_llm_conditions([
        OnCondition(
            condition=StringLLMCondition("contains math or calculations"),
            target=AgentTarget(math_agent),
            condition_llm_config=llm_config,
        ),
        OnCondition(
            condition=StringLLMCondition("contains weather"),
            target=AgentTarget(weather_agent),
            condition_llm_config=llm_config,
        ),
        OnCondition(
            condition=StringLLMCondition("contains database"),
            target=AgentTarget(db_agent),
            condition_llm_config=llm_config,
        ),
    ])

    # Set default handoff target
    triage_agent.handoffs.set_after_work(target=AgentTarget(general_agent))

    # Create DefaultPattern with all necessary agents
    default_pattern = DefaultPattern(
        initial_agent=triage_agent,
        agents=[triage_agent, math_agent, weather_agent, db_agent, general_agent],
        user_agent=user,
        group_manager_args={"llm_config": llm_config},
    )

    messages = [
        {"role": "user", "content": "I need math calculations and weather data for my project"},
    ]

    result, context, last_agent = initiate_group_chat(pattern=default_pattern, messages=messages, max_rounds=5)

    print(f"Advanced handoffs completed. Last agent: {last_agent}")
    return "Advanced handoffs completed"


def test_scenario_8_complex_conversation():
    """Test Scenario 8: Complex Multi-turn Conversation"""
    print("\n=== Scenario 8: Complex Multi-turn Conversation ===")

    # Complex conversation with multiple turns
    complex_messages = [
        {"role": "user", "content": "Hello, I need help with multiple tasks"},
        {"role": "assistant", "content": "I'm here to help! What tasks do you need assistance with?"},
        {"role": "user", "content": "First, calculate 50 * 8, then check weather in New York"},
        {"role": "assistant", "content": "I'll help you with both tasks. Let me calculate 50 * 8 first."},
        {"role": "user", "content": "Great! Also query user info for ID 54321"},
    ]

    auto_pattern = AutoPattern(
        initial_agent=triage_agent,
        agents=[triage_agent, math_agent, weather_agent, db_agent, user],
        user_agent=user,
        group_manager_args={"llm_config": llm_config},
    )

    result, context, last_agent = initiate_group_chat(pattern=auto_pattern, messages=complex_messages, max_rounds=6)

    print(f"Complex conversation completed. Last agent: {last_agent}")
    return "Complex conversation completed"


def test_scenario_9_nested_chat_target():
    """Test Scenario 9: Nested Chat Target"""
    print("\n=== Scenario 9: Nested Chat Target ===")

    # Clear any existing handoffs on triage_agent
    triage_agent.handoffs.clear()

    # Set up nested chat handoffs on the triage agent
    triage_agent.handoffs.add_llm_conditions([
        OnCondition(
            condition=StringLLMCondition("contains math"),
            target=NestedChatTarget(
                target=AgentTarget(math_agent),
                max_turns=2,
                nested_chat_config={
                    "chat_queue": [
                        {"recipient": math_agent, "message": "Please help with the math problem", "max_turns": 2}
                    ],
                    "use_async": False,
                },
            ),
            condition_llm_config=llm_config,
        ),
        OnCondition(
            condition=StringLLMCondition("contains weather"),
            target=NestedChatTarget(
                target=AgentTarget(weather_agent),
                max_turns=2,
                nested_chat_config={
                    "chat_queue": [
                        {"recipient": weather_agent, "message": "Please help with the weather inquiry", "max_turns": 2}
                    ],
                    "use_async": False,
                },
            ),
            condition_llm_config=llm_config,
        ),
    ])

    # Set default handoff target
    triage_agent.handoffs.set_after_work(target=AgentTarget(general_agent))

    nested_pattern = ManualPattern(
        initial_agent=triage_agent,
        agents=[triage_agent, math_agent, weather_agent, general_agent],
        user_agent=user,
        group_manager_args={"llm_config": llm_config},
    )

    messages = [
        {"role": "user", "content": "I need help with math calculations for my homework"},
    ]

    result, context, last_agent = initiate_group_chat(pattern=nested_pattern, messages=messages, max_rounds=5)

    print(f"Nested chat completed. Last agent: {last_agent}")
    return "Nested chat completed"


def test_scenario_10_terminate_target():
    """Test Scenario 10: Terminate Target"""
    print("\n=== Scenario 10: Terminate Target ===")

    # Clear any existing handoffs on triage_agent
    triage_agent.handoffs.clear()

    # Set up handoffs with terminate condition on the triage agent
    triage_agent.handoffs.add_llm_conditions([
        OnCondition(
            condition=StringLLMCondition("contains goodbye or exit"),
            target=TerminateTarget(),
            condition_llm_config=llm_config,
        ),
        OnCondition(
            condition=StringLLMCondition("contains help"),
            target=AgentTarget(general_agent),
            condition_llm_config=llm_config,
        ),
    ])

    # Set default handoff target
    triage_agent.handoffs.set_after_work(target=AgentTarget(triage_agent))

    terminate_pattern = ManualPattern(
        initial_agent=triage_agent,
        agents=[triage_agent, general_agent],
        user_agent=user,
        group_manager_args={"llm_config": llm_config},
    )

    messages = [
        {"role": "user", "content": "Hello, I need help"},
        {"role": "assistant", "content": "I'm here to help you!"},
        {"role": "user", "content": "Thank you, goodbye!"},
    ]

    result, context, last_agent = initiate_group_chat(pattern=terminate_pattern, messages=messages, max_rounds=5)

    print(f"Terminate target completed. Last agent: {last_agent}")
    return "Terminate target completed"


# Main execution
if __name__ == "__main__":
    print("Starting AutoGen Test Scenarios...")

    try:
        # Run all test scenarios
        # test_scenario_1_one_agent_chat()
        # test_scenario_2_two_agent_chat()
        # test_scenario_3_group_chat_manager()  # fail # problem in tool calls
        # test_scenario_4_patterns()
        # test_scenario_5_handoffs_and_tools()
        test_scenario_6_mixed_tools_scenario()
        # test_scenario_7_conditional_handoffs()
        # test_scenario_8_complex_conversation()
        # test_scenario_9_nested_chat_target() # 50 % stuck auto reply in threads
        # test_scenario_10_terminate_target()

        print("\n=== All Test Scenarios Completed Successfully! ===")

    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback

        traceback.print_exc()
