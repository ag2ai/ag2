# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import copy
from unittest.mock import MagicMock, patch

import pytest

from autogen.agentchat import ConversableAgent
from autogen.agentchat.chat import ChatResult
from autogen.agentchat.conversable_agent import register_function
from autogen.agentchat.group.context_condition import StringContextCondition
from autogen.agentchat.group.guardrails import GuardrailResult, RegexGuardrail
from autogen.agentchat.group.llm_condition import StringLLMCondition
from autogen.agentchat.group.on_condition import OnCondition
from autogen.agentchat.group.on_context_condition import OnContextCondition
from autogen.agentchat.group.targets.transition_target import AgentNameTarget
from autogen.events.agent_events import (
    ClearConversableAgentHistoryEvent,
    ClearConversableAgentHistoryWarningEvent,
    ConversableAgentUsageSummaryEvent,
    ConversableAgentUsageSummaryNoCostIncurredEvent,
)

# ============================================================================
# __init__ and _register_default_reply_handlers Tests
# ============================================================================


def test__init__registers_default_reply_handlers():
    """Test that __init__ registers all default reply handlers."""
    agent = ConversableAgent(name="test_agent", llm_config=False, code_execution_config=False)

    # Check that reply handlers are registered
    assert len(agent._reply_func_list) > 0

    # Check for OAI reply handlers
    oai_handlers = [
        func_tuple
        for func_tuple in agent._reply_func_list
        if "generate_oai_reply" in str(func_tuple.get("reply_func", ""))
    ]
    assert len(oai_handlers) >= 1


def test__register_default_reply_handlers_code_execution_with_executor():
    """Test code execution handler registration when executor is configured."""
    agent = ConversableAgent(
        name="test_agent",
        llm_config=False,
        code_execution_config={"executor": "commandline-local", "use_docker": False},
    )

    # Check that executor-based handler is registered
    executor_handlers = [
        func_tuple
        for func_tuple in agent._reply_func_list
        if "_generate_code_execution_reply_using_executor" in str(func_tuple.get("reply_func", ""))
    ]
    assert len(executor_handlers) >= 1


def test__register_default_reply_handlers_code_execution_legacy():
    """Test code execution handler registration with legacy config."""
    agent = ConversableAgent(
        name="test_agent", llm_config=False, code_execution_config={"use_docker": False, "last_n_messages": 1}
    )

    # Check that legacy handler is registered
    legacy_handlers = [
        func_tuple
        for func_tuple in agent._reply_func_list
        if "generate_code_execution_reply" in str(func_tuple.get("reply_func", ""))
        and "_generate_code_execution_reply_using_executor" not in str(func_tuple.get("reply_func", ""))
    ]
    assert len(legacy_handlers) >= 1


# ============================================================================
# get_chat_results Tests
# ============================================================================


def test_get_chat_results_index_out_of_range():
    """Test get_chat_results raises IndexError for out of range index."""
    agent = ConversableAgent(name="test_agent", llm_config=False)
    agent._finished_chats = [ChatResult(chat_history=[], summary="Test", cost={}, human_input=[])]

    with pytest.raises(IndexError):
        agent.get_chat_results(10)


# ============================================================================
# reset Tests
# ============================================================================


def test_reset_clears_history():
    """Test that reset calls clear_history."""
    agent = ConversableAgent(name="test_agent", llm_config=False)
    recipient = ConversableAgent(name="recipient", llm_config=False)

    agent.chat_messages[recipient] = [{"content": "test", "role": "user"}]
    agent.reset()

    assert len(agent.chat_messages) == 0


def test_reset_resets_consecutive_counter():
    """Test that reset calls reset_consecutive_auto_reply_counter."""
    agent = ConversableAgent(name="test_agent", llm_config=False)
    recipient = ConversableAgent(name="recipient", llm_config=False)

    agent._consecutive_auto_reply_counter[recipient] = 5
    agent.reset()

    assert len(agent._consecutive_auto_reply_counter) == 0


def test_reset_stops_reply_at_receive():
    """Test that reset calls stop_reply_at_receive."""
    agent = ConversableAgent(name="test_agent", llm_config=False)
    recipient = ConversableAgent(name="recipient", llm_config=False)

    agent.reply_at_receive[recipient] = True
    agent.reset()

    assert len(agent.reply_at_receive) == 0


def test_reset_clears_usage_summary():
    """Test that reset clears usage summary when client exists."""
    agent = ConversableAgent(name="test_agent", llm_config=False)
    mock_client = MagicMock()
    agent.client = mock_client

    agent.reset()

    mock_client.clear_usage_summary.assert_called_once()


def test_reset_resets_reply_configs():
    """Test that reset resets reply function configs."""
    agent = ConversableAgent(name="test_agent", llm_config=False)

    # Create a mock reply function with reset_config
    mock_reset_config = MagicMock()
    mock_config = {"key": "value"}
    mock_init_config = {"key": "initial"}

    agent._reply_func_list = [
        {
            "reply_func": lambda x: x,
            "config": copy.copy(mock_config),
            "init_config": mock_init_config,
            "reset_config": mock_reset_config,
        }
    ]

    agent.reset()

    mock_reset_config.assert_called_once_with(mock_config)


def test_reset_resets_reply_configs_without_reset_config():
    """Test that reset copies init_config when reset_config is None."""
    agent = ConversableAgent(name="test_agent", llm_config=False)

    mock_config = {"key": "modified"}
    mock_init_config = {"key": "initial"}

    agent._reply_func_list = [
        {
            "reply_func": lambda x: x,
            "config": copy.copy(mock_config),
            "init_config": mock_init_config,
            "reset_config": None,
        }
    ]

    agent.reset()

    assert agent._reply_func_list[0]["config"] == mock_init_config


# ============================================================================
# stop_reply_at_receive Tests
# ============================================================================


def test_stop_reply_at_receive_with_sender():
    """Test stop_reply_at_receive with specific sender."""
    agent = ConversableAgent(name="test_agent", llm_config=False)
    sender = ConversableAgent(name="sender", llm_config=False)

    agent.reply_at_receive[sender] = True
    agent.stop_reply_at_receive(sender)

    assert agent.reply_at_receive[sender] is False


def test_stop_reply_at_receive_without_sender():
    """Test stop_reply_at_receive clears all when sender is None."""
    agent = ConversableAgent(name="test_agent", llm_config=False)
    sender1 = ConversableAgent(name="sender1", llm_config=False)
    sender2 = ConversableAgent(name="sender2", llm_config=False)

    agent.reply_at_receive[sender1] = True
    agent.reply_at_receive[sender2] = True

    agent.stop_reply_at_receive()

    assert len(agent.reply_at_receive) == 0


def test_stop_reply_at_receive_multiple_senders():
    """Test stop_reply_at_receive handles multiple senders correctly."""
    agent = ConversableAgent(name="test_agent", llm_config=False)
    sender1 = ConversableAgent(name="sender1", llm_config=False)
    sender2 = ConversableAgent(name="sender2", llm_config=False)

    agent.reply_at_receive[sender1] = True
    agent.reply_at_receive[sender2] = True

    agent.stop_reply_at_receive(sender1)

    assert agent.reply_at_receive[sender1] is False
    assert agent.reply_at_receive[sender2] is True


# ============================================================================
# reset_consecutive_auto_reply_counter Tests
# ============================================================================


def test_reset_consecutive_auto_reply_counter_with_sender():
    """Test reset_consecutive_auto_reply_counter with specific sender."""
    agent = ConversableAgent(name="test_agent", llm_config=False)
    sender = ConversableAgent(name="sender", llm_config=False)

    agent._consecutive_auto_reply_counter[sender] = 5
    agent.reset_consecutive_auto_reply_counter(sender)

    assert agent._consecutive_auto_reply_counter[sender] == 0


def test_reset_consecutive_auto_reply_counter_without_sender():
    """Test reset_consecutive_auto_reply_counter clears all when sender is None."""
    agent = ConversableAgent(name="test_agent", llm_config=False)
    sender1 = ConversableAgent(name="sender1", llm_config=False)
    sender2 = ConversableAgent(name="sender2", llm_config=False)

    agent._consecutive_auto_reply_counter[sender1] = 3
    agent._consecutive_auto_reply_counter[sender2] = 5

    agent.reset_consecutive_auto_reply_counter()

    assert len(agent._consecutive_auto_reply_counter) == 0


def test_reset_consecutive_auto_reply_counter_multiple_senders():
    """Test reset_consecutive_auto_reply_counter handles multiple senders correctly."""
    agent = ConversableAgent(name="test_agent", llm_config=False)
    sender1 = ConversableAgent(name="sender1", llm_config=False)
    sender2 = ConversableAgent(name="sender2", llm_config=False)

    agent._consecutive_auto_reply_counter[sender1] = 3
    agent._consecutive_auto_reply_counter[sender2] = 5

    agent.reset_consecutive_auto_reply_counter(sender1)

    assert agent._consecutive_auto_reply_counter[sender1] == 0
    assert agent._consecutive_auto_reply_counter[sender2] == 5


# ============================================================================
# clear_history Tests
# ============================================================================


def test_clear_history_all_recipients():
    """Test clear_history clears all history when recipient is None."""
    agent = ConversableAgent(name="test_agent", llm_config=False)
    recipient1 = ConversableAgent(name="recipient1", llm_config=False)
    recipient2 = ConversableAgent(name="recipient2", llm_config=False)

    agent.chat_messages[recipient1] = [{"content": "msg1", "role": "user"}]
    agent.chat_messages[recipient2] = [{"content": "msg2", "role": "user"}]

    agent.clear_history()

    assert len(agent.chat_messages) == 0


def test_clear_history_specific_recipient():
    """Test clear_history clears history for specific recipient."""
    agent = ConversableAgent(name="test_agent", llm_config=False)
    recipient1 = ConversableAgent(name="recipient1", llm_config=False)
    recipient2 = ConversableAgent(name="recipient2", llm_config=False)

    agent.chat_messages[recipient1] = [{"content": "msg1", "role": "user"}]
    agent.chat_messages[recipient2] = [{"content": "msg2", "role": "user"}]

    agent.clear_history(recipient=recipient1)

    assert len(agent.chat_messages[recipient1]) == 0
    assert len(agent.chat_messages[recipient2]) == 1


def test_clear_history_preserves_messages():
    """Test clear_history preserves last N messages correctly."""
    agent = ConversableAgent(name="test_agent", llm_config=False)
    recipient = ConversableAgent(name="recipient", llm_config=False)

    agent.chat_messages[recipient] = [
        {"content": "msg1", "role": "user"},
        {"content": "msg2", "role": "user"},
        {"content": "msg3", "role": "user"},
        {"content": "msg4", "role": "user"},
    ]

    agent.clear_history(recipient=None, nr_messages_to_preserve=2)

    assert len(agent.chat_messages[recipient]) == 2
    assert agent.chat_messages[recipient][0]["content"] == "msg3"
    assert agent.chat_messages[recipient][1]["content"] == "msg4"


def test_clear_history_preserves_tool_responses():
    """Test clear_history handles tool_responses preservation correctly."""
    agent = ConversableAgent(name="test_agent", llm_config=False)
    recipient = ConversableAgent(name="recipient", llm_config=False)

    agent.chat_messages[recipient] = [
        {"content": "msg1", "role": "user"},
        {"content": "msg2", "role": "user"},
        {"content": "msg3", "role": "user"},
        {"tool_responses": [{"tool_call_id": "call1"}], "role": "assistant"},
    ]

    mock_iostream = MagicMock()
    with patch("autogen.agentchat.conversableAgent.conversable_agent.IOStream.get_default", return_value=mock_iostream):
        agent.clear_history(recipient=None, nr_messages_to_preserve=1)

    # Should preserve 2 messages (1 requested + 1 for tool_responses)
    assert len(agent.chat_messages[recipient]) == 2
    mock_iostream.send.assert_called_once()
    assert isinstance(mock_iostream.send.call_args[0][0], ClearConversableAgentHistoryEvent)


def test_clear_history_emits_event_all():
    """Test clear_history emits ClearConversableAgentHistoryEvent when clearing all."""
    agent = ConversableAgent(name="test_agent", llm_config=False)
    recipient = ConversableAgent(name="recipient", llm_config=False)

    agent.chat_messages[recipient] = [{"content": "msg", "role": "user"}]

    mock_iostream = MagicMock()
    with patch("autogen.agentchat.conversableAgent.conversable_agent.IOStream.get_default", return_value=mock_iostream):
        agent.clear_history()

    mock_iostream.send.assert_called()
    # Check that event was sent (may be called multiple times)
    calls = [call[0][0] for call in mock_iostream.send.call_args_list]
    assert any(isinstance(event, ClearConversableAgentHistoryEvent) for event in calls)


def test_clear_history_emits_warning_specific():
    """Test clear_history emits warning when preserving with specific recipient."""
    agent = ConversableAgent(name="test_agent", llm_config=False)
    recipient = ConversableAgent(name="recipient", llm_config=False)

    agent.chat_messages[recipient] = [{"content": "msg", "role": "user"}]

    mock_iostream = MagicMock()
    with patch("autogen.agentchat.conversableAgent.conversable_agent.IOStream.get_default", return_value=mock_iostream):
        agent.clear_history(recipient=recipient, nr_messages_to_preserve=1)

    mock_iostream.send.assert_called_once()
    assert isinstance(mock_iostream.send.call_args[0][0], ClearConversableAgentHistoryWarningEvent)


def test_clear_history_no_preserve_all():
    """Test clear_history clears all messages when nr_messages_to_preserve is None."""
    agent = ConversableAgent(name="test_agent", llm_config=False)
    recipient = ConversableAgent(name="recipient", llm_config=False)

    agent.chat_messages[recipient] = [
        {"content": "msg1", "role": "user"},
        {"content": "msg2", "role": "user"},
    ]

    agent.clear_history(recipient=None, nr_messages_to_preserve=None)

    assert len(agent.chat_messages) == 0


def test_clear_history_edge_cases():
    """Test clear_history edge cases."""
    agent = ConversableAgent(name="test_agent", llm_config=False)
    recipient = ConversableAgent(name="recipient", llm_config=False)

    # Test with empty history
    agent.chat_messages[recipient] = []
    agent.clear_history(recipient=recipient)
    assert len(agent.chat_messages[recipient]) == 0

    # Test with preserve count > history length
    agent.chat_messages[recipient] = [{"content": "msg", "role": "user"}]
    agent.clear_history(recipient=None, nr_messages_to_preserve=10)
    assert len(agent.chat_messages[recipient]) == 1


# ============================================================================
# print_usage_summary Tests
# ============================================================================


def test_print_usage_summary_with_client():
    """Test print_usage_summary with client emits event and calls client method."""
    agent = ConversableAgent(name="test_agent", llm_config=False)
    mock_client = MagicMock()
    agent.client = mock_client

    mock_iostream = MagicMock()
    with patch("autogen.agentchat.conversableAgent.conversable_agent.IOStream.get_default", return_value=mock_iostream):
        agent.print_usage_summary()

    mock_iostream.send.assert_called_once()
    assert isinstance(mock_iostream.send.call_args[0][0], ConversableAgentUsageSummaryEvent)
    mock_client.print_usage_summary.assert_called_once_with(["actual", "total"])


def test_print_usage_summary_without_client():
    """Test print_usage_summary without client emits no cost event."""
    agent = ConversableAgent(name="test_agent", llm_config=False)
    agent.client = None

    mock_iostream = MagicMock()
    with patch("autogen.agentchat.conversableAgent.conversable_agent.IOStream.get_default", return_value=mock_iostream):
        agent.print_usage_summary()

    mock_iostream.send.assert_called_once()
    assert isinstance(mock_iostream.send.call_args[0][0], ConversableAgentUsageSummaryNoCostIncurredEvent)


def test_print_usage_summary_mode_parameter():
    """Test print_usage_summary passes mode parameter correctly."""
    agent = ConversableAgent(name="test_agent", llm_config=False)
    mock_client = MagicMock()
    agent.client = mock_client

    agent.print_usage_summary(mode="actual")

    mock_client.print_usage_summary.assert_called_once_with("actual")


# ============================================================================
# get_actual_usage Tests
# ============================================================================


def test_get_actual_usage_with_client():
    """Test get_actual_usage returns client.actual_usage_summary."""
    agent = ConversableAgent(name="test_agent", llm_config=False)
    mock_client = MagicMock()
    mock_client.actual_usage_summary = {"tokens": 100}
    agent.client = mock_client

    result = agent.get_actual_usage()

    assert result == {"tokens": 100}


def test_get_actual_usage_without_client():
    """Test get_actual_usage returns None when client is None."""
    agent = ConversableAgent(name="test_agent", llm_config=False)
    agent.client = None

    result = agent.get_actual_usage()

    assert result is None


# ============================================================================
# get_total_usage Tests
# ============================================================================


def test_get_total_usage_with_client():
    """Test get_total_usage returns client.total_usage_summary."""
    agent = ConversableAgent(name="test_agent", llm_config=False)
    mock_client = MagicMock()
    mock_client.total_usage_summary = {"tokens": 200}
    agent.client = mock_client

    result = agent.get_total_usage()

    assert result == {"tokens": 200}


def test_get_total_usage_without_client():
    """Test get_total_usage returns None when client is None."""
    agent = ConversableAgent(name="test_agent", llm_config=False)
    agent.client = None

    result = agent.get_total_usage()

    assert result is None


# ============================================================================
# register_handoff Tests
# ============================================================================


def test_register_handoff_single_condition():
    """Test register_handoff registers single OnCondition."""
    agent = ConversableAgent(name="test_agent", llm_config=False)
    target = AgentNameTarget(agent_name="target_agent")
    condition = OnCondition(target=target, condition=StringLLMCondition(prompt="test"))

    agent.register_handoff(condition)

    assert condition in agent.handoffs.llm_conditions


def test_register_handoff_on_context_condition():
    """Test register_handoff registers OnContextCondition."""
    agent = ConversableAgent(name="test_agent", llm_config=False)
    target = AgentNameTarget(agent_name="target_agent")
    condition = OnContextCondition(target=target, condition=StringContextCondition(variable_name="test_var"))

    agent.register_handoff(condition)

    assert condition in agent.handoffs.context_conditions


def test_register_handoff_calls_handoffs_add():
    """Test register_handoff calls handoffs.add()."""
    agent = ConversableAgent(name="test_agent", llm_config=False)
    target = AgentNameTarget(agent_name="target_agent")
    condition = OnCondition(target=target, condition=StringLLMCondition(prompt="test"))

    with patch.object(agent.handoffs, "add") as mock_add:
        agent.register_handoff(condition)

    mock_add.assert_called_once_with(condition)


# ============================================================================
# register_handoffs Tests
# ============================================================================


def test_register_handoffs_multiple_conditions():
    """Test register_handoffs registers multiple conditions."""
    agent = ConversableAgent(name="test_agent", llm_config=False)
    target = AgentNameTarget(agent_name="target_agent")
    condition1 = OnCondition(target=target, condition=StringLLMCondition(prompt="test1"))
    condition2 = OnCondition(target=target, condition=StringLLMCondition(prompt="test2"))
    conditions = [condition1, condition2]

    agent.register_handoffs(conditions)

    assert condition1 in agent.handoffs.llm_conditions
    assert condition2 in agent.handoffs.llm_conditions


def test_register_handoffs_calls_handoffs_add_many():
    """Test register_handoffs calls handoffs.add_many()."""
    agent = ConversableAgent(name="test_agent", llm_config=False)
    target = AgentNameTarget(agent_name="target_agent")
    condition = OnCondition(target=target, condition=StringLLMCondition(prompt="test"))
    conditions = [condition]

    with patch.object(agent.handoffs, "add_many") as mock_add_many:
        agent.register_handoffs(conditions)

    mock_add_many.assert_called_once_with(conditions)


# ============================================================================
# register_input_guardrail Tests
# ============================================================================


def test_register_input_guardrail_single():
    """Test register_input_guardrail appends single guardrail."""
    agent = ConversableAgent(name="test_agent", llm_config=False)
    target = AgentNameTarget(agent_name="target")
    guardrail = RegexGuardrail(name="test", condition=r"test", target=target)

    agent.register_input_guardrail(guardrail)

    assert guardrail in agent.input_guardrails
    assert len(agent.input_guardrails) == 1


def test_register_input_guardrail_multiple_calls():
    """Test register_input_guardrail handles multiple registrations."""
    agent = ConversableAgent(name="test_agent", llm_config=False)
    target = AgentNameTarget(agent_name="target")
    guardrail1 = RegexGuardrail(name="test1", condition=r"test1", target=target)
    guardrail2 = RegexGuardrail(name="test2", condition=r"test2", target=target)

    agent.register_input_guardrail(guardrail1)
    agent.register_input_guardrail(guardrail2)

    assert len(agent.input_guardrails) == 2
    assert guardrail1 in agent.input_guardrails
    assert guardrail2 in agent.input_guardrails


# ============================================================================
# register_input_guardrails Tests
# ============================================================================


def test_register_input_guardrails_multiple():
    """Test register_input_guardrails extends with list."""
    agent = ConversableAgent(name="test_agent", llm_config=False)
    target = AgentNameTarget(agent_name="target")
    guardrail1 = RegexGuardrail(name="test1", condition=r"test1", target=target)
    guardrail2 = RegexGuardrail(name="test2", condition=r"test2", target=target)
    guardrails = [guardrail1, guardrail2]

    agent.register_input_guardrails(guardrails)

    assert len(agent.input_guardrails) == 2
    assert guardrail1 in agent.input_guardrails
    assert guardrail2 in agent.input_guardrails


def test_register_input_guardrails_empty_list():
    """Test register_input_guardrails handles empty list."""
    agent = ConversableAgent(name="test_agent", llm_config=False)
    initial_count = len(agent.input_guardrails)

    agent.register_input_guardrails([])

    assert len(agent.input_guardrails) == initial_count


# ============================================================================
# register_output_guardrail Tests
# ============================================================================


def test_register_output_guardrail_single():
    """Test register_output_guardrail appends single guardrail."""
    agent = ConversableAgent(name="test_agent", llm_config=False)
    target = AgentNameTarget(agent_name="target")
    guardrail = RegexGuardrail(name="test", condition=r"test", target=target)

    agent.register_output_guardrail(guardrail)

    assert guardrail in agent.output_guardrails
    assert len(agent.output_guardrails) == 1


def test_register_output_guardrail_multiple_calls():
    """Test register_output_guardrail handles multiple registrations."""
    agent = ConversableAgent(name="test_agent", llm_config=False)
    target = AgentNameTarget(agent_name="target")
    guardrail1 = RegexGuardrail(name="test1", condition=r"test1", target=target)
    guardrail2 = RegexGuardrail(name="test2", condition=r"test2", target=target)

    agent.register_output_guardrail(guardrail1)
    agent.register_output_guardrail(guardrail2)

    assert len(agent.output_guardrails) == 2
    assert guardrail1 in agent.output_guardrails
    assert guardrail2 in agent.output_guardrails


# ============================================================================
# register_output_guardrails Tests
# ============================================================================


def test_register_output_guardrails_multiple():
    """Test register_output_guardrails extends with list."""
    agent = ConversableAgent(name="test_agent", llm_config=False)
    target = AgentNameTarget(agent_name="target")
    guardrail1 = RegexGuardrail(name="test1", condition=r"test1", target=target)
    guardrail2 = RegexGuardrail(name="test2", condition=r"test2", target=target)
    guardrails = [guardrail1, guardrail2]

    agent.register_output_guardrails(guardrails)

    assert len(agent.output_guardrails) == 2
    assert guardrail1 in agent.output_guardrails
    assert guardrail2 in agent.output_guardrails


def test_register_output_guardrails_empty_list():
    """Test register_output_guardrails handles empty list."""
    agent = ConversableAgent(name="test_agent", llm_config=False)
    initial_count = len(agent.output_guardrails)

    agent.register_output_guardrails([])

    assert len(agent.output_guardrails) == initial_count


# ============================================================================
# run_input_guardrails Tests
# ============================================================================


def test_run_input_guardrails_no_activation():
    """Test run_input_guardrails returns None when no guardrail activates."""
    agent = ConversableAgent(name="test_agent", llm_config=False)
    target = AgentNameTarget(agent_name="target")
    guardrail = RegexGuardrail(name="test", condition=r"no_match", target=target)
    agent.register_input_guardrail(guardrail)

    result = agent.run_input_guardrails(messages=[{"content": "test message", "role": "user"}])

    assert result is None


def test_run_input_guardrails_first_activation():
    """Test run_input_guardrails returns first activated guardrail result."""
    agent = ConversableAgent(name="test_agent", llm_config=False)
    target = AgentNameTarget(agent_name="target")
    guardrail1 = RegexGuardrail(name="test1", condition=r"test", target=target)
    guardrail2 = RegexGuardrail(name="test2", condition=r"test", target=target)
    agent.register_input_guardrail(guardrail1)
    agent.register_input_guardrail(guardrail2)

    result = agent.run_input_guardrails(messages=[{"content": "test message", "role": "user"}])

    assert result is not None
    assert result.activated is True
    assert result.guardrail == guardrail1


def test_run_input_guardrails_stops_on_activation():
    """Test run_input_guardrails stops checking after first activation."""
    agent = ConversableAgent(name="test_agent", llm_config=False)
    target = AgentNameTarget(agent_name="target")
    guardrail1 = RegexGuardrail(name="test1", condition=r"test", target=target)
    guardrail2 = RegexGuardrail(name="test2", condition=r"test", target=target)
    agent.register_input_guardrail(guardrail1)
    agent.register_input_guardrail(guardrail2)

    with patch.object(guardrail2, "check") as mock_check:
        agent.run_input_guardrails(messages=[{"content": "test message", "role": "user"}])

    # Second guardrail should not be checked
    mock_check.assert_not_called()


def test_run_input_guardrails_with_messages():
    """Test run_input_guardrails passes messages to guardrail.check()."""
    agent = ConversableAgent(name="test_agent", llm_config=False)
    target = AgentNameTarget(agent_name="target")
    guardrail = RegexGuardrail(name="test", condition=r"no_match", target=target)
    agent.register_input_guardrail(guardrail)

    messages = [{"content": "test", "role": "user"}]
    with patch.object(guardrail, "check") as mock_check:
        mock_check.return_value = GuardrailResult(activated=False, guardrail=guardrail)
        agent.run_input_guardrails(messages=messages)

    mock_check.assert_called_once_with(context=messages)


def test_run_input_guardrails_empty_list():
    """Test run_input_guardrails handles empty guardrails list."""
    agent = ConversableAgent(name="test_agent", llm_config=False)

    result = agent.run_input_guardrails(messages=[{"content": "test", "role": "user"}])

    assert result is None


# ============================================================================
# run_output_guardrails Tests
# ============================================================================


def test_run_output_guardrails_no_activation():
    """Test run_output_guardrails returns None when no guardrail activates."""
    agent = ConversableAgent(name="test_agent", llm_config=False)
    target = AgentNameTarget(agent_name="target")
    guardrail = RegexGuardrail(name="test", condition=r"no_match", target=target)
    agent.register_output_guardrail(guardrail)

    result = agent.run_output_guardrails(reply="test message")

    assert result is None


def test_run_output_guardrails_first_activation():
    """Test run_output_guardrails returns first activated guardrail result."""
    agent = ConversableAgent(name="test_agent", llm_config=False)
    target = AgentNameTarget(agent_name="target")
    guardrail1 = RegexGuardrail(name="test1", condition=r"test", target=target)
    guardrail2 = RegexGuardrail(name="test2", condition=r"test", target=target)
    agent.register_output_guardrail(guardrail1)
    agent.register_output_guardrail(guardrail2)

    result = agent.run_output_guardrails(reply="test message")

    assert result is not None
    assert result.activated is True
    assert result.guardrail == guardrail1


def test_run_output_guardrails_stops_on_activation():
    """Test run_output_guardrails stops checking after first activation."""
    agent = ConversableAgent(name="test_agent", llm_config=False)
    target = AgentNameTarget(agent_name="target")
    guardrail1 = RegexGuardrail(name="test1", condition=r"test", target=target)
    guardrail2 = RegexGuardrail(name="test2", condition=r"test", target=target)
    agent.register_output_guardrail(guardrail1)
    agent.register_output_guardrail(guardrail2)

    with patch.object(guardrail2, "check") as mock_check:
        agent.run_output_guardrails(reply="test message")

    # Second guardrail should not be checked
    mock_check.assert_not_called()


def test_run_output_guardrails_with_string_reply():
    """Test run_output_guardrails handles string reply."""
    agent = ConversableAgent(name="test_agent", llm_config=False)
    target = AgentNameTarget(agent_name="target")
    guardrail = RegexGuardrail(name="test", condition=r"no_match", target=target)
    agent.register_output_guardrail(guardrail)

    reply = "test message"
    with patch.object(guardrail, "check") as mock_check:
        mock_check.return_value = GuardrailResult(activated=False, guardrail=guardrail)
        agent.run_output_guardrails(reply=reply)

    mock_check.assert_called_once_with(context=reply)


def test_run_output_guardrails_with_dict_reply():
    """Test run_output_guardrails handles dict reply."""
    agent = ConversableAgent(name="test_agent", llm_config=False)
    target = AgentNameTarget(agent_name="target")
    guardrail = RegexGuardrail(name="test", condition=r"no_match", target=target)
    agent.register_output_guardrail(guardrail)

    reply = {"content": "test message", "role": "assistant"}
    with patch.object(guardrail, "check") as mock_check:
        mock_check.return_value = GuardrailResult(activated=False, guardrail=guardrail)
        agent.run_output_guardrails(reply=reply)

    mock_check.assert_called_once_with(context=reply)


# ============================================================================
# register_function (standalone function) Tests
# ============================================================================


def test_register_function_with_name():
    """Test register_function with custom name parameter."""
    caller = ConversableAgent(name="caller", llm_config=False)
    executor = ConversableAgent(name="executor", llm_config=False)

    def test_func(x: str) -> str:
        return x

    register_function(test_func, caller=caller, executor=executor, name="custom_name", description="Test function")

    assert "custom_name" in executor.function_map


def test_register_function_calls_both_decorators():
    """Test register_function calls both register_for_llm and register_for_execution."""
    caller = ConversableAgent(name="caller", llm_config=False)
    executor = ConversableAgent(name="executor", llm_config=False)

    def test_func(x: str) -> str:
        return x

    with (
        patch.object(caller, "register_for_llm") as mock_llm,
        patch.object(executor, "register_for_execution") as mock_exec,
    ):
        register_function(test_func, caller=caller, executor=executor, description="Test")

    mock_llm.assert_called_once()
    mock_exec.assert_called_once()


def test_register_function_with_none_name():
    """Test register_function uses function name when name is None."""
    caller = ConversableAgent(name="caller", llm_config=False)
    executor = ConversableAgent(name="executor", llm_config=False)

    def test_func(x: str) -> str:
        return x

    register_function(test_func, caller=caller, executor=executor, name=None, description="Test function")

    assert "test_func" in executor.function_map
