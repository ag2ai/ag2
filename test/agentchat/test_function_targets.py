# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any
from unittest import mock

import pytest
from pytest import MonkeyPatch

from autogen import ConversableAgent
from autogen.agentchat.group import (
    AgentNameTarget,
    AgentTarget,
    FunctionTarget,
    FunctionTargetResult,
    RevertToUserTarget,
    StayTarget,
)
from autogen.agentchat.group.context_variables import ContextVariables
from autogen.agentchat.group.speaker_selection_result import SpeakerSelectionResult
from autogen.agentchat.group.targets.function_target import (
    FunctionTargetMessage,
    broadcast,
    construct_broadcast_messages_list,
    validate_fn_sig,
)
from autogen.agentchat.groupchat import GroupChat

###############################################################################
# Helpers and test doubles
###############################################################################


class DummyGroupManager:
    """Simple stand-in for a GroupChatManager that just records sends."""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    def send(self, msg: dict[str, Any], recipient: ConversableAgent, request_reply: bool = False, silent: bool = False):
        self.sent.append({
            "msg": msg,
            "recipient": recipient,
            "request_reply": request_reply,
            "silent": silent,
        })


class DummyContextVariables:
    """Minimal context variable wrapper that matches the real ctx variable interface."""

    def __init__(self, data: dict[str, Any] | None = None) -> None:
        self._data: dict[str, Any] = dict(data or {})

    def update(self, new_data: dict[str, Any]) -> None:
        self._data.update(new_data)

    def to_dict(self) -> dict[str, Any]:
        return dict(self._data)


# Shared test agent
test_agent = ConversableAgent(name="test_agent", llm_config=None)


# Library of test function_targets
def minimal_correct_fn_target(output: str, context_variables: Any) -> FunctionTargetResult:
    """Minimal valid function_target: 2 params, returns FunctionTargetResult with a simple AgentTarget."""
    return FunctionTargetResult(
        messages=output,
        target=AgentTarget(test_agent),
    )


def correct_fn_target_extra_args(output: str, context_variables: Any, extra_param: int) -> FunctionTargetResult:
    """Valid function_target with an extra required parameter satisfied via extra_args."""
    return FunctionTargetResult(
        messages=f"{output} (extra={extra_param})",
        target=AgentTarget(test_agent),
    )


def fn_with_optional_param(output: str, context_variables: Any, opt: int = 1) -> FunctionTargetResult:
    """Valid function_target with an optional param that should not require extra_args."""
    return FunctionTargetResult(messages=f"{output} (opt={opt})", target=AgentTarget(test_agent))


def fn_with_kwargs(output: str, context_variables: Any, **kwargs: Any) -> FunctionTargetResult:
    """Valid function_target that accepts arbitrary keyword arguments via **kwargs."""
    return FunctionTargetResult(
        messages=f"{output} (kwargs={json.dumps(kwargs, sort_keys=True)})",
        target=AgentTarget(test_agent),
    )


def invalid_fn(output: str) -> FunctionTargetResult:
    """Invalid function_target: only one positional argument."""
    return FunctionTargetResult(
        messages=output,
        target=AgentTarget(test_agent),
    )


###############################################################################
# FunctionTarget __init__ and basic behavior
###############################################################################


def test_fn_target_init():
    """FunctionTarget wraps the incoming callable and exposes basic metadata."""
    ft = FunctionTarget(minimal_correct_fn_target)

    assert ft.fn_name == "minimal_correct_fn_target"
    assert ft.fn is minimal_correct_fn_target
    assert ft.extra_args == {}


def test_fn_target_init_with_extra_args():
    """FunctionTarget correctly stores and exposes extra_args."""
    extra_args = {"extra_param": 100}
    ft = FunctionTarget(correct_fn_target_extra_args, extra_args=extra_args)

    assert ft.fn_name == "correct_fn_target_extra_args"
    assert ft.fn is correct_fn_target_extra_args
    assert ft.extra_args == extra_args


def test_fn_target_init_invalid_args():
    """Functions with fewer than 2 positional params must be rejected."""
    with pytest.raises(ValueError, match="must accept at least two positional parameters"):
        FunctionTarget(invalid_fn)


def test_fn_target_init_non_callable_raises():
    """Non-callable incoming_fn should raise a clear ValueError."""
    with pytest.raises(ValueError, match="must be initialized with a callable function"):
        FunctionTarget("not_a_callable")  # type: ignore[arg-type]


def test_function_target_display_and_normalized_name():
    """display_name, normalized_name and __str__ expose user-friendly information."""
    ft = FunctionTarget(minimal_correct_fn_target)

    assert ft.display_name() == "minimal_correct_fn_target"
    assert ft.normalized_name() == "minimal_correct_fn_target"
    assert "minimal_correct_fn_target" in str(ft)


def test_function_target_never_needs_wrapper_and_raises_if_created():
    """FunctionTarget is executed inline and should not create wrapper agents."""
    ft = FunctionTarget(minimal_correct_fn_target)

    assert ft.needs_agent_wrapper() is False

    with pytest.raises(NotImplementedError, match="executed inline and needs no wrapper"):
        ft.create_wrapper_agent(parent_agent=test_agent, index=0)


###############################################################################
# validate_fn_sig tests
###############################################################################


def test_validate_fn_sig_valid_fn():
    """Ensure that valid function targets pass signature validation."""
    validate_fn_sig(minimal_correct_fn_target, extra_args={})
    validate_fn_sig(correct_fn_target_extra_args, extra_args={"extra_param": 100})


def test_validate_fn_sig_raises_on_missing_args():
    """Required additional parameters must be provided via extra_args."""
    with pytest.raises(ValueError, match="Missing required extra_args"):
        validate_fn_sig(correct_fn_target_extra_args, extra_args={})


def test_validate_fn_sig_raises_on_invalid_args():
    """extra_args keys must correspond to parameters when **kwargs is not used."""
    with pytest.raises(ValueError, match="Invalid extra_args for function"):
        validate_fn_sig(minimal_correct_fn_target, extra_args={"missing_param": 42})


def test_validate_fn_sig_allows_optional_params_without_extra_args():
    """Optional parameters with defaults must not require extra_args."""
    # Should not raise
    validate_fn_sig(fn_with_optional_param, extra_args={})


def test_validate_fn_sig_allows_kwargs_for_unknown_extra_args():
    """When **kwargs is present, arbitrary extra_args keys are allowed."""
    # No error expected even though 'foo' is not explicitly listed as a param
    validate_fn_sig(fn_with_kwargs, extra_args={"foo": 1, "bar": 2})


###############################################################################
# construct_broadcast_messages_list tests
###############################################################################


def test_construct_broadcast_messages_list_with_string_and_agenttarget():
    """String messages + AgentTarget -> single FunctionTargetMessage to that agent."""
    current_agent = ConversableAgent(name="current", llm_config=None)
    recipient = ConversableAgent(name="recipient", llm_config=None)

    groupchat = GroupChat(agents=[current_agent, recipient], messages=[])
    target = AgentTarget(recipient)

    messages_list = construct_broadcast_messages_list(
        messages="hello",
        group_chat=groupchat,
        current_agent=current_agent,
        target=target,
        user_agent=None,
    )

    assert len(messages_list) == 1
    assert isinstance(messages_list[0], FunctionTargetMessage)
    assert messages_list[0].content == "hello"
    assert messages_list[0].msg_target is recipient


def test_construct_broadcast_messages_list_with_string_and_agentnametarget():
    """String messages + AgentNameTarget -> resolved by agent.name in GroupChat."""
    current_agent = ConversableAgent(name="current", llm_config=None)
    recipient = ConversableAgent(name="named_agent", llm_config=None)

    groupchat = GroupChat(agents=[current_agent, recipient], messages=[])
    target = AgentNameTarget(agent_name="named_agent")

    messages_list = construct_broadcast_messages_list(
        messages="hello name target",
        group_chat=groupchat,
        current_agent=current_agent,
        target=target,
        user_agent=None,
    )

    assert len(messages_list) == 1
    msg = messages_list[0]
    assert msg.content == "hello name target"
    assert msg.msg_target is recipient


def test_construct_broadcast_messages_list_raises_if_agent_name_not_found():
    """AgentNameTarget should fail clearly if the agent is not in the GroupChat."""
    current_agent = ConversableAgent(name="current", llm_config=None)
    groupchat = GroupChat(agents=[current_agent], messages=[])
    target = AgentNameTarget(agent_name="missing_agent")

    with pytest.raises(ValueError, match="No agent found with in the group chat matching the target agent name"):
        construct_broadcast_messages_list(
            messages="hello",
            group_chat=groupchat,
            current_agent=current_agent,
            target=target,
            user_agent=None,
        )


def test_construct_broadcast_messages_list_with_string_and_revert_to_user_target():
    """String messages + RevertToUserTarget go to user_agent when provided."""
    current_agent = ConversableAgent(name="current", llm_config=None)
    user_agent = ConversableAgent(name="user", llm_config=None)
    groupchat = GroupChat(agents=[current_agent, user_agent], messages=[])

    target = RevertToUserTarget()

    messages_list = construct_broadcast_messages_list(
        messages="hello user",
        group_chat=groupchat,
        current_agent=current_agent,
        target=target,
        user_agent=user_agent,
    )

    assert len(messages_list) == 1
    assert messages_list[0].msg_target is user_agent


def test_construct_broadcast_messages_list_with_string_and_revert_to_user_no_user_defaults_to_current():
    """If RevertToUserTarget has no user_agent, fallback to current_agent."""
    current_agent = ConversableAgent(name="current", llm_config=None)
    groupchat = GroupChat(agents=[current_agent], messages=[])

    target = RevertToUserTarget()

    messages_list = construct_broadcast_messages_list(
        messages="fallback",
        group_chat=groupchat,
        current_agent=current_agent,
        target=target,
        user_agent=None,
    )

    assert len(messages_list) == 1
    assert messages_list[0].msg_target is current_agent


def test_construct_broadcast_messages_list_with_string_and_staytarget():
    """String messages + StayTarget -> message to current_agent."""
    current_agent = ConversableAgent(name="current", llm_config=None)
    other_agent = ConversableAgent(name="other", llm_config=None)
    groupchat = GroupChat(agents=[current_agent, other_agent], messages=[])

    target = StayTarget()

    messages_list = construct_broadcast_messages_list(
        messages="stay here",
        group_chat=groupchat,
        current_agent=current_agent,
        target=target,
        user_agent=None,
    )

    assert len(messages_list) == 1
    assert messages_list[0].msg_target is current_agent


def test_construct_broadcast_messages_list_passthrough_for_list_of_messages():
    """If messages is already a list[FunctionTargetMessage], it should be returned as-is."""
    current_agent = ConversableAgent(name="current", llm_config=None)
    other_agent = ConversableAgent(name="other", llm_config=None)
    groupchat = GroupChat(agents=[current_agent, other_agent], messages=[])

    msgs = [
        FunctionTargetMessage(content="to other", msg_target=other_agent),
        FunctionTargetMessage(content="to current", msg_target=current_agent),
    ]

    messages_list = construct_broadcast_messages_list(
        messages=msgs,
        group_chat=groupchat,
        current_agent=current_agent,
        target=StayTarget(),
        user_agent=None,
    )

    assert messages_list is msgs
    assert [m.msg_target for m in messages_list] == [other_agent, current_agent]


###############################################################################
# broadcast tests (using DummyGroupManager)
###############################################################################


def test_broadcast_string_to_agent_uses_group_manager():
    """broadcast() should wrap content in a FUNCTION_HANDOFF system message and send via _group_manager."""
    current_agent = ConversableAgent(name="current", llm_config=None)
    target_agent = ConversableAgent(name="target", llm_config=None)

    groupchat = GroupChat(agents=[current_agent, target_agent], messages=[])
    current_agent._group_manager = DummyGroupManager()  # type: ignore[attr-defined]

    target = AgentTarget(target_agent)
    messages = "Hello, World!"
    fn_name = "minimal_correct_fn_target"

    broadcast(
        messages=messages,
        group_chat=groupchat,
        current_agent=current_agent,
        fn_name=fn_name,
        target=target,
    )

    gm: DummyGroupManager = current_agent._group_manager  # type: ignore[assignment]
    assert len(gm.sent) == 1
    record = gm.sent[0]

    assert record["recipient"] is target_agent
    assert record["request_reply"] is False
    assert record["silent"] is False

    msg = record["msg"]
    assert msg["role"] == "system"
    assert msg["name"] == fn_name
    assert "[FUNCTION_HANDOFF] - Reply from function minimal_correct_fn_target" in msg["content"]


def test_broadcast_list_of_messages_sends_to_all_targets():
    """broadcast() should send one message per FunctionTargetMessage."""
    current_agent = ConversableAgent(name="current", llm_config=None)
    agent1 = ConversableAgent(name="agent1", llm_config=None)
    agent2 = ConversableAgent(name="agent2", llm_config=None)

    groupchat = GroupChat(agents=[current_agent, agent1, agent2], messages=[])
    current_agent._group_manager = DummyGroupManager()  # type: ignore[attr-defined]

    msgs = [
        FunctionTargetMessage(content="to agent1", msg_target=agent1),
        FunctionTargetMessage(content="to agent2", msg_target=agent2),
    ]

    broadcast(
        messages=msgs,
        group_chat=groupchat,
        current_agent=current_agent,
        fn_name="multi_fn",
        target=StayTarget(),
    )

    gm: DummyGroupManager = current_agent._group_manager  # type: ignore[assignment]
    assert len(gm.sent) == 2
    recipients = [r["recipient"] for r in gm.sent]
    assert recipients == [agent1, agent2]


def test_broadcast_raises_without_group_manager():
    """If current_agent lacks a _group_manager, broadcast should fail loudly."""
    current_agent = ConversableAgent(name="current", llm_config=None)
    agent1 = ConversableAgent(name="agent1", llm_config=None)
    groupchat = GroupChat(agents=[current_agent, agent1], messages=[])

    target = AgentTarget(agent1)

    with pytest.raises(ValueError, match="Current agent must have a group manager"):
        broadcast(
            messages="hi",
            group_chat=groupchat,
            current_agent=current_agent,
            fn_name="fn_name",
            target=target,
        )


###############################################################################
# FunctionTarget.resolve integration tests
###############################################################################


def function_target_updates_context_and_messages(
    output: str, ctx: DummyContextVariables, value: int
) -> FunctionTargetResult:
    """Test function that updates context_variables and emits a reply."""
    new_ctx = ContextVariables(data={"from_fn": value, "last_output": output})
    return FunctionTargetResult(
        messages="fn reply",
        context_variables=new_ctx,
        target=StayTarget(),
    )


def function_target_no_messages_just_target(output: str, ctx: Any) -> FunctionTargetResult:
    """Test function that only returns a target and no messages."""
    return FunctionTargetResult(
        messages=None,
        context_variables=None,
        target=StayTarget(),
    )


def function_target_returns_wrong_type(output: str, ctx: Any) -> str:  # type: ignore[override]
    """Intentionally incorrect: returns a plain string, not FunctionTargetResult."""
    return "not a result"


def test_function_target_resolve_updates_context_and_broadcasts(monkeypatch: MonkeyPatch):
    """resolve() should:
    - pass last message content and context into the function,
    - update current_agent.context_variables if provided,
    - broadcast messages,
    - and return the next SpeakerSelectionResult from the target.
    """
    current_agent = ConversableAgent(name="current", llm_config=None)
    current_agent.context_variables = DummyContextVariables({"existing": "value"})  # type: ignore[attr-defined]
    current_agent._group_manager = DummyGroupManager()  # type: ignore[attr-defined]

    groupchat = GroupChat(
        agents=[current_agent],
        messages=[{"role": "user", "content": "last user message"}],
    )

    ft = FunctionTarget(
        function_target_updates_context_and_messages,
        extra_args={"value": 7},
    )

    result = ft.resolve(
        groupchat=groupchat,
        current_agent=current_agent,
        user_agent=None,
    )

    # Returned object should be a SpeakerSelectionResult
    assert isinstance(result, SpeakerSelectionResult)

    # Context variables should be updated with new values
    ctx: DummyContextVariables = current_agent.context_variables  # type: ignore[assignment]
    ctx_dict = ctx.to_dict()
    assert ctx_dict["existing"] == "value"
    assert ctx_dict["from_fn"] == 7
    assert ctx_dict["last_output"] == "last user message"

    # A FUNCTION_HANDOFF system message should have been broadcast
    gm: DummyGroupManager = current_agent._group_manager  # type: ignore[assignment]
    assert len(gm.sent) == 1
    content = gm.sent[0]["msg"]["content"]
    assert "Reply from function function_target_updates_context_and_messages" in content
    assert "fn reply" in content


def test_function_target_resolve_without_messages_does_not_broadcast():
    """If FunctionTargetResult.messages is None, no broadcast should occur."""
    current_agent = ConversableAgent(name="current", llm_config=None)
    current_agent.context_variables = DummyContextVariables({})  # type: ignore[attr-defined]
    current_agent._group_manager = DummyGroupManager()  # type: ignore[attr-defined]

    groupchat = GroupChat(
        agents=[current_agent],
        messages=[{"role": "user", "content": "hello"}],
    )

    ft = FunctionTarget(function_target_no_messages_just_target)

    result = ft.resolve(
        groupchat=groupchat,
        current_agent=current_agent,
        user_agent=None,
    )

    assert isinstance(result, SpeakerSelectionResult)

    gm: DummyGroupManager = current_agent._group_manager  # type: ignore[assignment]
    assert gm.sent == []  # No messages were broadcast


def test_function_target_resolve_raises_if_function_returns_wrong_type():
    """If the wrapped function does not return a FunctionTargetResult, resolve must raise ValueError."""
    current_agent = ConversableAgent(name="current", llm_config=None)
    current_agent.context_variables = DummyContextVariables({})  # type: ignore[attr-defined]
    current_agent._group_manager = DummyGroupManager()  # type: ignore[attr-defined]

    groupchat = GroupChat(
        agents=[current_agent],
        messages=[{"role": "user", "content": "hello"}],
    )

    ft = FunctionTarget(function_target_returns_wrong_type)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="must return a FunctionTargetResult object"):
        ft.resolve(
            groupchat=groupchat,
            current_agent=current_agent,
            user_agent=None,
        )


def test_function_target_resolve_uses_provided_target_resolve(monkeypatch: MonkeyPatch):
    """Ensure that FunctionTarget.resolve delegates to the underlying TransitionTarget.resolve."""
    current_agent = ConversableAgent(name="current", llm_config=None)
    current_agent.context_variables = DummyContextVariables({})  # type: ignore[attr-defined]
    current_agent._group_manager = DummyGroupManager()  # type: ignore[attr-defined]

    groupchat = GroupChat(
        agents=[current_agent],
        messages=[{"role": "user", "content": "hello"}],
    )

    # Build a custom FunctionTargetResult with a StayTarget, but we monkeypatch its resolve.
    stay_target = StayTarget()

    sentinel = SpeakerSelectionResult(next_agent=current_agent, reason="patched")

    with mock.patch.object(
        type(stay_target),
        "resolve",
        return_value=sentinel,
    ) as mock_resolve:

        def fn(output: str, ctx: Any) -> FunctionTargetResult:
            return FunctionTargetResult(messages=None, context_variables=None, target=stay_target)

        ft = FunctionTarget(fn)

        result = ft.resolve(
            groupchat=groupchat,
            current_agent=current_agent,
            user_agent=None,
        )

        mock_resolve.assert_called_once_with(groupchat, current_agent, None)
        assert result is sentinel
