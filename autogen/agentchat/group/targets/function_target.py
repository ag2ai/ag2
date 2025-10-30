# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from ...agent import Agent
from ..context_variables import ContextVariables
from ..speaker_selection_result import SpeakerSelectionResult
from .transition_target import AgentNameTarget, AgentTarget, RevertToUserTarget, StayTarget, TransitionTarget

if TYPE_CHECKING:
    from ...conversable_agent import ConversableAgent
    from ...groupchat import GroupChat

__all__ = ["FunctionTarget", "FunctionTargetMessage", "FunctionTargetResult", "broadcast"]


class FunctionTargetMessage(BaseModel):
    """Message and target that can be sent as part of the FunctionTargetResult.

    Attributes:
        content: The content of the message to be sent.
        msg_target: The agent to whom the message is to be sent.
    """

    content: str
    msg_target: Agent

    class Config:
        arbitrary_types_allowed = True


class FunctionTargetResult(BaseModel):
    """Result of a function handoff that is used to provide the return message and the target to transition to.

    Attributes:
        messages: Optional list of messages to be broadcast to specific agents, or a single string message.
        context_variables: Optional updated context variables that will be applied to the group chat context variables.
        target: The next target to transition to.
    """

    messages: list[FunctionTargetMessage] | str | None = None
    context_variables: ContextVariables | None = None
    target: TransitionTarget


def construct_broadcast_messages_list(
    messages: list[FunctionTargetMessage] | str,
    group_chat: GroupChat,
    current_agent: ConversableAgent,
    target: TransitionTarget,
    user_agent: ConversableAgent | None = None,
) -> list[FunctionTargetMessage]:
    """Construct a list of FunctionTargetMessage from input messages and target."""
    if isinstance(messages, str):
        if isinstance(target, (AgentTarget, AgentNameTarget)):
            next_target = target.agent_name
            for agent in group_chat.agents:
                if agent.name == next_target:
                    messages_list = [FunctionTargetMessage(content=messages, msg_target=agent)]
                    break
            else:
                raise ValueError(f"No agent found with in the group chat matching the target agent name {next_target}.")
        elif isinstance(target, RevertToUserTarget) and user_agent is not None:
            messages_list = [FunctionTargetMessage(content=messages, msg_target=user_agent)]
        elif isinstance(target, StayTarget):
            messages_list = [FunctionTargetMessage(content=messages, msg_target=current_agent)]
        else:
            # Default to current agent if no target is not agent-based is found
            messages_list = [FunctionTargetMessage(content=messages, msg_target=current_agent)]
    else:
        messages_list = messages
    return messages_list


def broadcast(
    messages: list[FunctionTargetMessage] | str,
    group_chat: GroupChat,
    current_agent: ConversableAgent,
    fn_name: str,
    target: TransitionTarget,
    user_agent: ConversableAgent | None = None,
) -> None:
    """Broadcast message(s) to their target agent."""
    messages_list = construct_broadcast_messages_list(messages, group_chat, current_agent, target, user_agent)

    for message in messages_list:
        content = message.content
        broadcast = {
            "role": "system",
            "name": f"{fn_name}",
            "content": f"[FUNCTION_HANDOFF] - Reply from function {fn_name}: \n\n {content}",
        }
        if hasattr(current_agent, "_group_manager") and current_agent._group_manager is not None:
            current_agent._group_manager.send(
                broadcast,
                message.msg_target,
                request_reply=False,
                silent=False,
            )
        else:
            raise ValueError("Current agent must have a group manager to broadcast messages.")


def validate_fn_sig(
    incoming_fn: Callable[..., FunctionTargetResult],
    extra_args: dict[str, Any],
) -> None:
    """
    Validate a user-defined afterwork_function signature.

    Rules:
    1. Must have at least two positional parameters (whatever their names).
    2. All provided extra_args must exist in the function signature (unless **kwargs is present).
    3. All additional required (non-default) params beyond the first two must be satisfied via extra_args.
    """
    sig = inspect.signature(incoming_fn)
    params = list(sig.parameters.values())

    # 1️⃣ Must have at least two positional parameters (whatever names)
    if len(params) < 2:
        raise ValueError(
            f"Function '{incoming_fn.__name__}' must accept at least two positional parameters: "
            f"(output, ctx). Current: {[p.name for p in params]}"
        )

    # 2️⃣ Detect **kwargs
    has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)

    # Build list of extra (non-core) params after the first two
    extra_params = [
        p for p in params[2:] if p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]

    # 3️⃣ If **kwargs not present, check for invalid extra_args
    if not has_kwargs:
        valid_names = {p.name for p in params}
        invalid_keys = [k for k in extra_args if k not in valid_names]
        if invalid_keys:
            raise ValueError(
                f"Invalid extra_args for function '{incoming_fn.__name__}': {invalid_keys}. "
                f"Allowed parameters are: {[p.name for p in params]}."
            )

    # 4️⃣ Check for missing required params (no defaults) among the extra ones
    missing_keys = [p.name for p in extra_params if p.default is p.empty and p.name not in extra_args]

    if missing_keys:
        raise ValueError(
            f"Missing required extra_args for function '{incoming_fn.__name__}': {missing_keys}. "
            f"You must supply them via `extra_args`."
        )


class FunctionTarget(TransitionTarget):
    """Transition target that invokes a tool function with (prev_output, context).

    The function must return a FunctionTargetResult object that includes the next target to transition to.
    """

    fn_name: str = Field(...)
    fn: Callable[..., FunctionTargetResult] = Field(..., repr=False)
    extra_args: dict[str, Any] = Field(default_factory=dict)

    def __init__(
        self,
        incoming_fn: Callable[..., FunctionTargetResult],
        *,
        extra_args: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if callable(incoming_fn):
            extra_args = extra_args or {}

            validate_fn_sig(incoming_fn, extra_args)

            super().__init__(fn_name=incoming_fn.__name__, fn=incoming_fn, extra_args=extra_args, **kwargs)
        else:
            raise ValueError(
                "FunctionTarget must be initialized with a callable function as the first argument or 'fn' keyword argument."
            )

    def can_resolve_for_speaker_selection(self) -> bool:
        return False

    def resolve(
        self,
        groupchat: GroupChat,
        current_agent: ConversableAgent,
        user_agent: ConversableAgent | None,
    ) -> SpeakerSelectionResult:
        """Invoke the function, update context variables (optional), broadcast messages (optional), and return the next target to transition to."""
        last_message = (
            groupchat.messages[-1]["content"] if groupchat.messages and "content" in groupchat.messages[-1] else ""
        )

        # Run the function to get the FunctionTargetResult
        function_target_result = self.fn(
            last_message,
            current_agent.context_variables,
            **self.extra_args,
        )

        if not isinstance(function_target_result, FunctionTargetResult):
            raise ValueError("FunctionTarget function must return a FunctionTargetResult object.")

        if function_target_result.context_variables:
            # Update the group's Context Variables if the function returned any
            current_agent.context_variables.update(function_target_result.context_variables.to_dict())

        if function_target_result.messages:
            # If we have messages, we need to broadcast them to the appropriate agent based on the target
            broadcast(
                function_target_result.messages,
                groupchat,
                current_agent,
                self.fn_name,
                function_target_result.target,
                user_agent,
            )

        # Resolve and return the next target
        return function_target_result.target.resolve(groupchat, current_agent, user_agent)

    def display_name(self) -> str:
        return self.fn_name

    def normalized_name(self) -> str:
        return self.fn_name.replace(" ", "_")

    def __str__(self) -> str:
        return f"Transfer to tool {self.fn_name}"

    def needs_agent_wrapper(self) -> bool:
        return False

    def create_wrapper_agent(self, parent_agent: ConversableAgent, index: int) -> ConversableAgent:
        raise NotImplementedError("FunctionTarget is executed inline and needs no wrapper")
