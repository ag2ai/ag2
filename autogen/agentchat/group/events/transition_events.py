# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from typing import Any

from autogen.agentchat.agent import Agent
from autogen.agentchat.group.targets.transition_target import TransitionTarget
from autogen.events.base_event import BaseEvent, wrap_event
from autogen.formatting_utils import colored


@wrap_event
class AfterWorksTransitionEvent(BaseEvent):
    """Event for after works handoffs"""

    model_config = {"arbitrary_types_allowed": True}

    source_agent: Agent
    transition_target: TransitionTarget

    def __init__(self, source_agent: Agent, transition_target: TransitionTarget):
        super().__init__(source_agent=source_agent, transition_target=transition_target)

    def print(self, f: Callable[..., Any] | None = None) -> None:
        f = f or print
        super().print(f)

        f(
            colored(
                f"***** AfterWork handoff ({self.source_agent.name if hasattr(self.source_agent, 'name') else self.source_agent}): {self.transition_target.display_name()} *****",
                "blue",
            ),
            flush=True,
        )


@wrap_event
class OnContextConditionTransitionEvent(BaseEvent):
    """Event for OnContextCondition handoffs"""

    model_config = {"arbitrary_types_allowed": True}

    source_agent: Agent
    transition_target: TransitionTarget

    def __init__(self, source_agent: Agent, transition_target: TransitionTarget):
        super().__init__(source_agent=source_agent, transition_target=transition_target)

    def print(self, f: Callable[..., Any] | None = None) -> None:
        f = f or print
        super().print(f)

        f(
            colored(
                f"***** OnContextCondition handoff ({self.source_agent.name if hasattr(self.source_agent, 'name') else self.source_agent}): {self.transition_target.display_name()} *****",
                "blue",
            ),
            flush=True,
        )


@wrap_event
class OnConditionLLMTransitionEvent(BaseEvent):
    """Event for LLM-based OnCondition handoffs"""

    model_config = {"arbitrary_types_allowed": True}

    source_agent: Agent
    transition_target: TransitionTarget

    def __init__(self, source_agent: Agent, transition_target: TransitionTarget):
        super().__init__(source_agent=source_agent, transition_target=transition_target)

    def print(self, f: Callable[..., Any] | None = None) -> None:
        f = f or print
        super().print(f)

        f(
            colored(
                f"***** LLM-based OnCondition handoff ({self.source_agent.name if hasattr(self.source_agent, 'name') else self.source_agent}): {self.transition_target.display_name()} *****",
                "blue",
            ),
            flush=True,
        )


@wrap_event
class ReplyResultTransitionEvent(BaseEvent):
    """Event for reply result transitions"""

    model_config = {"arbitrary_types_allowed": True}

    source_agent: Agent
    transition_target: TransitionTarget

    def __init__(self, source_agent: Agent, transition_target: TransitionTarget):
        super().__init__(source_agent=source_agent, transition_target=transition_target)

    def print(self, f: Callable[..., Any] | None = None) -> None:
        f = f or print
        super().print(f)

        f(
            colored(
                f"***** ReplyResult transition ({self.source_agent.name if hasattr(self.source_agent, 'name') else self.source_agent}): {self.transition_target.display_name()} *****",
                "blue",
            ),
            flush=True,
        )
