# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
"""
Minimal FunctionTarget test wiring for a two-agent group chat.
"""

import os
from typing import Any

from dotenv import load_dotenv

from autogen import ConversableAgent, LLMConfig
from autogen.agentchat import initiate_group_chat
from autogen.agentchat.group import AgentTarget, ContextVariables, FunctionTarget
from autogen.agentchat.group.function_target_result import FunctionTargetMessage, FunctionTargetResult
from autogen.agentchat.group.patterns import DefaultPattern
from autogen.agentchat.group.targets.transition_target import StayTarget

load_dotenv()


def main(session_id: str | None = None) -> dict:
    # LLM config
    cfg = LLMConfig(api_type="openai", model="gpt-4o", api_key=os.environ["OPENAI_API_KEY"])

    # Shared context
    ctx = ContextVariables(data={"application": "<empty>"})

    # Agents
    first_agent = ConversableAgent(
        name="first_agent",
        llm_config=cfg,
        system_message="Output a sample email you would send to apply to a job in tech. "
        "Listen to the specifics of the instructions.",
    )

    second_agent = ConversableAgent(
        name="second_agent",
        llm_config=cfg,
        system_message="Do whatever the message sent to you tells you to do.",
    )

    user_agent = ConversableAgent(
        name="user",
        human_input_mode="ALWAYS",
    )

    # After-work hook
    def afterwork_function(output: str, context_variables: Any, *args) -> FunctionTargetResult:
        """
        Switches a context variable and routes the next turn.
        """
        if context_variables.get("application") == "<empty>":
            context_variables["application"] = output
            return FunctionTargetResult(
                messages="apply for a job in gpu optimization",
                target=StayTarget(),
                context_variables=context_variables,
            )

        return FunctionTargetResult(
            messages=[
                FunctionTargetMessage(
                    content=f"Revise the draft written by the first agent: {output}", msg_target=second_agent
                )
            ],
            target=AgentTarget(second_agent),
            context_variables=context_variables,
        )

    # Conversation pattern
    pattern = DefaultPattern(
        initial_agent=first_agent,
        agents=[first_agent, second_agent],
        user_agent=user_agent,
        context_variables=ctx,
        group_manager_args={"llm_config": cfg},
    )

    # Register after-work handoff
    first_agent.handoffs.set_after_work(FunctionTarget(afterwork_function))

    # Run
    initiate_group_chat(
        pattern=pattern,
        messages="the job you are applying to is specifically in machine learning",
        max_rounds=20,
    )

    return {"session_id": session_id}


if __name__ == "__main__":
    main()
