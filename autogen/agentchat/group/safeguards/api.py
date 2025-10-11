# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ....llm_config import LLMConfig

# Import types that are safe to import at runtime
from ...agent import Agent
from ...conversable_agent import ConversableAgent
from ...groupchat import GroupChatManager

if TYPE_CHECKING:
    from .enforcer import SafeguardEnforcer


def reset_safeguard_policy(
    *,
    agents: list[ConversableAgent] | None = None,
    groupchat_manager: GroupChatManager | None = None,
) -> None:
    """Reset/remove all safeguards from agents and groupchat managers.

    This function removes all safeguard hooks and inter-agent guardrails that were
    previously applied by apply_safeguard_policy. Events are sent to track the reset process.

    Args:
        agents: List of agents to remove safeguards from (optional if groupchat_manager provided)
        groupchat_manager: GroupChatManager to remove safeguards from (optional if agents provided)

    Example:
        ```python
        from autogen.agentchat.group.safeguards import reset_safeguard_policy

        # Remove safeguards from agents
        reset_safeguard_policy(agents=[agent1, agent2, agent3])

        # Or remove from GroupChatManager
        reset_safeguard_policy(groupchat_manager=manager)
        ```
    """
    from ....events.print_event import PrintEvent
    from ....io.base import IOStream

    iostream = IOStream.get_default()

    # Send initial reset event
    iostream.send(PrintEvent("Resetting safeguards..."))
    # Determine which agents to remove safeguards from
    target_agents: list[ConversableAgent | Agent] = []

    if groupchat_manager:
        if not isinstance(groupchat_manager, GroupChatManager):
            raise ValueError("groupchat_manager must be an instance of GroupChatManager")

        target_agents.extend([agent for agent in groupchat_manager.groupchat.agents if hasattr(agent, "hook_lists")])

        agent_names = [agent.name for agent in target_agents]
        iostream.send(PrintEvent(f"📋 Found {len(target_agents)} agents in GroupChat: {', '.join(agent_names)}"))

        # Clear inter-agent guardrails from the groupchat
        if hasattr(groupchat_manager.groupchat, "_inter_agent_guardrails"):
            guardrail_count = len(groupchat_manager.groupchat._inter_agent_guardrails)
            if guardrail_count > 0:
                iostream.send(PrintEvent(f"🔗 Clearing {guardrail_count} inter-agent guardrails from GroupChat"))
            groupchat_manager.groupchat._inter_agent_guardrails.clear()

    elif agents:
        target_agents.extend(agents)
        agent_names = [agent.name for agent in target_agents]
        iostream.send(PrintEvent(f"📋 Resetting safeguards for {len(target_agents)} agents: {', '.join(agent_names)}"))
    else:
        raise ValueError("Either agents or groupchat_manager must be provided")

    # Remove safeguard hooks from each agent
    safeguard_hook_names = [
        "safeguard_tool_inputs",
        "safeguard_tool_outputs",
        "safeguard_llm_inputs",
        "safeguard_llm_outputs",
        "safeguard_human_inputs",
        "process_message_before_send",  # Inter-agent communication hooks for direct agent communication
    ]

    for agent in target_agents:
        if hasattr(agent, "hook_lists"):
            # Use the agent's reset_safeguards method for agent-to-environment safeguards
            if hasattr(agent, "reset_safeguards"):
                agent.reset_safeguards()
            else:
                # Fallback to manual clearing for older agent versions
                for hook_name in safeguard_hook_names:
                    if hook_name in agent.hook_lists:
                        # Clear all hooks in safeguard-specific hook lists
                        agent.hook_lists[hook_name].clear()

            # Manually clear inter-agent safeguards (process_message_before_send)
            if "process_message_before_send" in agent.hook_lists:
                agent.hook_lists["process_message_before_send"].clear()
        else:
            raise ValueError(
                f"Agent {agent.name} does not support hooks. Please ensure it inherits from ConversableAgent."
            )

    # Send completion event
    iostream.send(PrintEvent(f"✅ Safeguard reset completed for {len(target_agents)} agents"))


def apply_safeguard_policy(
    *,
    agents: list[ConversableAgent] | None = None,
    groupchat_manager: GroupChatManager | None = None,
    policy: dict[str, Any] | str,
    safeguard_llm_config: LLMConfig | None = None,
    mask_llm_config: LLMConfig | None = None,
) -> SafeguardEnforcer:
    """Apply safeguards to agents using a policy file.

    This is the main function for applying safeguards. It supports the policy format
    with 'inter_agent_safeguards' and 'agent_environment_safeguards' sections.

    Args:
        agents: List of agents to apply safeguards to (optional if groupchat_manager provided)
        groupchat_manager: GroupChatManager to apply safeguards to (optional if agents provided)
        policy: Safeguard policy dict or path to JSON file
        safeguard_llm_config: LLM configuration for safeguard checks
        mask_llm_config: LLM configuration for masking

    Returns:
        SafeguardEnforcer instance for further configuration

    Example:
        ```python
        from autogen.agentchat.group.safeguards import apply_safeguard_policy

        # Apply safeguards to agents
        safeguard_enforcer = apply_safeguard_policy(
            agents=[agent1, agent2, agent3],
            policy="path/to/policy.json",
            safeguard_llm_config=safeguard_llm_config,
        )

        # Or apply to GroupChatManager
        safeguard_enforcer = apply_safeguard_policy(
            groupchat_manager=manager,
            policy="path/to/policy.json",
            safeguard_llm_config=safeguard_llm_config,
            mask_llm_config=mask_llm_config,
        )
        ```
    """
    from .enforcer import SafeguardEnforcer

    enforcer = SafeguardEnforcer(
        policy=policy,
        safeguard_llm_config=safeguard_llm_config,
        mask_llm_config=mask_llm_config,
        groupchat_manager=groupchat_manager,
        agents=agents,
    )

    # Determine which agents to apply safeguards to
    target_agents: list[ConversableAgent | Agent] = []
    all_agent_names = []

    if groupchat_manager:
        if not isinstance(groupchat_manager, GroupChatManager):
            raise ValueError("groupchat_manager must be an instance of GroupChatManager")

        target_agents.extend([
            agent
            for agent in groupchat_manager.groupchat.agents
            if hasattr(agent, "hook_lists") and agent.name != "_Group_Tool_Executor"
        ])
        all_agent_names = [
            agent.name for agent in groupchat_manager.groupchat.agents if agent.name != "_Group_Tool_Executor"
        ]
        all_agent_names.append(groupchat_manager.name)

        # Register inter-agent guardrails with the groupchat
        # Ensure the list exists and append our enforcer
        if not hasattr(groupchat_manager.groupchat, "_inter_agent_guardrails"):
            groupchat_manager.groupchat._inter_agent_guardrails = []
        groupchat_manager.groupchat._inter_agent_guardrails.clear()  # Clear any existing
        groupchat_manager.groupchat._inter_agent_guardrails.append(enforcer)
    elif agents:
        target_agents.extend(agents)
        all_agent_names = [agent.name for agent in agents]
    else:
        raise ValueError("Either agents or groupchat_manager must be provided")

    # Build agent-to-tool mapping for validation
    agent_tool_mapping = {}
    for agent in target_agents:
        agent_tools = []

        # Get tools from the tools property (Tool objects)
        if hasattr(agent, "tools"):
            for tool in agent.tools:
                agent_tools.append(tool.name)

        # Get tools from function_map (functions registered with @register_for_execution)
        if hasattr(agent, "function_map"):
            agent_tools.extend(agent.function_map.keys())

        agent_tool_mapping[agent.name] = agent_tools

    # Validate policy including agent names and tool names
    try:
        from .validator import SafeguardValidator

        validator = SafeguardValidator(enforcer.policy)  # Use enforcer's loaded policy dict
        validator.validate_policy_complete(agent_names=all_agent_names, agent_tool_mapping=agent_tool_mapping)
    except ValueError as e:
        raise ValueError(f"Policy validation failed: {e}")

    # Apply hooks to each agent
    for agent in target_agents:
        if hasattr(agent, "hook_lists"):
            hooks = enforcer.create_agent_hooks(agent.name)
            for hook_name, hook_func in hooks.items():
                if hook_name in agent.hook_lists:
                    agent.hook_lists[hook_name].append(hook_func)
        else:
            raise ValueError(
                f"Agent {agent.name} does not support hooks. Please ensure it inherits from ConversableAgent."
            )

    # Apply hooks to GroupToolExecutor if it exists (for GroupChat scenarios)
    if groupchat_manager and enforcer.group_tool_executor and hasattr(enforcer.group_tool_executor, "hook_lists"):
        # Create hooks for GroupToolExecutor - it needs tool interaction hooks
        # since it's the one actually executing tools in GroupChat
        hooks = enforcer.create_agent_hooks(enforcer.group_tool_executor.name)
        for hook_name, hook_func in hooks.items():
            if hook_name in enforcer.group_tool_executor.hook_lists:
                enforcer.group_tool_executor.hook_lists[hook_name].append(hook_func)

    return enforcer
