# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from typing import Any


class SafeguardValidator:
    """Validator for safeguard policy format and content."""

    def __init__(self, policy: dict[str, Any]):
        """Initialize validator with policy.

        Args:
            policy: The safeguard policy to validate
        """
        self.policy = policy

    def validate_policy_structure(self) -> None:
        """Validate policy format and syntax only."""
        if not isinstance(self.policy, dict):
            raise ValueError("Policy must be a dictionary")

        # Validate inter-agent safeguards
        if "inter_agent_safeguards" in self.policy:
            self._validate_inter_agent_safeguards()

        # Validate environment safeguards
        if "agent_environment_safeguards" in self.policy:
            self._validate_environment_safeguards()

    def validate_policy_complete(self, agent_names: list[str], agent_tool_mapping: dict[str, list[str]]) -> None:
        """Validate agent and tool names (assumes policy structure already validated).

        Args:
            agent_names: List of available agent names for validation
            agent_tool_mapping: Mapping of agent names to their tool names
        """
        # Validate agent names
        self.validate_agent_names(agent_names)

        # Validate tool names if any tools exist
        if any(tools for tools in agent_tool_mapping.values()):
            self.validate_tool_names(agent_tool_mapping, agent_names)

    def _validate_inter_agent_safeguards(self) -> None:
        """Validate inter-agent safeguards section."""
        inter_agent = self.policy["inter_agent_safeguards"]
        if not isinstance(inter_agent, dict):
            raise ValueError("inter_agent_safeguards must be a dictionary")

        # Validate agent_transitions
        if "agent_transitions" in inter_agent:
            if not isinstance(inter_agent["agent_transitions"], list):
                raise ValueError("agent_transitions must be a list")

            for i, rule in enumerate(inter_agent["agent_transitions"]):
                if not isinstance(rule, dict):
                    raise ValueError(f"agent_transitions[{i}] must be a dictionary")

                # Required fields
                required_fields = ["message_source", "message_destination"]
                for field in required_fields:
                    if field not in rule:
                        raise ValueError(f"agent_transitions[{i}] missing required field: {field}")

                # Check method validation - no default, must be explicit
                if "check_method" not in rule:
                    raise ValueError(f"agent_transitions[{i}] missing required field: check_method")
                check_method = rule["check_method"]
                if check_method not in ["llm", "regex"]:
                    raise ValueError(
                        f"agent_transitions[{i}] invalid check_method: {check_method}. Must be 'llm' or 'regex'"
                    )

                # LLM-specific validation
                if check_method == "llm":
                    if "custom_prompt" not in rule and "disallow_item" not in rule:
                        raise ValueError(
                            f"agent_transitions[{i}] with check_method 'llm' must have either 'custom_prompt' or 'disallow_item'"
                        )
                    if "disallow_item" in rule and not isinstance(rule["disallow_item"], list):
                        raise ValueError(f"agent_transitions[{i}] disallow_item must be a list")

                # Regex-specific validation
                if check_method == "regex":
                    if "pattern" not in rule:
                        raise ValueError(f"agent_transitions[{i}] with check_method 'regex' must have 'pattern'")
                    if not isinstance(rule["pattern"], str):
                        raise ValueError(f"agent_transitions[{i}] pattern must be a string")

                    # Test regex pattern validity
                    try:
                        re.compile(rule["pattern"])
                    except re.error as e:
                        raise ValueError(f"agent_transitions[{i}] invalid regex pattern '{rule['pattern']}': {e}")

                # Validate action - no default, must be explicit
                if "violation_response" not in rule and "action" not in rule:
                    raise ValueError(f"agent_transitions[{i}] missing required field: violation_response or action")
                action = rule.get("violation_response", rule.get("action"))
                if action not in ["block", "mask", "warning"]:
                    raise ValueError(
                        f"agent_transitions[{i}] invalid action: {action}. Must be 'block', 'mask', or 'warning'"
                    )

        # Validate groupchat_message_check
        if "groupchat_message_check" in inter_agent:
            rule = inter_agent["groupchat_message_check"]
            if not isinstance(rule, dict):
                raise ValueError("groupchat_message_check must be a dictionary")
            if "disallow_item" in rule and not isinstance(rule["disallow_item"], list):
                raise ValueError("groupchat_message_check disallow_item must be a list")

    def _validate_environment_safeguards(self) -> None:
        """Validate environment safeguards section."""
        env_rules = self.policy["agent_environment_safeguards"]
        if not isinstance(env_rules, dict):
            raise ValueError("agent_environment_safeguards must be a dictionary")

        # Validate tool_interaction rules
        if "tool_interaction" in env_rules:
            if not isinstance(env_rules["tool_interaction"], list):
                raise ValueError("tool_interaction must be a list")

            for i, rule in enumerate(env_rules["tool_interaction"]):
                if not isinstance(rule, dict):
                    raise ValueError(f"tool_interaction[{i}] must be a dictionary")

                # Check method validation - no default, must be explicit
                if "check_method" not in rule:
                    raise ValueError(f"tool_interaction[{i}] missing required field: check_method")
                check_method = rule["check_method"]
                if check_method not in ["llm", "regex"]:
                    raise ValueError(
                        f"tool_interaction[{i}] invalid check_method: {check_method}. Must be 'llm' or 'regex'"
                    )

                # Validate action - no default, must be explicit
                if "violation_response" not in rule and "action" not in rule:
                    raise ValueError(f"tool_interaction[{i}] missing required field: violation_response or action")
                action = rule.get("violation_response", rule.get("action"))
                if action not in ["block", "mask", "warning"]:
                    raise ValueError(
                        f"tool_interaction[{i}] invalid action: {action}. Must be 'block', 'mask', or 'warning'"
                    )

                # All tool_interaction rules must have message_source and message_destination
                if "message_source" not in rule or "message_destination" not in rule:
                    raise ValueError(f"tool_interaction[{i}] must have 'message_source' and 'message_destination'")

                if check_method == "llm":
                    # LLM-based checking requires either custom_prompt or disallow_item
                    if "custom_prompt" not in rule and "disallow_item" not in rule:
                        raise ValueError(
                            f"tool_interaction[{i}] with check_method 'llm' must have either 'custom_prompt' or 'disallow_item'"
                        )
                    if "disallow_item" in rule and not isinstance(rule["disallow_item"], list):
                        raise ValueError(f"tool_interaction[{i}] disallow_item must be a list")

                elif check_method == "regex":
                    # Regex-based checking requires pattern
                    if "pattern" not in rule:
                        raise ValueError(f"tool_interaction[{i}] with check_method 'regex' must have 'pattern'")
                    if not isinstance(rule["pattern"], str):
                        raise ValueError(f"tool_interaction[{i}] pattern must be a string")
                    # Test regex pattern validity
                    try:
                        re.compile(rule["pattern"])
                    except re.error as e:
                        raise ValueError(f"tool_interaction[{i}] invalid regex pattern '{rule['pattern']}': {e}")

        # Validate llm_interaction rules
        if "llm_interaction" in env_rules:
            if not isinstance(env_rules["llm_interaction"], list):
                raise ValueError("llm_interaction must be a list")

            for i, rule in enumerate(env_rules["llm_interaction"]):
                if not isinstance(rule, dict):
                    raise ValueError(f"llm_interaction[{i}] must be a dictionary")

                # Check method validation - no default, must be explicit
                if "check_method" not in rule:
                    raise ValueError(f"llm_interaction[{i}] missing required field: check_method")
                check_method = rule["check_method"]
                if check_method not in ["llm", "regex"]:
                    raise ValueError(
                        f"llm_interaction[{i}] invalid check_method: {check_method}. Must be 'llm' or 'regex'"
                    )

                # Validate action - no default, must be explicit
                if "action" not in rule:
                    raise ValueError(f"llm_interaction[{i}] missing required field: action")
                action = rule["action"]
                if action not in ["block", "mask", "warning"]:
                    raise ValueError(
                        f"llm_interaction[{i}] invalid action: {action}. Must be 'block', 'mask', or 'warning'"
                    )

                # All llm_interaction rules must have message_source and message_destination
                if "message_source" not in rule or "message_destination" not in rule:
                    raise ValueError(f"llm_interaction[{i}] must have 'message_source' and 'message_destination'")

                if check_method == "llm":
                    # LLM-based checking requires either custom_prompt or disallow_item
                    if "custom_prompt" not in rule and "disallow_item" not in rule:
                        raise ValueError(
                            f"llm_interaction[{i}] with check_method 'llm' must have either 'custom_prompt' or 'disallow_item'"
                        )
                    if "disallow_item" in rule and not isinstance(rule["disallow_item"], list):
                        raise ValueError(f"llm_interaction[{i}] disallow_item must be a list")

                elif check_method == "regex":
                    # Regex-based checking requires pattern
                    if "pattern" not in rule:
                        raise ValueError(f"llm_interaction[{i}] with check_method 'regex' must have 'pattern'")
                    if not isinstance(rule["pattern"], str):
                        raise ValueError(f"llm_interaction[{i}] pattern must be a string")
                    # Test regex pattern validity
                    try:
                        re.compile(rule["pattern"])
                    except re.error as e:
                        raise ValueError(f"llm_interaction[{i}] invalid regex pattern '{rule['pattern']}': {e}")

        # Validate user_interaction rules
        if "user_interaction" in env_rules:
            if not isinstance(env_rules["user_interaction"], list):
                raise ValueError("user_interaction must be a list")

            for i, rule in enumerate(env_rules["user_interaction"]):
                if not isinstance(rule, dict):
                    raise ValueError(f"user_interaction[{i}] must be a dictionary")

                # Check method validation - no default, must be explicit
                if "check_method" not in rule:
                    raise ValueError(f"user_interaction[{i}] missing required field: check_method")
                check_method = rule["check_method"]
                if check_method not in ["llm", "regex"]:
                    raise ValueError(
                        f"user_interaction[{i}] invalid check_method: {check_method}. Must be 'llm' or 'regex'"
                    )

                # Validate action - no default, must be explicit
                if "action" not in rule:
                    raise ValueError(f"user_interaction[{i}] missing required field: action")
                action = rule["action"]
                if action not in ["block", "mask", "warning"]:
                    raise ValueError(
                        f"user_interaction[{i}] invalid action: {action}. Must be 'block', 'mask', or 'warning'"
                    )

                # All user_interaction rules must have message_source and message_destination
                if "message_source" not in rule or "message_destination" not in rule:
                    raise ValueError(f"user_interaction[{i}] must have 'message_source' and 'message_destination'")

                if check_method == "llm":
                    # LLM-based checking requires either custom_prompt or disallow_item
                    if "custom_prompt" not in rule and "disallow_item" not in rule:
                        raise ValueError(
                            f"user_interaction[{i}] with check_method 'llm' must have either 'custom_prompt' or 'disallow_item'"
                        )
                    if "disallow_item" in rule and not isinstance(rule["disallow_item"], list):
                        raise ValueError(f"user_interaction[{i}] disallow_item must be a list")

                elif check_method == "regex":
                    # Regex-based checking requires pattern
                    if "pattern" not in rule:
                        raise ValueError(f"user_interaction[{i}] with check_method 'regex' must have 'pattern'")
                    if not isinstance(rule["pattern"], str):
                        raise ValueError(f"user_interaction[{i}] pattern must be a string")
                    # Test regex pattern validity
                    try:
                        re.compile(rule["pattern"])
                    except re.error as e:
                        raise ValueError(f"user_interaction[{i}] invalid regex pattern '{rule['pattern']}': {e}")

    def validate_agent_names(self, agent_names: list[str]) -> None:
        """Validate that agent names referenced in policy actually exist."""
        available_agents = set(agent_names)

        # Check inter-agent safeguards
        if "inter_agent_safeguards" in self.policy:
            inter_agent = self.policy["inter_agent_safeguards"]

            # Check agent_transitions
            for i, rule in enumerate(inter_agent.get("agent_transitions", [])):
                src_agent = rule.get("message_source")
                dst_agent = rule.get("message_destination")

                # Skip wildcard patterns
                if src_agent != "*" and src_agent not in available_agents:
                    raise ValueError(
                        f"agent_transitions[{i}] references unknown source agent: '{src_agent}'. Available agents: {sorted(available_agents)}"
                    )

                if dst_agent != "*" and dst_agent not in available_agents:
                    raise ValueError(
                        f"agent_transitions[{i}] references unknown destination agent: '{dst_agent}'. Available agents: {sorted(available_agents)}"
                    )

        # Check environment safeguards
        if "agent_environment_safeguards" in self.policy:
            env_rules = self.policy["agent_environment_safeguards"]

            # Check tool_interaction rules - only support message_source/message_destination format
            for i, rule in enumerate(env_rules.get("tool_interaction", [])):
                # Only validate message_source/message_destination format
                if "message_source" in rule and "message_destination" in rule:
                    # Skip detailed validation since we can't distinguish agent vs tool names
                    pass
                elif "pattern" in rule and "message_source" not in rule:
                    # Simple pattern rules are allowed
                    pass
                else:
                    raise ValueError(
                        f"tool_interaction[{i}] must use either (message_source, message_destination) or pattern-only format"
                    )

            # Check llm_interaction rules
            for i, rule in enumerate(env_rules.get("llm_interaction", [])):
                # New format
                if "message_source" in rule and "message_destination" in rule:
                    src = rule["message_source"]
                    dst = rule["message_destination"]

                    # Check agent references (LLM interactions have agent <-> llm)
                    if src != "llm" and src.lower() != "llm" and src not in available_agents:
                        raise ValueError(
                            f"llm_interaction[{i}] references unknown agent: '{src}'. Available agents: {sorted(available_agents)}"
                        )
                    if dst != "llm" and dst.lower() != "llm" and dst not in available_agents:
                        raise ValueError(
                            f"llm_interaction[{i}] references unknown agent: '{dst}'. Available agents: {sorted(available_agents)}"
                        )

                elif "agent_name" in rule:
                    agent_name = rule["agent_name"]
                    if agent_name not in available_agents:
                        raise ValueError(
                            f"llm_interaction[{i}] references unknown agent: '{agent_name}'. Available agents: {sorted(available_agents)}"
                        )

            # Check user_interaction rules
            for i, rule in enumerate(env_rules.get("user_interaction", [])):
                agent_name = rule.get("agent")
                if agent_name and agent_name not in available_agents:
                    raise ValueError(
                        f"user_interaction[{i}] references unknown agent: '{agent_name}'. Available agents: {sorted(available_agents)}"
                    )

    def validate_tool_names(self, agent_tool_mapping: dict[str, list[str]], agent_names: list[str]) -> None:
        """Validate that tool names referenced in policy actually exist and belong to the correct agents.

        Args:
            agent_tool_mapping: Dict mapping agent names to their tool names
            agent_names: List of available agent names
        """
        available_agents = set(agent_names)
        # Get all available tools across all agents
        all_available_tools = set()
        for tools in agent_tool_mapping.values():
            all_available_tools.update(tools)

        # Check environment safeguards for tool references
        if "agent_environment_safeguards" in self.policy:
            env_rules = self.policy["agent_environment_safeguards"]

            # Check tool_interaction rules
            for i, rule in enumerate(env_rules.get("tool_interaction", [])):
                # Check message_source/message_destination format
                if "message_source" in rule and "message_destination" in rule:
                    src = rule["message_source"]
                    dst = rule["message_destination"]

                    # Validate agent-tool relationships
                    self._validate_agent_tool_relationship(
                        i, "message_source", src, dst, available_agents, agent_tool_mapping, all_available_tools
                    )

    def _validate_agent_tool_relationship(
        self,
        rule_index: int,
        src_field: str,
        src: str,
        dst: str,
        available_agents: set[str],
        agent_tool_mapping: dict[str, list[str]],
        all_available_tools: set[str],
    ) -> None:
        """Validate that agent-tool relationships in policy are correct."""
        # Skip wildcards and special cases
        if src == "*" or dst == "*" or src.lower() == "user" or dst.lower() == "user":
            return

        # Case 1: Agent -> Tool (agent uses tool)
        if src in available_agents and dst not in available_agents:
            # dst should be a tool that belongs to src agent
            if dst not in all_available_tools:
                raise ValueError(
                    f"tool_interaction[{rule_index}] references unknown tool: '{dst}'. Available tools: {sorted(all_available_tools)}"
                )
            if dst not in agent_tool_mapping.get(src, []):
                agent_tools = agent_tool_mapping.get(src, [])
                raise ValueError(
                    f"tool_interaction[{rule_index}] agent '{src}' does not have access to tool '{dst}'. "
                    f"Agent's tools: {sorted(agent_tools)}"
                )

        # Case 2: Tool -> Agent (tool output to agent)
        elif src not in available_agents and dst in available_agents:
            # src should be a tool that belongs to dst agent
            if src not in all_available_tools:
                raise ValueError(
                    f"tool_interaction[{rule_index}] references unknown tool: '{src}'. Available tools: {sorted(all_available_tools)}"
                )
            if src not in agent_tool_mapping.get(dst, []):
                agent_tools = agent_tool_mapping.get(dst, [])
                raise ValueError(
                    f"tool_interaction[{rule_index}] agent '{dst}' does not have access to tool '{src}'. "
                    f"Agent's tools: {sorted(agent_tools)}"
                )

        # Case 3: Tool -> Tool (unusual, but validate both exist)
        elif src not in available_agents and dst not in available_agents:
            if src not in all_available_tools:
                raise ValueError(
                    f"tool_interaction[{rule_index}] references unknown tool: '{src}'. Available tools: {sorted(all_available_tools)}"
                )
            if dst not in all_available_tools:
                raise ValueError(
                    f"tool_interaction[{rule_index}] references unknown tool: '{dst}'. Available tools: {sorted(all_available_tools)}"
                )
