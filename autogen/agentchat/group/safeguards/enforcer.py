# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from ....code_utils import content_str
from ....io.base import IOStream
from ....llm_config import LLMConfig
from ...conversable_agent import ConversableAgent
from ...groupchat import GroupChatManager
from ..guardrails import LLMGuardrail, RegexGuardrail
from ..targets.transition_target import TransitionTarget
from .events import SafeguardEvent


class SafeguardEnforcer:
    """Main safeguard enforcer - executes safeguard policies"""

    @staticmethod
    def _stringify_content(value: Any) -> str:
        if isinstance(value, (str, list)) or value is None:
            try:
                return content_str(value)
            except (TypeError, ValueError, AssertionError):
                pass
        return "" if value is None else str(value)

    def __init__(
        self,
        policy: dict[str, Any] | str,
        safeguard_llm_config: LLMConfig | dict[str, Any] | None = None,
        mask_llm_config: LLMConfig | dict[str, Any] | None = None,
        groupchat_manager: GroupChatManager | None = None,
        agents: list[ConversableAgent] | None = None,
    ):
        """Initialize the safeguard enforcer.

        Args:
            policy: Safeguard policy dict or path to JSON file
            safeguard_llm_config: LLM configuration for safeguard checks
            mask_llm_config: LLM configuration for masking
            groupchat_manager: GroupChat manager instance for group chat scenarios
            agents: List of conversable agents to apply safeguards to
        """
        self.policy = self._load_policy(policy)
        self.safeguard_llm_config = safeguard_llm_config
        self.mask_llm_config = mask_llm_config
        self.groupchat_manager = groupchat_manager
        self.agents = agents
        self.group_tool_executor = None
        if self.groupchat_manager:
            for agent in self.groupchat_manager.groupchat.agents:
                if agent.name == "_Group_Tool_Executor":
                    self.group_tool_executor = agent  # type: ignore[assignment]
                    break

        # Validate policy format before proceeding
        self._validate_policy()

        # Create mask agent for content masking
        if self.mask_llm_config:
            from ...conversable_agent import ConversableAgent

            self.mask_agent = ConversableAgent(
                name="mask_agent",
                system_message="You are a agent responsible for masking sensitive information.",
                llm_config=self.mask_llm_config,
                human_input_mode="NEVER",
                max_consecutive_auto_reply=1,
            )

        # Parse safeguard rules
        self.inter_agent_rules = self._parse_inter_agent_rules()
        self.environment_rules = self._parse_environment_rules()

        # Send load event
        self._send_safeguard_event(
            event_type="load",
            message=f"Loaded {len(self.inter_agent_rules)} inter-agent and {len(self.environment_rules)} environment safeguard rules",
        )

    def _send_safeguard_event(
        self,
        event_type: str,
        message: str,
        source_agent: str | None = None,
        target_agent: str | None = None,
        guardrail_type: str | None = None,
        action: str | None = None,
        content_preview: str | None = None,
    ) -> None:
        """Send a safeguard event to the IOStream."""
        iostream = IOStream.get_default()
        event = SafeguardEvent(
            event_type=event_type,
            message=message,
            source_agent=source_agent,
            target_agent=target_agent,
            guardrail_type=guardrail_type,
            action=action,
            content_preview=content_preview,
        )
        iostream.send(event)

    def _load_policy(self, policy: dict[str, Any] | str) -> dict[str, Any]:
        """Load policy from file or use provided dict."""
        if isinstance(policy, str):
            with open(policy) as f:
                result: dict[str, Any] = json.load(f)
                return result
        return policy

    def _validate_policy(self) -> None:
        """Validate policy format and syntax."""
        from .validator import SafeguardValidator

        validator = SafeguardValidator(self.policy)
        validator.validate_policy_structure()

    def _parse_inter_agent_rules(self) -> list[dict[str, Any]]:
        """Parse inter-agent safeguard rules from policy."""
        rules = []
        inter_agent = self.policy.get("inter_agent_safeguards", {})

        # Agent transitions
        for rule in inter_agent.get("agent_transitions", []):
            # Create guardrail based on check_method
            check_method = rule.get("check_method", "regex")
            guardrail: LLMGuardrail | RegexGuardrail | None = None
            action = rule.get("violation_response", rule.get("action", "block"))  # Support both field names

            if check_method == "llm":
                if not self.safeguard_llm_config:
                    raise ValueError(
                        f"safeguard_llm_config is required for LLM-based guardrail: {rule['message_source']} -> {rule['message_destination']}"
                    )

                # Handle different LLM check types
                if "custom_prompt" in rule:
                    # Custom prompt for LLM guardrail
                    condition = rule["custom_prompt"]

                elif "disallow_item" in rule:
                    # Disallow items for LLM guardrail
                    condition = f"Check if this content contains any of these disallowed categories: {', '.join(rule['disallow_item'])}"

                else:
                    raise ValueError(
                        f"Either custom_prompt or disallow_item must be provided for LLM guardrail: {rule['message_source']} -> {rule['message_destination']}"
                    )

                # Create LLM guardrail - handle dict config by converting to LLMConfig
                llm_config = self.safeguard_llm_config
                if isinstance(llm_config, dict):
                    llm_config = LLMConfig(config_list=[llm_config])

                guardrail = LLMGuardrail(
                    name=f"llm_guard_{rule['message_source']}_{rule['message_destination']}",
                    condition=condition,
                    target=TransitionTarget(),
                    llm_config=llm_config,
                    activation_message=rule.get("activation_message", "LLM detected violation"),
                )

            elif check_method == "regex":
                if "pattern" in rule:
                    # Regex pattern guardrail
                    guardrail = RegexGuardrail(
                        name=f"regex_guard_{rule['message_source']}_{rule['message_destination']}",
                        condition=rule["pattern"],
                        target=TransitionTarget(),
                        activation_message=rule.get("activation_message", "Regex pattern matched"),
                    )

            # Add rule with guardrail
            parsed_rule = {
                "type": "agent_transition",
                "source": rule["message_source"],
                "target": rule["message_destination"],
                "action": action,
                "guardrail": guardrail,
                "activation_message": rule.get("activation_message", "Content blocked by safeguard"),
            }

            # Keep legacy fields for backward compatibility
            if "disallow_item" in rule:
                parsed_rule["disallow"] = rule["disallow_item"]
            if "pattern" in rule:
                parsed_rule["pattern"] = rule["pattern"]
            if "custom_prompt" in rule:
                parsed_rule["custom_prompt"] = rule["custom_prompt"]

            rules.append(parsed_rule)

        # Groupchat message check
        if "groupchat_message_check" in inter_agent:
            rule = inter_agent["groupchat_message_check"]
            rules.append({
                "type": "groupchat_message",
                "source": "*",
                "target": "*",
                "action": rule.get("pet_action", "block"),
                "disallow": rule.get("disallow_item", []),
            })

        return rules

    def _parse_environment_rules(self) -> list[dict[str, Any]]:
        """Parse agent-environment safeguard rules from policy."""
        rules = []
        env_rules = self.policy.get("agent_environment_safeguards", {})

        # Tool interaction rules
        for rule in env_rules.get("tool_interaction", []):
            check_method = rule.get("check_method", "regex")  # default to regex for backward compatibility
            action = rule.get("violation_response", rule.get("action", "block"))

            if check_method == "llm":
                # LLM-based tool interaction rule - requires message_source/message_destination
                if "message_source" not in rule or "message_destination" not in rule:
                    raise ValueError(
                        "tool_interaction with check_method 'llm' must have 'message_source' and 'message_destination'"
                    )

                parsed_rule = {
                    "type": "tool_interaction",
                    "message_source": rule["message_source"],
                    "message_destination": rule["message_destination"],
                    "check_method": "llm",
                    "action": action,
                    "activation_message": rule.get("activation_message", "LLM blocked tool output"),
                }

                # Add LLM-specific parameters
                if "custom_prompt" in rule:
                    parsed_rule["custom_prompt"] = rule["custom_prompt"]
                elif "disallow_item" in rule:
                    parsed_rule["disallow"] = rule["disallow_item"]

                rules.append(parsed_rule)

            elif check_method == "regex":
                # Regex pattern-based rule - now requires message_source/message_destination
                if "message_source" not in rule or "message_destination" not in rule:
                    raise ValueError(
                        "tool_interaction with check_method 'regex' must have 'message_source' and 'message_destination'"
                    )
                if "pattern" not in rule:
                    raise ValueError("tool_interaction with check_method 'regex' must have 'pattern'")

                rules.append({
                    "type": "tool_interaction",
                    "message_source": rule["message_source"],
                    "message_destination": rule["message_destination"],
                    "check_method": "regex",
                    "pattern": rule["pattern"],
                    "action": action,
                    "activation_message": rule.get("activation_message", "Content blocked by safeguard"),
                })
            else:
                raise ValueError(
                    "tool_interaction rule must have check_method 'llm' or 'regex' with appropriate parameters"
                )

        # LLM interaction rules
        for rule in env_rules.get("llm_interaction", []):
            check_method = rule.get("check_method", "regex")  # default to regex for backward compatibility
            action = rule.get("action", "block")

            # All llm_interaction rules now require message_source/message_destination
            if "message_source" not in rule or "message_destination" not in rule:
                raise ValueError("llm_interaction rule must have 'message_source' and 'message_destination'")

            if check_method == "llm":
                # LLM-based LLM interaction rule
                parsed_rule = {
                    "type": "llm_interaction",
                    "message_source": rule["message_source"],
                    "message_destination": rule["message_destination"],
                    "check_method": "llm",
                    "action": action,
                    "activation_message": rule.get("activation_message", "LLM blocked content"),
                }

                # Add LLM-specific parameters
                if "custom_prompt" in rule:
                    parsed_rule["custom_prompt"] = rule["custom_prompt"]
                elif "disallow_item" in rule:
                    parsed_rule["disallow_item"] = rule["disallow_item"]
                else:
                    raise ValueError(
                        "llm_interaction with check_method 'llm' must have either 'custom_prompt' or 'disallow_item'"
                    )

                rules.append(parsed_rule)

            elif check_method == "regex":
                # Regex-based LLM interaction rule
                if "pattern" not in rule:
                    raise ValueError("llm_interaction with check_method 'regex' must have 'pattern'")

                rules.append({
                    "type": "llm_interaction",
                    "message_source": rule["message_source"],
                    "message_destination": rule["message_destination"],
                    "check_method": "regex",
                    "pattern": rule["pattern"],
                    "action": action,
                    "activation_message": rule.get("activation_message", "Content blocked by safeguard"),
                })
            else:
                raise ValueError(
                    "llm_interaction rule must have check_method 'llm' or 'regex' with appropriate parameters"
                )

        # User interaction rules
        for rule in env_rules.get("user_interaction", []):
            check_method = rule.get("check_method", "llm")  # default to llm for backward compatibility
            action = rule.get("action", "block")

            # All user_interaction rules now require message_source/message_destination
            if "message_source" not in rule or "message_destination" not in rule:
                raise ValueError("user_interaction rule must have 'message_source' and 'message_destination'")

            if check_method == "llm":
                # LLM-based user interaction rule
                parsed_rule = {
                    "type": "user_interaction",
                    "message_source": rule["message_source"],
                    "message_destination": rule["message_destination"],
                    "check_method": "llm",
                    "action": action,
                }

                # Add LLM-specific parameters
                if "custom_prompt" in rule:
                    parsed_rule["custom_prompt"] = rule["custom_prompt"]
                elif "disallow_item" in rule:
                    parsed_rule["disallow_item"] = rule["disallow_item"]
                else:
                    raise ValueError(
                        "user_interaction with check_method 'llm' must have either 'custom_prompt' or 'disallow_item'"
                    )

                rules.append(parsed_rule)

            elif check_method == "regex":
                # Regex-based user interaction rule
                if "pattern" not in rule:
                    raise ValueError("user_interaction with check_method 'regex' must have 'pattern'")

                rules.append({
                    "type": "user_interaction",
                    "message_source": rule["message_source"],
                    "message_destination": rule["message_destination"],
                    "check_method": "regex",
                    "pattern": rule["pattern"],
                    "action": action,
                })
            else:
                raise ValueError(
                    "user_interaction rule must have check_method 'llm' or 'regex' with appropriate parameters"
                )

        return rules

    def create_agent_hooks(self, agent_name: str) -> dict[str, Callable[..., Any]]:
        """Create hook functions for a specific agent, only for rule types that exist."""
        hooks = {}

        # Check if we have any tool interaction rules that apply to this agent
        if agent_name == "_Group_Tool_Executor":
            # group tool executor is running all tools, so we need to check all tool interaction rules
            agent_tool_rules = [rule for rule in self.environment_rules if rule["type"] == "tool_interaction"]
        else:
            agent_tool_rules = [
                rule
                for rule in self.environment_rules
                if rule["type"] == "tool_interaction"
                and (
                    rule.get("message_destination") == agent_name
                    or rule.get("message_source") == agent_name
                    or rule.get("agent_name") == agent_name
                    or "message_destination" not in rule
                )
            ]
        if agent_tool_rules:

            def tool_input_hook(tool_input: dict[str, Any]) -> dict[str, Any] | None:
                result = self._check_tool_interaction(agent_name, tool_input, "input")
                return result if result is not None else tool_input

            def tool_output_hook(tool_input: dict[str, Any]) -> dict[str, Any] | None:
                result = self._check_tool_interaction(agent_name, tool_input, "output")
                return result if result is not None else tool_input

            hooks["safeguard_tool_inputs"] = tool_input_hook
            hooks["safeguard_tool_outputs"] = tool_output_hook

        # Check if we have any LLM interaction rules that apply to this agent
        agent_llm_rules = [
            rule
            for rule in self.environment_rules
            if rule["type"] == "llm_interaction"
            and (
                rule.get("message_destination") == agent_name
                or rule.get("message_source") == agent_name
                or rule.get("agent_name") == agent_name
                or "message_destination" not in rule
            )
        ]  # Simple pattern rules apply to all

        if agent_llm_rules:

            def llm_input_hook(tool_input: dict[str, Any]) -> dict[str, Any] | None:
                # Extract messages from the data structure if needed
                messages = tool_input if isinstance(tool_input, list) else tool_input.get("messages", tool_input)
                result = self._check_llm_interaction(agent_name, messages, "input")
                if isinstance(result, list) and isinstance(tool_input, dict) and "messages" in tool_input:
                    return {**tool_input, "messages": result}
                elif isinstance(result, dict):
                    return result
                elif result is not None and not isinstance(result, dict):
                    # Convert string or other types to dict format
                    return {"content": str(result), "role": "function"}
                elif result is not None and isinstance(result, dict) and result != tool_input:
                    # Return the modified dict result
                    return result
                return tool_input

            def llm_output_hook(tool_input: dict[str, Any]) -> dict[str, Any] | None:
                result = self._check_llm_interaction(agent_name, tool_input, "output")
                if isinstance(result, dict):
                    return result
                elif result is not None and not isinstance(result, dict):
                    # Convert string or other types to dict format
                    return {"content": str(result), "role": "function"}
                elif result is not None and isinstance(result, dict) and result != tool_input:
                    # Return the modified dict result
                    return result
                return tool_input

            hooks["safeguard_llm_inputs"] = llm_input_hook
            hooks["safeguard_llm_outputs"] = llm_output_hook

        # Check if we have any user interaction rules that apply to this agent
        agent_user_rules = [
            rule
            for rule in self.environment_rules
            if rule["type"] == "user_interaction" and rule.get("message_destination") == agent_name
        ]

        if agent_user_rules:

            def human_input_hook(tool_input: dict[str, Any]) -> dict[str, Any] | None:
                # Extract human input from data structure
                human_input = tool_input.get("content", str(tool_input))
                result = self._check_user_interaction(agent_name, human_input)
                if result != human_input and isinstance(tool_input, dict):
                    return {**tool_input, "content": result}
                return tool_input if result == human_input else {"content": result}

            hooks["safeguard_human_inputs"] = human_input_hook

        # Check if we have any inter-agent rules that apply to this agent
        # Note: For group chats, inter-agent communication is handled by GroupChat._run_inter_agent_guardrails()
        # But for direct agent-to-agent communication, we need the process_message_before_send hook
        agent_inter_rules = [
            rule
            for rule in self.inter_agent_rules
            if (
                rule.get("source") == agent_name
                or rule.get("target") == agent_name
                or rule.get("source") == "*"
                or rule.get("target") == "*"
            )
        ]

        if agent_inter_rules:

            def message_before_send_hook(
                sender: Any, message: dict[str, Any] | str, recipient: Any, silent: Any = None
            ) -> dict[str, Any] | str:
                _ = silent  # Unused parameter
                result = self._check_inter_agent_communication(sender.name, recipient.name, message)
                return result

            hooks["process_message_before_send"] = message_before_send_hook  # type: ignore[assignment]

        return hooks

    def _check_llm_violation(
        self, content: str, disallow_items: list[str] | None = None, custom_prompt: str | None = None
    ) -> tuple[bool, str]:
        """Check if content violates LLM-based safeguard conditions."""
        if not self.safeguard_llm_config:
            raise ValueError("safeguard_llm_config is required for LLM-based safeguard checks")

        # Determine condition based on available parameters
        if custom_prompt:
            condition = custom_prompt
        elif disallow_items:
            condition = (
                f"Check if this content contains any of these disallowed categories: {', '.join(disallow_items)}"
            )
        else:
            raise ValueError("Either custom_prompt or disallow_items must be provided")

        # Create LLM guardrail for checking
        # Handle dict config by converting to LLMConfig
        llm_config = self.safeguard_llm_config
        if isinstance(llm_config, dict):
            llm_config = LLMConfig(config_list=[llm_config])

        from ..targets.transition_target import TransitionTarget

        guardrail = LLMGuardrail(
            name="temp_safeguard_check",
            condition=condition,
            target=TransitionTarget(),
            llm_config=llm_config,
            activation_message="Content violates safeguard conditions",
        )

        try:
            result = guardrail.check(content)
            return result.activated, result.justification
        except Exception as e:
            raise RuntimeError(f"Safeguard check failed: {e}")

    def _check_regex_violation(self, content: str, pattern: str) -> tuple[bool, str]:
        """Check if content matches a regex pattern."""
        try:
            if re.search(pattern, content, re.IGNORECASE):
                return True, f"Content matched pattern: {pattern}"
        except re.error as e:
            raise ValueError(f"Invalid regex pattern '{pattern}': {e}")

        return False, "No pattern match"

    def _apply_action(
        self,
        action: str,
        content: str | dict[str, Any] | list[Any],
        disallow_items: list[str],
        explanation: str,
        custom_message: str | None = None,
        pattern: str | None = None,
        guardrail_type: str | None = None,
        source_agent: str | None = None,
        target_agent: str | None = None,
        content_preview: str | None = None,
    ) -> str | dict[str, Any] | list[Any]:
        """Apply the specified action to content."""
        message = custom_message or explanation

        if action == "block":
            self._send_safeguard_event(
                event_type="action",
                message=f"BLOCKED: {message}",
                action="block",
                source_agent=source_agent,
                target_agent=target_agent,
                content_preview=content_preview,
            )
            return self._handle_blocked_content(content, message)
        elif action == "mask":
            self._send_safeguard_event(
                event_type="action",
                message=f"MASKED: {message}",
                action="mask",
                source_agent=source_agent,
                target_agent=target_agent,
                content_preview=content_preview,
            )

            def mask_func(text: str) -> str:
                return self._mask_content(text, disallow_items, explanation, pattern)

            return self._handle_masked_content(content, mask_func)
        elif action == "warning":
            self._send_safeguard_event(
                event_type="action",
                message=f"WARNING: {message}",
                action="warning",
                source_agent=source_agent,
                target_agent=target_agent,
                content_preview=content_preview,
            )
            return content
        else:
            return content

    def _mask_content(
        self, content: str, disallow_items: list[str], explanation: str, pattern: str | None = None
    ) -> str:
        """Mask sensitive content using LLM, pattern-based, or simple replacement."""
        # If we have a specific pattern from a regex guardrail, use it first
        if pattern:
            try:
                masked = re.sub(pattern, "[SENSITIVE_INFO]", content, flags=re.IGNORECASE)
                if masked != content:  # Only return if something was actually masked
                    return masked
            except re.error as e:
                raise ValueError(f"Pattern masking failed: {e}")

        # Try LLM-based masking if available
        if self.mask_agent and disallow_items:
            mask_prompt = f"""
            Mask the sensitive information in this content with [SENSITIVE_INFO]:

            Content: {content}
            Sensitive categories: {", ".join(disallow_items)}
            Reason: {explanation}

            Return only the masked content, nothing else.
            """

            try:
                response = self.mask_agent.generate_oai_reply(messages=[{"role": "user", "content": mask_prompt}])

                if response[0] and response[1]:
                    masked = response[1].get("content", content) if isinstance(response[1], dict) else str(response[1])
                    return masked
            except Exception as e:
                raise ValueError(f"LLM masking failed: {e}")

        return masked

    def _handle_blocked_content(
        self, content: str | dict[str, Any] | list[Any], block_message: str
    ) -> str | dict[str, Any] | list[Any]:
        """Handle blocked content based on its structure."""
        block_msg = f"🛡️ BLOCKED: {block_message}"

        if isinstance(content, dict):
            blocked_content = content.copy()

            # Handle tool_responses (like in tool outputs)
            if "tool_responses" in blocked_content and blocked_content["tool_responses"]:
                blocked_content["content"] = block_msg
                blocked_content["tool_responses"] = [
                    {**response, "content": block_msg} for response in blocked_content["tool_responses"]
                ]
            # Handle tool_calls (like in tool inputs)
            elif "tool_calls" in blocked_content and blocked_content["tool_calls"]:
                blocked_content["tool_calls"] = [
                    {**tool_call, "function": {**tool_call["function"], "arguments": json.dumps({"error": block_msg})}}
                    for tool_call in blocked_content["tool_calls"]
                ]
            # Handle regular content
            elif "content" in blocked_content:
                blocked_content["content"] = block_msg
            # Handle arguments (for some tool formats)
            elif "arguments" in blocked_content:
                blocked_content["arguments"] = block_msg
            else:
                # Default case - add content field
                blocked_content["content"] = block_msg

            return blocked_content

        elif isinstance(content, list):
            # Handle list of messages (like LLM inputs)
            blocked_list = []
            for item in content:
                if isinstance(item, dict):
                    blocked_item = item.copy()
                    if "content" in blocked_item:
                        blocked_item["content"] = block_msg
                    if "tool_calls" in blocked_item:
                        blocked_item["tool_calls"] = [
                            {
                                **tool_call,
                                "function": {**tool_call["function"], "arguments": json.dumps({"error": block_msg})},
                            }
                            for tool_call in blocked_item["tool_calls"]
                        ]
                    if "tool_responses" in blocked_item:
                        blocked_item["tool_responses"] = [
                            {**response, "content": block_msg} for response in blocked_item["tool_responses"]
                        ]
                    blocked_list.append(blocked_item)
                else:
                    blocked_list.append({"content": block_msg, "role": "function"})
            return blocked_list

        else:
            # String or other content - return as function message
            return {"content": block_msg, "role": "function"}

    def _handle_masked_content(
        self, content: str | dict[str, Any] | list[Any], mask_func: Callable[[str], str]
    ) -> str | dict[str, Any] | list[Any]:
        """Handle masked content based on its structure."""
        if isinstance(content, dict):
            masked_content = content.copy()

            # Handle tool_responses
            if "tool_responses" in masked_content and masked_content["tool_responses"]:
                if "content" in masked_content:
                    masked_content["content"] = mask_func(self._stringify_content(masked_content.get("content")))
                masked_content["tool_responses"] = [
                    {
                        **response,
                        "content": mask_func(self._stringify_content(response.get("content"))),
                    }
                    for response in masked_content["tool_responses"]
                ]
            # Handle tool_calls
            elif "tool_calls" in masked_content and masked_content["tool_calls"]:
                masked_content["tool_calls"] = [
                    {
                        **tool_call,
                        "function": {
                            **tool_call["function"],
                            "arguments": mask_func(self._stringify_content(tool_call["function"].get("arguments"))),
                        },
                    }
                    for tool_call in masked_content["tool_calls"]
                ]
            # Handle regular content
            elif "content" in masked_content:
                masked_content["content"] = mask_func(self._stringify_content(masked_content.get("content")))
            # Handle arguments
            elif "arguments" in masked_content:
                masked_content["arguments"] = mask_func(self._stringify_content(masked_content.get("arguments")))

            return masked_content

        elif isinstance(content, list):
            # Handle list of messages
            masked_list = []
            for item in content:
                if isinstance(item, dict):
                    masked_item = item.copy()
                    if "content" in masked_item:
                        masked_item["content"] = mask_func(self._stringify_content(masked_item.get("content")))
                    if "tool_calls" in masked_item:
                        masked_item["tool_calls"] = [
                            {
                                **tool_call,
                                "function": {
                                    **tool_call["function"],
                                    "arguments": mask_func(
                                        self._stringify_content(tool_call["function"].get("arguments"))
                                    ),
                                },
                            }
                            for tool_call in masked_item["tool_calls"]
                        ]
                    if "tool_responses" in masked_item:
                        masked_item["tool_responses"] = [
                            {
                                **response,
                                "content": mask_func(self._stringify_content(response.get("content"))),
                            }
                            for response in masked_item["tool_responses"]
                        ]
                    masked_list.append(masked_item)
                else:
                    # For non-dict items, wrap the masked content in a dict
                    masked_item_content: str = mask_func(self._stringify_content(item))
                    masked_list.append({"content": masked_item_content, "role": "function"})
            return masked_list

        else:
            # String content
            return mask_func(self._stringify_content(content))

    def _check_inter_agent_communication(
        self, sender_name: str, recipient_name: str, message: str | dict[str, Any]
    ) -> str | dict[str, Any]:
        """Check inter-agent communication."""
        if isinstance(message, dict):
            if "tool_calls" in message and isinstance(message["tool_calls"], list):
                # Extract arguments from all tool calls and combine them
                tool_args = []
                for tool_call in message["tool_calls"]:
                    if "function" in tool_call and "arguments" in tool_call["function"]:
                        tool_args.append(tool_call["function"]["arguments"])
                content_to_check = " | ".join(tool_args) if tool_args else ""
            elif "tool_responses" in message and isinstance(message["tool_responses"], list):
                # Extract content from all tool responses and combine them
                tool_contents = []
                for tool_response in message["tool_responses"]:
                    if "content" in tool_response:
                        tool_contents.append(str(tool_response["content"]))
                content_to_check = " | ".join(tool_contents) if tool_contents else ""
            else:
                content_to_check = str(message.get("content", ""))
        elif isinstance(message, str):
            content_to_check = message
        else:
            raise ValueError("Message must be a dictionary or a string")

        for rule in self.inter_agent_rules:
            if rule["type"] == "agent_transition":
                # Check if this rule applies
                source_match = rule["source"] == "*" or rule["source"] == sender_name
                target_match = rule["target"] == "*" or rule["target"] == recipient_name

                if source_match and target_match:
                    # Prepare content preview
                    content_preview = str(content_to_check)[:100] + ("..." if len(str(content_to_check)) > 100 else "")

                    # Use guardrail if available
                    if "guardrail" in rule and rule["guardrail"]:
                        # Send single check event with guardrail info
                        self._send_safeguard_event(
                            event_type="check",
                            message="Checking inter-agent communication",
                            source_agent=sender_name,
                            target_agent=recipient_name,
                            guardrail_type=type(rule["guardrail"]).__name__,
                            # action=rule.get('action', 'N/A'),
                            content_preview=content_preview,
                        )

                        try:
                            result = rule["guardrail"].check(content_to_check)
                            if result.activated:
                                self._send_safeguard_event(
                                    event_type="violation",
                                    message=f"VIOLATION DETECTED: {result.justification}",
                                    source_agent=sender_name,
                                    target_agent=recipient_name,
                                    guardrail_type=type(rule["guardrail"]).__name__,
                                    content_preview=content_preview,
                                )
                                # Pass the pattern if it's a regex guardrail
                                pattern = rule.get("pattern") if isinstance(rule["guardrail"], RegexGuardrail) else None
                                action_result = self._apply_action(
                                    action=rule["action"],
                                    content=message,
                                    disallow_items=[],
                                    explanation=result.justification,
                                    custom_message=rule.get("activation_message", result.justification),
                                    pattern=pattern,
                                    guardrail_type=type(rule["guardrail"]).__name__,
                                    source_agent=sender_name,
                                    target_agent=recipient_name,
                                    content_preview=content_preview,
                                )
                                if isinstance(action_result, (str, dict)):
                                    return action_result
                                else:
                                    return message
                            else:
                                # Content passed - no additional event needed, already sent check event above
                                pass
                        except Exception as e:
                            raise ValueError(f"Guardrail check failed: {e}")

                    # Handle legacy pattern-based rules
                    elif "pattern" in rule and rule["pattern"]:
                        # Send single check event for pattern-based rules
                        self._send_safeguard_event(
                            event_type="check",
                            message="Checking inter-agent communication",
                            source_agent=sender_name,
                            target_agent=recipient_name,
                            guardrail_type="RegexGuardrail",
                            # action=rule.get('action', 'N/A'),
                            content_preview=content_preview,
                        )
                        is_violation, explanation = self._check_regex_violation(content_to_check, rule["pattern"])
                        if is_violation:
                            result_value = self._apply_action(
                                action=rule["action"],
                                content=message,
                                disallow_items=[],
                                explanation=explanation,
                                custom_message=rule.get("activation_message"),
                                pattern=rule["pattern"],
                                guardrail_type="RegexGuardrail",
                                source_agent=sender_name,
                                target_agent=recipient_name,
                                content_preview=content_preview,
                            )
                            if isinstance(result_value, (str, dict)):
                                return result_value
                            else:
                                return message
                        else:
                            pass

                    # Handle legacy disallow-based rules and custom prompts
                    elif "disallow" in rule or "custom_prompt" in rule:
                        # Send single check event for LLM-based legacy rules
                        self._send_safeguard_event(
                            event_type="check",
                            message="Checking inter-agent communication",
                            source_agent=sender_name,
                            target_agent=recipient_name,
                            guardrail_type="LLMGuardrail",
                            # action=rule.get('action', 'N/A'),
                            content_preview=content_preview,
                        )
                        if "custom_prompt" in rule:
                            is_violation, explanation = self._check_llm_violation(
                                content_to_check, custom_prompt=rule["custom_prompt"]
                            )
                        else:
                            is_violation, explanation = self._check_llm_violation(
                                content_to_check, disallow_items=rule["disallow"]
                            )

                        if is_violation:
                            result_value = self._apply_action(
                                action=rule["action"],
                                content=message,
                                disallow_items=rule.get("disallow", []),
                                explanation=explanation,
                                custom_message=None,
                                pattern=None,
                                guardrail_type="LLMGuardrail",
                                source_agent=sender_name,
                                target_agent=recipient_name,
                                content_preview=content_preview,
                            )
                            if isinstance(result_value, (str, dict)):
                                return result_value
                            else:
                                return message
                        else:
                            pass

        return message

    def _check_interaction(
        self,
        interaction_type: str,
        source_name: str,
        dest_name: str,
        content: str,
        data: str | dict[str, Any] | list[dict[str, Any]],
        context_info: str,
    ) -> str | dict[str, Any] | list[dict[str, Any]] | None:
        """Unified method to check any type of interaction."""
        for rule in self.environment_rules:
            if (
                rule["type"] == interaction_type
                and "message_source" in rule
                and "message_destination" in rule
                and rule["message_source"] == source_name
                and rule["message_destination"] == dest_name
            ):
                content_preview = content[:100] + ("..." if len(content) > 100 else "")
                check_method = rule.get("check_method", "regex")
                guardrail_type = "LLMGuardrail" if check_method == "llm" else "RegexGuardrail"

                # Send check event
                self._send_safeguard_event(
                    event_type="check",
                    message=f"Checking {interaction_type.replace('_', ' ')}: {context_info}",
                    source_agent=source_name,
                    target_agent=dest_name,
                    guardrail_type=guardrail_type,
                    content_preview=content_preview,
                )

                # Perform check based on method
                is_violation, explanation = self._perform_check(rule, content, check_method)

                if is_violation:
                    # Send violation event
                    self._send_safeguard_event(
                        event_type="violation",
                        message=f"{guardrail_type.replace('Guardrail', '').upper()} VIOLATION: {explanation}",
                        source_agent=source_name,
                        target_agent=dest_name,
                        guardrail_type=guardrail_type,
                        content_preview=content_preview,
                    )

                    # Apply action
                    result = self._apply_action(
                        action=rule["action"],
                        content=data,
                        disallow_items=rule.get("disallow_item", []),
                        explanation=explanation,
                        custom_message=rule.get("activation_message"),
                        pattern=rule.get("pattern"),
                        guardrail_type=guardrail_type,
                        source_agent=source_name,
                        target_agent=dest_name,
                        content_preview=content_preview,
                    )
                    return result

        return None

    def _perform_check(self, rule: dict[str, Any], content: str, check_method: str) -> tuple[bool, str]:
        """Perform the actual check based on the method."""
        if check_method == "llm":
            if not self.safeguard_llm_config:
                raise ValueError(
                    f"safeguard_llm_config is required for LLM-based {rule['type']} rule: {rule['message_source']} -> {rule['message_destination']}"
                )

            if "custom_prompt" in rule:
                return self._check_llm_violation(content, custom_prompt=rule["custom_prompt"])
            elif "disallow_item" in rule:
                return self._check_llm_violation(content, disallow_items=rule["disallow_item"])
            else:
                raise ValueError(
                    f"Either custom_prompt or disallow_item must be provided for LLM-based {rule['type']}: {rule['message_source']} -> {rule['message_destination']}"
                )

        elif check_method == "regex":
            if "pattern" not in rule:
                raise ValueError(
                    f"pattern is required for regex-based {rule['type']}: {rule['message_source']} -> {rule['message_destination']}"
                )
            return self._check_regex_violation(content, rule["pattern"])

        else:
            raise ValueError(f"Unsupported check_method: {check_method}")

    def _check_tool_interaction(self, agent_name: str, data: dict[str, Any], direction: str) -> dict[str, Any]:
        """Check tool interactions."""
        # Extract tool name from data
        tool_name = data.get("name", data.get("tool_name", ""))

        # Resolve the actual agent name if this is GroupToolExecutor
        actual_agent_name = agent_name
        if agent_name == "_Group_Tool_Executor" and self.group_tool_executor:
            # Get the original tool caller from GroupToolExecutor
            originator = self.group_tool_executor.get_tool_call_originator()  # type: ignore[attr-defined]
            if originator:
                actual_agent_name = originator

        # Determine source/destination based on direction
        if direction == "output":
            source_name, dest_name = tool_name, actual_agent_name
            content = str(data.get("content", ""))
        else:  # input
            source_name, dest_name = actual_agent_name, tool_name
            content = str(data.get("arguments", ""))

        result = self._check_interaction(
            interaction_type="tool_interaction",
            source_name=source_name,
            dest_name=dest_name,
            content=content,
            data=data,
            context_info=f"{actual_agent_name} <-> {tool_name} ({direction})",
        )

        if result is not None:
            if isinstance(result, dict):
                return result
            else:
                # Convert string or list result back to dict format
                return {"content": str(result), "name": tool_name}
        return data

    def _check_llm_interaction(
        self, agent_name: str, data: str | dict[str, Any] | list[dict[str, Any]], direction: str
    ) -> str | dict[str, Any] | list[dict[str, Any]]:
        """Check LLM interactions."""
        content = str(data)

        # Determine source/destination based on direction
        if direction == "input":
            source_name, dest_name = agent_name, "llm"
        else:  # output
            source_name, dest_name = "llm", agent_name

        result = self._check_interaction(
            interaction_type="llm_interaction",
            source_name=source_name,
            dest_name=dest_name,
            content=content,
            data=data,
            context_info=f"{agent_name} <-> llm ({direction})",
        )

        return result if result is not None else data

    def _check_user_interaction(self, agent_name: str, user_input: str) -> str | None:
        """Check user interactions."""
        result = self._check_interaction(
            interaction_type="user_interaction",
            source_name="user",
            dest_name=agent_name,
            content=user_input,
            data=user_input,
            context_info=f"user <-> {agent_name}",
        )

        if result is not None and isinstance(result, str):
            return result
        return user_input

    def check_and_act(
        self, src_agent_name: str, dst_agent_name: str, message_content: str | dict[str, Any]
    ) -> str | dict[str, Any] | None:
        """Check and act on inter-agent communication for GroupChat integration.

        This method is called by GroupChat._run_inter_agent_guardrails to check
        messages between agents and potentially modify or block them.

        Args:
            src_agent_name: Name of the source agent
            dst_agent_name: Name of the destination agent
            message_content: The message content to check

        Returns:
            Optional replacement message if a safeguard triggers, None otherwise
        """
        # Handle GroupToolExecutor transparency for safeguards
        if src_agent_name == "_Group_Tool_Executor":
            actual_src_agent_name = self._resolve_tool_executor_source(src_agent_name, self.group_tool_executor)
        else:
            actual_src_agent_name = src_agent_name

        # Store original message for comparison
        original_message = message_content

        result = self._check_inter_agent_communication(actual_src_agent_name, dst_agent_name, message_content)

        # Check if the result is different from the original
        if result != original_message:
            return result

        return None

    def _resolve_tool_executor_source(self, src_agent_name: str, tool_executor: Any = None) -> str:
        """Resolve the actual source agent when GroupToolExecutor is involved.

        When src_agent_name is "_Group_Tool_Executor", get the original agent who called the tool.

        Args:
            src_agent_name: The source agent name from the communication
            tool_executor: GroupToolExecutor instance for getting originator

        Returns:
            The actual source agent name (original tool caller for tool responses)
        """
        if src_agent_name != "_Group_Tool_Executor":
            return src_agent_name

        # Handle GroupToolExecutor - get the original tool caller
        if tool_executor and hasattr(tool_executor, "get_tool_call_originator"):
            originator = tool_executor.get_tool_call_originator()
            if originator:
                return originator  # type: ignore[no-any-return]

        # Fallback: Could not determine original caller
        return "tool_executor"
