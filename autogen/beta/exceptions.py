# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0


class AG2Error(Exception):
    """Base exception for all AG2 beta errors."""


class ToolExecutionError(AG2Error):
    """Base exception for tool-related errors."""


class ToolNotFoundError(ToolExecutionError):
    """Raised when a requested tool cannot be found."""

    def __init__(self, name: str):
        super().__init__(f"Tool `{name}` not found")


class UnsupportedToolError(ToolExecutionError):
    """Raised when a tool type is not supported by a provider."""

    def __init__(self, tool_type: str, provider: str):
        super().__init__(f"Unsupported tool type `{tool_type}` for provider `{provider}`")


class UnsupportedInputError(AG2Error):
    """Raised when an input type is not supported by a provider."""

    def __init__(self, input_type: str, provider: str):
        super().__init__(f"Unsupported input type `{input_type}` for provider `{provider}`")


class HumanInputNotProvidedError(AG2Error):
    """Raised when human-in-the-loop input was requested but not provided."""

    def __init__(self, message: str | None = None) -> None:
        super().__init__(
            message
            or (
                "Human input was requested but not provided. "
                "Please set it for agent using `Agent(..., hitl_hook=func)` or `@agent.hitl_hook`."
            )
        )


class ConfigNotProvidedError(AG2Error):
    """Raised when no model configuration is available for an agent request."""

    def __init__(self, message: str | None = None) -> None:
        super().__init__(
            message
            or "No model config provided. Set config on the `Agent(config=...)` creation or pass it to call `ask(config=...)`."
        )


class SkillError(AG2Error):
    """Base exception for local skills loading (agentskills.io convention)."""


class SkillNotFoundError(SkillError, KeyError):
    """Raised when a skill cannot be found in configured paths."""


class InvalidSkillNameError(SkillError, ValueError):
    """Raised when a skill name is empty or malformed."""


class InvalidSkillError(SkillError, ValueError):
    """Raised when skill metadata violates the specification."""
