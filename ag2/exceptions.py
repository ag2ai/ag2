# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock


class AG2Error(Exception):
    """Base exception for all AG2 errors."""


class ToolConflictError(AG2Error):
    def __init__(self, tool_name: str) -> None:
        super().__init__(f"Could not add tool: `{tool_name}`. Tool with such name already registered.")


class ToolResolutionError(AG2Error):
    """Raised when one or more tools in an AgentSpec cannot be resolved from the available tools pool."""

    def __init__(self, missing: list[str], available: list[str]) -> None:
        self.missing = missing
        self.available = available
        super().__init__(f"Could not resolve tool(s): {missing}. Available: {sorted(available)}")


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


class HumanInputError(AG2Error):
    """Base for a failure of the human-input channel itself.

    Distinct from a tool that failed: nobody could be asked, nobody answered in
    time, or the asking blew up — so the turn has no honest answer to continue
    from. Everything that leaves :meth:`ag2.Context.input` unanswered leaves it
    as one of these, and tool execution lets them propagate instead of
    recording them as a tool result. A tool that wants to carry on without an
    answer catches this around ``context.input`` itself, which is the
    difference between choosing to proceed and never being told the question
    went nowhere.
    """


class HumanInputNotProvidedError(HumanInputError):
    """Raised when human-in-the-loop input was requested but not provided."""

    def __init__(self, message: str | None = None) -> None:
        super().__init__(
            message
            or (
                "Human input was requested but not provided. "
                "Please set it for agent using `Agent(..., hitl_hook=func)` or `@agent.hitl_hook`."
            )
        )


class HumanInputFailedError(HumanInputError):
    """Raised when the human-input channel raised instead of answering.

    A ``hitl_hook`` is the channel to the human, so its failure means the
    question was never put — an approval queue that is down denies nothing, it
    answers nothing. The original exception is kept on :attr:`cause` and as
    ``__cause__``, so a host that wants to tell its own failures apart still
    can::

        except HumanInputFailedError as exc:
            if isinstance(exc.cause, QueueUnavailable):
                ...
    """

    def __init__(self, cause: BaseException) -> None:
        super().__init__(f"The human-input channel failed with {type(cause).__name__}: {cause}")
        self.cause = cause

    def __reduce__(self) -> "tuple[type[HumanInputFailedError], tuple[BaseException]]":
        # ``args`` holds the formatted message, so the default reconstruction
        # would feed it back in as the cause — restating the sentence around
        # itself and leaving ``.cause`` a string. This exception travels: it can
        # reach a caller on a ``TaskFailed`` event, and ``.cause`` is the whole
        # reason it is a wrapper rather than a re-type.
        return type(self), (self.cause,)


class HumanInputTimeoutError(HumanInputError):
    """Raised when nobody answered a human-input request in time.

    The same outcome as no hook at all — the turn has no answer — and reported
    the same way, because a question left unanswered for the timeout is not the
    asking tool failing.
    """

    def __init__(self, timeout: float) -> None:
        super().__init__(f"Nobody answered the human-input request within {timeout} seconds.")
        self.timeout = timeout

    def __reduce__(self) -> "tuple[type[HumanInputTimeoutError], tuple[float]]":
        # See :meth:`HumanInputFailedError.__reduce__`: reconstructing from
        # ``args`` would put the message where the number goes.
        return type(self), (self.timeout,)


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


class SkillDownloadError(SkillError):
    """Raised when a skill cannot be downloaded from the remote registry."""


class SkillInstallError(SkillError):
    """Raised when a downloaded skill archive cannot be extracted or validated."""


def missing_additional_dependency(name: str, dependency: str, error: ImportError) -> Mock:
    def _raise(*args: object, **kwargs: object) -> None:
        raise ImportError(
            f'{name} requires optional dependencies. Install with `pip install "{dependency}"`'
        ) from error

    return Mock(side_effect=_raise)


def missing_optional_dependency(name: str, extra: str, error: ImportError) -> Mock:
    return missing_additional_dependency(name, f"ag2[{extra}]", error)
