"""Assertion evaluation for agent test cases."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .cases import EvalAssertion


@dataclass
class AssertionResult:
    """Result of evaluating a single assertion."""

    passed: bool
    assertion_type: str
    message: str
    expected: Any = None
    actual: Any = None


def check_assertion(
    assertion: EvalAssertion,
    output: str,
    turns: int = 0,
    errors: list[str] | None = None,
) -> AssertionResult:
    """Evaluate a single assertion against agent output.

    Args:
        assertion: The assertion to check.
        output: The agent's final output text.
        turns: Number of conversation turns.
        errors: List of errors that occurred during execution.
    """
    errors = errors or []
    atype = assertion.type

    if atype == "contains":
        val = str(assertion.value)
        passed = val in output
        return AssertionResult(
            passed=passed,
            assertion_type=atype,
            message=f"Output {'contains' if passed else 'does not contain'} '{val}'",
            expected=val,
            actual=output[:200] if not passed else None,
        )

    if atype == "contains_all":
        vals = assertion.values or []
        missing = [v for v in vals if str(v) not in output]
        passed = len(missing) == 0
        return AssertionResult(
            passed=passed,
            assertion_type=atype,
            message=f"Missing: {missing}" if not passed else "All substrings found",
            expected=vals,
            actual=missing if not passed else None,
        )

    if atype == "contains_any":
        vals = assertion.values or []
        found = [v for v in vals if str(v) in output]
        passed = len(found) > 0
        return AssertionResult(
            passed=passed,
            assertion_type=atype,
            message=f"Found: {found}" if passed else "None of the expected substrings found",
            expected=vals,
            actual=None,
        )

    if atype == "not_contains":
        val = str(assertion.value)
        passed = val not in output
        return AssertionResult(
            passed=passed,
            assertion_type=atype,
            message=f"Output {'does not contain' if passed else 'contains'} '{val}'",
            expected=f"not '{val}'",
            actual=None,
        )

    if atype == "regex":
        pattern = assertion.pattern or str(assertion.value)
        match = re.search(pattern, output)
        passed = match is not None
        return AssertionResult(
            passed=passed,
            assertion_type=atype,
            message=f"Pattern {'matched' if passed else 'not matched'}: {pattern}",
            expected=pattern,
            actual=match.group() if passed else None,
        )

    if atype == "min_length":
        try:
            min_len = int(assertion.value)
        except (TypeError, ValueError):
            return AssertionResult(
                passed=False,
                assertion_type=atype,
                message=f"Invalid min_length value (expected integer, got {assertion.value!r})",
            )
        actual_len = len(output)
        passed = actual_len >= min_len
        return AssertionResult(
            passed=passed,
            assertion_type=atype,
            message=f"Length {actual_len} {'>=':s} {min_len}" if passed else f"Length {actual_len} < {min_len}",
            expected=min_len,
            actual=actual_len,
        )

    if atype == "max_length":
        try:
            max_len = int(assertion.value)
        except (TypeError, ValueError):
            return AssertionResult(
                passed=False,
                assertion_type=atype,
                message=f"Invalid max_length value (expected integer, got {assertion.value!r})",
            )
        actual_len = len(output)
        passed = actual_len <= max_len
        return AssertionResult(
            passed=passed,
            assertion_type=atype,
            message=f"Length {actual_len} <= {max_len}" if passed else f"Length {actual_len} > {max_len}",
            expected=max_len,
            actual=actual_len,
        )

    if atype == "max_turns":
        try:
            max_t = int(assertion.value)
        except (TypeError, ValueError):
            return AssertionResult(
                passed=False,
                assertion_type=atype,
                message=f"Invalid max_turns value (expected integer, got {assertion.value!r})",
            )
        passed = turns <= max_t
        return AssertionResult(
            passed=passed,
            assertion_type=atype,
            message=f"Turns {turns} <= {max_t}" if passed else f"Turns {turns} > {max_t}",
            expected=max_t,
            actual=turns,
        )

    if atype == "no_error":
        passed = len(errors) == 0
        return AssertionResult(
            passed=passed,
            assertion_type=atype,
            message="No errors" if passed else f"{len(errors)} error(s): {errors[0]}",
            expected="no errors",
            actual=errors if not passed else None,
        )

    if atype == "llm_judge":
        return _check_llm_judge(assertion, output)

    return AssertionResult(
        passed=False,
        assertion_type=atype,
        message=f"Unknown assertion type: {atype}",
    )


def _check_llm_judge(assertion: EvalAssertion, output: str) -> AssertionResult:
    """Use an LLM to judge whether the output meets the given criteria.

    Requires ag2 to be installed. Uses the model specified in the assertion
    (defaults to gpt-4o). The LLM is asked to evaluate the output against
    the criteria and respond with PASS or FAIL plus a brief explanation.

    YAML usage:
        assertions:
          - type: llm_judge
            criteria: "Response correctly explains photosynthesis"
            model: gpt-4o  # optional, defaults to gpt-4o
    """
    criteria = assertion.criteria
    if not criteria:
        return AssertionResult(
            passed=False,
            assertion_type="llm_judge",
            message="No 'criteria' provided for llm_judge assertion",
        )

    model = getattr(assertion, "model", None) or "gpt-4o"

    try:
        import autogen
        from autogen.io.base import IOStream
    except ImportError:
        return AssertionResult(
            passed=False,
            assertion_type="llm_judge",
            message="ag2 is required for llm_judge assertions (pip install ag2)",
        )

    judge_prompt = (
        "You are an evaluation judge. Given the following agent output and "
        "evaluation criteria, determine if the output passes.\n\n"
        f"## Agent Output\n{output}\n\n"
        f"## Criteria\n{criteria}\n\n"
        "Respond with ONLY one of these formats:\n"
        "PASS: <brief explanation>\n"
        "FAIL: <brief explanation>"
    )

    try:
        llm_config = autogen.LLMConfig({"model": model})
        judge = autogen.AssistantAgent(
            name="judge",
            system_message="You are a strict evaluation judge. Always respond with PASS or FAIL followed by a brief explanation.",
            llm_config=llm_config,
        )
        user = autogen.UserProxyAgent(
            name="eval_runner",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=0,
            code_execution_config=False,
        )

        # Suppress AG2's default print output during judge execution
        class _SilentIO:
            def print(self, *a: Any, **kw: Any) -> None:
                pass

            def send(self, message: Any) -> None:
                pass

            def input(self, prompt: str = "", *, password: bool = False) -> str:
                return ""

        with IOStream.set_default(_SilentIO()):  # type: ignore[arg-type]
            chat_result = user.initiate_chat(judge, message=judge_prompt)

        last_msg = ""
        if chat_result.chat_history:
            agent_msgs = [m for m in chat_result.chat_history if m.get("role") != "user" and m.get("content")]
            if agent_msgs:
                last_msg = agent_msgs[-1]["content"]

        passed = last_msg.strip().upper().startswith("PASS")
        return AssertionResult(
            passed=passed,
            assertion_type="llm_judge",
            message=last_msg.strip()[:200],
            expected=criteria,
            actual=output[:200],
        )
    except Exception as exc:
        return AssertionResult(
            passed=False,
            assertion_type="llm_judge",
            message=f"LLM judge failed: {exc}",
            expected=criteria,
            actual=output[:200],
        )
