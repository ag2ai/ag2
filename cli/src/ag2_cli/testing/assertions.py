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
        min_len = int(assertion.value)
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
        max_len = int(assertion.value)
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
        max_t = int(assertion.value)
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

    return AssertionResult(
        passed=False,
        assertion_type=atype,
        message=f"Unknown assertion type: {atype}",
    )
