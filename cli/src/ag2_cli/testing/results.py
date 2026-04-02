"""Shared result types for eval/arena commands."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .assertions import AssertionResult
from .cases import EvalCase


@dataclass
class CaseResult:
    """Result of running a single eval case."""

    case: EvalCase
    assertion_results: list[AssertionResult] = field(default_factory=list)
    output: str = ""
    turns: int = 0
    errors: list[str] = field(default_factory=list)
    elapsed: float = 0.0
    cost: Any = None

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.assertion_results)

    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.assertion_results if r.passed)

    @property
    def total_count(self) -> int:
        return len(self.assertion_results)

    @property
    def score(self) -> float:
        if not self.assertion_results:
            return 0.0
        return self.passed_count / self.total_count
