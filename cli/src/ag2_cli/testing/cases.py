"""Eval case parsing from YAML files."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EvalAssertion:
    """A single assertion to check against agent output."""

    type: str
    value: Any = None
    values: list[Any] | None = None
    pattern: str | None = None
    criteria: str | None = None
    threshold: float | None = None


@dataclass
class EvalCase:
    """A single evaluation test case."""

    name: str
    input: str
    assertions: list[EvalAssertion] = field(default_factory=list)


@dataclass
class EvalSuite:
    """A collection of eval cases."""

    name: str
    description: str
    cases: list[EvalCase] = field(default_factory=list)


def _parse_assertion(raw: dict[str, Any]) -> EvalAssertion:
    """Parse a single assertion from YAML dict."""
    return EvalAssertion(
        type=raw["type"],
        value=raw.get("value"),
        values=raw.get("values"),
        pattern=raw.get("pattern"),
        criteria=raw.get("criteria"),
        threshold=raw.get("threshold"),
    )


def _parse_case(raw: dict[str, Any]) -> EvalCase:
    """Parse a single eval case from YAML dict."""
    assertions = [_parse_assertion(a) for a in raw.get("assertions", [])]
    return EvalCase(
        name=raw["name"],
        input=raw["input"],
        assertions=assertions,
    )


def load_eval_suite(path: Path) -> EvalSuite:
    """Load an evaluation suite from a YAML file.

    Expected format:
        name: "suite-name"
        description: "Description"
        cases:
          - name: "case_1"
            input: "..."
            assertions:
              - type: contains
                value: "expected"
    """
    import yaml

    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Eval file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Eval file must be a YAML mapping, got {type(raw).__name__}")

    cases = [_parse_case(c) for c in raw.get("cases", [])]
    return EvalSuite(
        name=raw.get("name", path.stem),
        description=raw.get("description", ""),
        cases=cases,
    )
