# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from autogen.beta import Agent
from autogen.beta.tools import tool


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


@tool
def greet(name: str) -> str:
    """Greet someone."""
    return f"Hello, {name}!"


def make_agent(**kwargs: Any) -> Agent:
    """Create a test agent without a real config."""
    defaults: dict[str, Any] = {
        "name": "test_agent",
        "prompt": "You are a test assistant.",
        "tools": [add, multiply],
    }
    defaults.update(kwargs)
    return Agent(**defaults)
