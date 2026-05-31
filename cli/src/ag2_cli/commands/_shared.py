"""Shared utilities for CLI commands."""

from __future__ import annotations

from typing import Any

import typer

from ..ui import console


def require_ag2() -> Any:
    """Import autogen or exit with a helpful error."""
    try:
        import autogen

        return autogen
    except ImportError:
        console.print("[error]ag2 is not installed.[/error]")
        console.print("Install it with: [command]pip install ag2[/command]")
        raise typer.Exit(1)


def extract_cost(cost_info: Any) -> float:
    """Extract total cost from AG2's cost dict structure."""
    if isinstance(cost_info, dict):
        usage = cost_info.get("usage_excluding_cached_inference", {})
        return usage.get("total_cost", 0)
    return 0.0
