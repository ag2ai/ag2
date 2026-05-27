"""Shared Rich console instances for consistent output."""

from __future__ import annotations

from rich.console import Console

from .theme import AG2_THEME

console = Console(theme=AG2_THEME)
err_console = Console(stderr=True, theme=AG2_THEME)
