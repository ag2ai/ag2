"""AG2 CLI Rich theme — consistent styling across all output."""

from __future__ import annotations

from rich.theme import Theme

AG2_THEME = Theme({
    "info": "cyan",
    "success": "bold green",
    "warning": "bold yellow",
    "error": "bold red",
    "heading": "bold bright_cyan",
    "command": "bold white",
    "path": "dim underline",
    "count": "bold magenta",
    "ag2": "bold dodger_blue1",
})
