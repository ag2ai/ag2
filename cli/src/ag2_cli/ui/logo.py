"""AG2 CLI branding — logo and banner display."""

from __future__ import annotations

from rich.console import Console
from rich.text import Text

# Pixel-art AG2 robot logo — faithful recreation of the brand icon
# Each "██" represents one filled pixel; decoded from the official SVG
_FACE_LINES = [
    "██                          ██",
    "  ██                      ██  ",
    "    ██    ██████████    ██    ",
    "      ████          ████      ",
    "    ██                  ██    ",
    "  ██    ██████████████    ██  ",
    "  ██  ████  ██████  ████  ██  ",
    "      ██████████████████      ",
]

_AG2_LINES = [
    "  ████      ██████  ██████  ",
    "██    ██  ██              ██",
    "████████  ██  ████    ████  ",
    "██    ██  ██    ██  ██      ",
    "██    ██    ████    ████████",
]

TAGLINE = "Build, run, test, and deploy multi-agent applications"


def print_banner(console: Console | None = None) -> None:
    """Print the AG2 pixel-art logo."""
    console = console or Console()

    for line in _FACE_LINES:
        console.print(Text(line, style="bold"), highlight=False)

    console.print()

    for line in _AG2_LINES:
        console.print(Text(f" {line} ", style="bold"), highlight=False)

    console.print()
    console.print(f"  {TAGLINE}", style="dim")
    console.print()
