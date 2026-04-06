# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from https://github.com/ag2ai/ag2 are under the MIT License.
# SPDX-License-Identifier: MIT
# Will return the filename relative to the workspace path
import re
from pathlib import Path
from typing import Any

filename_patterns = [
    re.compile(r"^<!-- (filename:)?(.+?) -->", re.DOTALL),
    re.compile(r"^/\* (filename:)?(.+?) \*/", re.DOTALL),
    re.compile(r"^// (filename:)?(.+?)$", re.DOTALL),
    re.compile(r"^# (filename:)?(.+?)$", re.DOTALL),
]


# Raises ValueError if the file is not in the workspace
def _get_file_name_from_content(code: str, workspace_path: Path) -> str | None:
    first_line = code.split("\n")[0].strip()
    # TODO - support other languages
    for pattern in filename_patterns:
        matches = pattern.match(first_line)
        if matches is not None:
            filename = matches.group(2).strip()

            # Handle relative paths in the filename
            path = Path(filename)
            if not path.is_absolute():
                path = workspace_path / path
            path = path.resolve()
            # Throws an error if the file is not in the workspace
            relative = path.relative_to(workspace_path.resolve())
            return str(relative)
    return None


def silence_pip(code: str, lang: str) -> str:
    """Apply -qqq flag to pip install commands."""
    if lang == "python":
        regex = r"^! ?pip install"
    elif lang in ["bash", "shell", "sh", "pwsh", "powershell", "ps1"]:
        regex = r"^pip install"
    else:
        return code

    # Find lines that start with pip install and make sure "-qqq" flag is added.
    lines = code.split("\n")
    for i, line in enumerate(lines):
        # use regex to find lines that start with pip install.
        match = re.search(regex, line)
        if match is not None and "-qqq" not in line:
            lines[i] = line.replace(match.group(0), match.group(0) + " -qqq")
    return "\n".join(lines)


def format_chat_result(result: Any) -> str:
    """
    Format a ChatResult object into a readable summary.

    Useful for displaying exploration results later.

    Args:
        result: The ChatResult object from explore() or initiate_chat()

    Returns:
        Formatted string summary

    Example:
        >>> from autogen.coding.utils import format_chat_result
        >>> result = executor.explore(verbose=False)
        >>> print(format_chat_result(result))
    """
    lines = []
    lines.append("=" * 80)
    lines.append("📊 Exploration Session Summary")
    lines.append("=" * 80)

    # Basic stats
    lines.append(f"\n• Total messages: {len(result.chat_history)}")
    lines.append(f"• Chat ID: {result.chat_id}")

    # Cost info
    if hasattr(result, "cost") and result.cost:
        total_cost = (result.cost or {}).get("usage_including_cached_inference", {}).get("total_cost", 0)
        lines.append(f"• Cost: ${total_cost:.4f}")

    # Summary
    if result.summary:
        lines.append("\n💬 Final Status:")
        summary_preview = result.summary[:300] + "..." if len(result.summary) > 300 else result.summary
        for line in summary_preview.split("\n"):
            lines.append(f"   {line}")

    # Last few messages
    lines.append("\n📝 Last 3 messages:")
    for msg in result.chat_history[-3:]:
        role = msg.get("name", msg.get("role", "unknown"))
        content_preview = msg.get("content", "")[:100]
        if len(msg.get("content", "")) > 100:
            content_preview += "..."
        lines.append(f"   [{role}]: {content_preview}")

    lines.append("\n" + "=" * 80)

    return "\n".join(lines)
