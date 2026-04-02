"""GitHub Copilot target — installs as instruction files."""

from __future__ import annotations

from pathlib import Path

from ..registry import ContentItem
from .base import Target, format_frontmatter


class CopilotTarget(Target):
    name = "copilot"
    display_name = "GitHub Copilot"
    detect_paths = [".github"]

    def install(self, project_dir: Path, items: list[ContentItem]) -> list[Path]:
        inst_dir = project_dir / ".github" / "instructions"
        inst_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for item in items:
            globs = item.frontmatter.get("globs", "**/*.py")
            fm = {"applyTo": globs}
            content = format_frontmatter(fm) + "\n\n" + item.body
            path = inst_dir / f"ag2-{item.name}.instructions.md"
            path.write_text(content)
            paths.append(path)
        return paths

    def uninstall(self, project_dir: Path) -> list[Path]:
        inst_dir = project_dir / ".github" / "instructions"
        if not inst_dir.is_dir():
            return []
        removed = []
        for f in inst_dir.glob("ag2-*.instructions.md"):
            f.unlink()
            removed.append(f)
        return removed
