"""Claude Code target — installs skills, commands, and rules."""

from __future__ import annotations

from pathlib import Path

from ..registry import ContentItem
from .base import Target, format_frontmatter


class ClaudeTarget(Target):
    name = "claude"
    display_name = "Claude Code"
    detect_paths = ["CLAUDE.md", ".claude"]

    def install(self, project_dir: Path, items: list[ContentItem]) -> list[Path]:
        paths = []
        for item in items:
            if item.category == "command":
                paths.append(self._install_command(project_dir, item))
            else:
                paths.append(self._install_skill(project_dir, item))
        return paths

    def _install_skill(self, project_dir: Path, item: ContentItem) -> Path:
        skill_dir = project_dir / ".claude" / "skills" / f"ag2-{item.name}"
        skill_dir.mkdir(parents=True, exist_ok=True)
        fm = {
            "name": f"ag2-{item.name}",
            "description": item.description,
        }
        if item.category == "rule":
            fm["auto_invocable"] = True
        if item.category in ("skill", "agent", "command"):
            fm["user_invocable"] = True
        content = format_frontmatter(fm) + "\n\n" + item.body
        path = skill_dir / "SKILL.md"
        path.write_text(content)
        return path

    def _install_command(self, project_dir: Path, item: ContentItem) -> Path:
        cmd_dir = project_dir / ".claude" / "commands"
        cmd_dir.mkdir(parents=True, exist_ok=True)
        content = f"# {item.name}\n\n{item.description}\n\n{item.body}"
        path = cmd_dir / f"ag2-{item.name}.md"
        path.write_text(content)
        return path

    def uninstall(self, project_dir: Path) -> list[Path]:
        import shutil

        removed = []
        skills_dir = project_dir / ".claude" / "skills"
        if skills_dir.is_dir():
            for d in skills_dir.iterdir():
                if d.is_dir() and d.name.startswith("ag2-"):
                    # Collect all files before removal for reporting
                    for f in d.rglob("*"):
                        if f.is_file():
                            removed.append(f)
                    shutil.rmtree(d)
        cmd_dir = project_dir / ".claude" / "commands"
        if cmd_dir.is_dir():
            for f in cmd_dir.glob("ag2-*.md"):
                f.unlink()
                removed.append(f)
        return removed
