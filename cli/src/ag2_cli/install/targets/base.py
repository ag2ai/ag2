"""Base target classes for installing skills to different IDEs/agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path

from ..registry import ContentItem

AG2_MARKER = "<!-- ag2-skills -->"


_YAML_SPECIAL_VALUES = frozenset({
    "true",
    "false",
    "yes",
    "no",
    "on",
    "off",
    "null",
    "~",
})


def _needs_quoting(v: str) -> bool:
    """Check if a YAML string value needs quoting."""
    if not v:
        return True
    if v.lower() in _YAML_SPECIAL_VALUES:
        return True
    # Strings that look like numbers
    try:
        float(v)
        return True
    except ValueError:
        pass
    # Strings with special YAML chars
    return " " in v or any(c in v for c in "{}[],:&*?|-<>=!%@#")


def format_frontmatter(fm: dict) -> str:
    """Format a dict as YAML frontmatter."""
    if not fm:
        return ""
    lines = ["---"]
    for k, v in fm.items():
        if isinstance(v, bool):
            lines.append(f"{k}: {str(v).lower()}")
        elif isinstance(v, list):
            items = ", ".join(str(i) for i in v)
            lines.append(f"{k}: [{items}]")
        elif isinstance(v, (int, float)):
            lines.append(f"{k}: {v}")
        elif isinstance(v, str) and _needs_quoting(v):
            escaped = v.replace("\\", "\\\\").replace('"', '\\"')
            lines.append(f'{k}: "{escaped}"')
        else:
            lines.append(f"{k}: {v}")
    lines.append("---")
    return "\n".join(lines)


class Target(ABC):
    """Base class for all install targets."""

    name: str
    display_name: str
    detect_paths: list[str]

    def detect(self, project_dir: Path) -> bool:
        """Check if this target is in use in the project."""
        return any((project_dir / p).exists() for p in self.detect_paths)

    @abstractmethod
    def install(self, project_dir: Path, items: list[ContentItem]) -> list[Path]:
        """Install items. Returns list of created file paths."""

    @abstractmethod
    def uninstall(self, project_dir: Path) -> list[Path]:
        """Remove installed AG2 items. Returns list of removed paths."""


class DirectoryTarget(Target):
    """Target that writes individual rule files to a directory."""

    def __init__(
        self,
        name: str,
        display_name: str,
        rules_dir: str,
        file_ext: str = ".md",
        detect_paths: list[str] | None = None,
        prefix: str = "ag2-",
        transform_frontmatter: Callable[[ContentItem], dict] | None = None,
    ):
        self.name = name
        self.display_name = display_name
        self.rules_dir = rules_dir
        self.file_ext = file_ext
        self.detect_paths = detect_paths or []
        self.prefix = prefix
        self._transform_fm = transform_frontmatter

    def _filename(self, item: ContentItem) -> str:
        return f"{self.prefix}{item.name}{self.file_ext}"

    def _transform(self, item: ContentItem) -> dict:
        if self._transform_fm:
            return self._transform_fm(item)
        return {"description": item.description}

    def install(self, project_dir: Path, items: list[ContentItem]) -> list[Path]:
        out_dir = project_dir / self.rules_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for item in items:
            fm = self._transform(item)
            content = format_frontmatter(fm) + "\n\n" + item.body
            path = out_dir / self._filename(item)
            path.write_text(content)
            paths.append(path)
        return paths

    def uninstall(self, project_dir: Path) -> list[Path]:
        out_dir = project_dir / self.rules_dir
        if not out_dir.is_dir():
            return []
        removed = []
        for f in out_dir.glob(f"{self.prefix}*{self.file_ext}"):
            f.unlink()
            removed.append(f)
        return removed


class SingleFileTarget(Target):
    """Target that appends all rules to a single file."""

    def __init__(
        self,
        name: str,
        display_name: str,
        file_path: str,
        detect_paths: list[str] | None = None,
    ):
        self.name = name
        self.display_name = display_name
        self.file_path = file_path
        self.detect_paths = detect_paths or []

    def _format_section(self, item: ContentItem) -> str:
        return f"## AG2: {item.name}\n\n{item.body}"

    def install(self, project_dir: Path, items: list[ContentItem]) -> list[Path]:
        path = project_dir / self.file_path
        sections = [self._format_section(item) for item in items]
        block = f"\n\n{AG2_MARKER}\n# AG2 Framework Skills\n\n" + "\n\n".join(sections) + f"\n{AG2_MARKER}\n"

        if path.exists():
            existing = path.read_text()
            if AG2_MARKER in existing:
                start = existing.index(AG2_MARKER)
                try:
                    end = existing.index(AG2_MARKER, start + 1) + len(AG2_MARKER)
                except ValueError:
                    # Single marker found (corrupted) — replace from marker to end
                    end = len(existing)
                content = existing[:start].rstrip() + block + existing[end:].lstrip("\n")
            else:
                content = existing.rstrip() + block
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            content = block.lstrip("\n")

        path.write_text(content)
        return [path]

    def uninstall(self, project_dir: Path) -> list[Path]:
        path = project_dir / self.file_path
        if not path.exists():
            return []
        content = path.read_text()
        if AG2_MARKER not in content:
            return []
        start = content.index(AG2_MARKER)
        try:
            end = content.index(AG2_MARKER, start + 1) + len(AG2_MARKER)
        except ValueError:
            # Single marker (corrupted) — remove from marker to end
            end = len(content)
        new_content = content[:start].rstrip() + content[end:].lstrip("\n")
        if new_content.strip():
            path.write_text(new_content)
        else:
            path.unlink()
        return [path]
