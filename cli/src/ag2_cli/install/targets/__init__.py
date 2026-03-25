"""Target registry — all supported IDE/agent install targets."""

from __future__ import annotations

from pathlib import Path

from ..registry import ContentItem
from .base import DirectoryTarget, SingleFileTarget, Target
from .claude import ClaudeTarget
from .copilot import CopilotTarget

# -- Frontmatter transform functions --


def _cursor_fm(item: ContentItem) -> dict:
    fm: dict = {"description": item.description}
    if "globs" in item.frontmatter:
        fm["globs"] = item.frontmatter["globs"]
    fm["alwaysApply"] = item.frontmatter.get("alwaysApply", False)
    return fm


def _windsurf_fm(item: ContentItem) -> dict:
    fm: dict = {"description": item.description}
    if "globs" in item.frontmatter:
        fm["globs"] = item.frontmatter["globs"]
    return fm


def _continue_fm(item: ContentItem) -> dict:
    fm: dict = {"name": f"ag2-{item.name}", "description": item.description}
    if "globs" in item.frontmatter:
        fm["globs"] = item.frontmatter["globs"]
    fm["alwaysApply"] = item.frontmatter.get("alwaysApply", False)
    return fm


def _openhands_fm(item: ContentItem) -> dict:
    return {
        "name": f"ag2-{item.name}",
        "trigger_type": "keyword",
        "keywords": "ag2, autogen, multi-agent",
    }


def _cline_fm(item: ContentItem) -> dict:
    fm: dict = {"description": item.description}
    if "globs" in item.frontmatter:
        fm["globs"] = item.frontmatter["globs"]
    return fm


# -- Target definitions --

_ALL_TARGETS: list[Target] = [
    # === Directory targets (individual files per rule) ===
    DirectoryTarget(
        name="cursor",
        display_name="Cursor",
        rules_dir=".cursor/rules",
        file_ext=".mdc",
        detect_paths=[".cursor"],
        transform_frontmatter=_cursor_fm,
    ),
    ClaudeTarget(),
    CopilotTarget(),
    DirectoryTarget(
        name="windsurf",
        display_name="Windsurf",
        rules_dir=".windsurf/rules",
        detect_paths=[".windsurf", ".windsurfrules"],
        transform_frontmatter=_windsurf_fm,
    ),
    DirectoryTarget(
        name="cline",
        display_name="Cline",
        rules_dir=".clinerules",
        detect_paths=[".clinerules", ".cline"],
        transform_frontmatter=_cline_fm,
    ),
    DirectoryTarget(
        name="roo",
        display_name="Roo Code",
        rules_dir=".roo/rules",
        detect_paths=[".roo", ".roorules"],
    ),
    DirectoryTarget(
        name="continue",
        display_name="Continue.dev",
        rules_dir=".continue/rules",
        detect_paths=[".continue", ".continuerules"],
        transform_frontmatter=_continue_fm,
    ),
    DirectoryTarget(
        name="jetbrains",
        display_name="JetBrains AI",
        rules_dir=".aiassistant/rules",
        detect_paths=[".aiassistant", ".idea"],
    ),
    DirectoryTarget(
        name="amazon-q",
        display_name="Amazon Q Developer",
        rules_dir=".amazonq/rules",
        detect_paths=[".amazonq"],
    ),
    DirectoryTarget(
        name="augment",
        display_name="Augment Code",
        rules_dir=".augment/rules",
        detect_paths=[".augment", ".augment-guidelines"],
    ),
    DirectoryTarget(
        name="tabnine",
        display_name="Tabnine",
        rules_dir=".tabnine/guidelines",
        detect_paths=[".tabnine"],
    ),
    DirectoryTarget(
        name="trae",
        display_name="Trae",
        rules_dir=".trae/rules",
        detect_paths=[".trae"],
    ),
    DirectoryTarget(
        name="openhands",
        display_name="OpenHands",
        rules_dir=".openhands/microagents",
        detect_paths=[".openhands"],
        transform_frontmatter=_openhands_fm,
    ),
    # === Single-file targets (append to one file) ===
    SingleFileTarget(
        name="aider",
        display_name="Aider",
        file_path="CONVENTIONS.md",
        detect_paths=["CONVENTIONS.md", ".aider.conf.yml"],
    ),
    SingleFileTarget(
        name="zed",
        display_name="Zed",
        file_path=".rules",
        detect_paths=[".rules"],
    ),
    SingleFileTarget(
        name="agents-md",
        display_name="AGENTS.md (cross-tool)",
        file_path="AGENTS.md",
        detect_paths=["AGENTS.md"],
    ),
    SingleFileTarget(
        name="replit",
        display_name="Replit",
        file_path="replit.md",
        detect_paths=["replit.md", ".replit"],
    ),
    SingleFileTarget(
        name="bolt",
        display_name="Bolt",
        file_path=".bolt/prompt",
        detect_paths=[".bolt"],
    ),
]

_TARGET_MAP = {t.name: t for t in _ALL_TARGETS}


def get_target(name: str) -> Target | None:
    """Get a target by name."""
    return _TARGET_MAP.get(name)


def get_all_targets() -> list[Target]:
    """Get all registered targets."""
    return list(_ALL_TARGETS)


def detect_targets(project_dir: Path) -> list[Target]:
    """Auto-detect which targets are present in a project directory."""
    return [t for t in _ALL_TARGETS if t.detect(project_dir)]
