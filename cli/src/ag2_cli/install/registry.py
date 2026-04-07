"""Content pack registry — loads bundled skill packs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

CONTENT_DIR = Path(__file__).parent.parent / "content"


@dataclass
class ContentItem:
    name: str
    description: str
    category: str  # 'rule', 'skill', 'agent', 'command'
    frontmatter: dict
    body: str


@dataclass
class Pack:
    name: str
    display_name: str
    description: str
    version: str
    items: list[ContentItem] = field(default_factory=list)


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Parse YAML frontmatter from markdown text."""
    if not text.startswith("---"):
        return {}, text
    end = text.find("---", 3)
    if end == -1:
        return {}, text
    fm_text = text[3:end].strip()
    body = text[end + 3 :].strip()
    fm: dict = {}
    for line in fm_text.split("\n"):
        if ":" in line:
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()
            # Handle bracket list syntax: [a, b, c]
            if value.startswith("[") and value.endswith("]"):
                inner = value[1:-1]
                fm[key] = [item.strip().strip('"').strip("'") for item in inner.split(",") if item.strip()]
                continue
            value = value.strip('"').strip("'")
            if value.lower() == "true":
                value = True
            elif value.lower() == "false":
                value = False
            fm[key] = value
    return fm, body


def load_pack(name: str) -> Pack | None:
    """Load a content pack by name."""
    pack_dir = CONTENT_DIR / name
    if not pack_dir.is_dir():
        return None
    manifest_path = pack_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text())
    pack = Pack(
        name=manifest["name"],
        display_name=manifest["display_name"],
        description=manifest["description"],
        version=manifest["version"],
    )
    for category, subdir in [
        ("rule", "rules"),
        ("skill", "skills"),
        ("agent", "agents"),
        ("command", "commands"),
    ]:
        cat_dir = pack_dir / subdir
        if not cat_dir.is_dir():
            continue
        for md_file in sorted(cat_dir.glob("*.md")):
            text = md_file.read_text()
            fm, body = parse_frontmatter(text)
            item = ContentItem(
                name=fm.get("name", md_file.stem),
                description=fm.get("description", ""),
                category=category,
                frontmatter=fm,
                body=body,
            )
            pack.items.append(item)
    return pack


def list_packs() -> list[str]:
    """List available pack names."""
    if not CONTENT_DIR.is_dir():
        return []
    return [d.name for d in sorted(CONTENT_DIR.iterdir()) if d.is_dir() and (d / "manifest.json").exists()]
