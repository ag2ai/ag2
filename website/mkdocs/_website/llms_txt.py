# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Generate ``llms.txt`` and ``llms-full.txt`` files from the User Guide docs.

The output follows the https://llmstxt.org/ specification:

- a single H1 with the project name,
- a blockquote summary,
- zero or more free-form markdown sections with details,
- zero or more H2 "file list" sections whose items are ``[name](url)`` links with
  an optional ``: description`` suffix,
- an optional H2 section literally titled "Optional" holding links that may be
  skipped when a shorter context is needed.

Scope is the User Guide (``docs/user-guide/**``) so coding assistants are fed the
current ``ag2`` API rather than the removed classic API.
"""

import json
import re
from pathlib import Path

from ag2._import_utils import optional_import_block, require_optional_import

with optional_import_block():
    import yaml
    from jinja2 import Template

BASE_URL = "https://docs.ag2.ai/latest"

PROJECT_TITLE = "AG2"

SUMMARY = (
    "AG2 (`ag2`) is an async, protocol-driven Python framework for building "
    "AI agents — covering agents, tools, multi-agent networks, structured output, memory, "
    "and evaluation. This file indexes the AG2 documentation for LLMs and coding assistants."
)

# Free-form details paragraph(s) emitted after the blockquote.
DETAILS = (
    "Build with `ag2` only. The classic `autogen` API (`ConversableAgent`, "
    "`initiate_chat`, `GroupChat`) has been removed — do not use it. For ready-made "
    "setup, install the AG2 Skills with `npx skills add ag2ai/ag2-skills`."
)

# Heading used for the standalone (non-grouped) pages.
DEFAULT_SECTION = "Documentation"

# Capitalization fixes when deriving a title from a filename.
_TITLE_KEYWORDS = {
    "Ag2": "AG2",
    "Rag": "RAG",
    "Llm": "LLM",
    "Ai": "AI",
    "Mcp": "MCP",
    "Hitl": "HITL",
    "Ui": "UI",
    "A2A": "A2A",
    "Stt": "STT",
    "Tts": "TTS",
}

_MAX_DESCRIPTION = 200

# Typographic punctuation that adds no value in a plain-text file meant for machine
# ingestion, and which renders as mojibake (``â€``…) in non-UTF-8 viewers. Mapped to
# ASCII. Meaningful symbols (box-drawing diagrams, checkmarks, math) are left untouched.
_PUNCT_TABLE = {
    ord("—"): "-",  # — em dash
    ord("–"): "-",  # – en dash
    ord("−"): "-",  # − minus sign
    ord("…"): "...",  # … ellipsis
    ord("‘"): "'",  # ' left single quote
    ord("’"): "'",  # ' right single quote
    ord("“"): '"',  # " left double quote
    ord("”"): '"',  # " right double quote
    ord(" "): " ",  # non-breaking space
    ord("·"): "-",  # · middle dot
    ord("›"): ">",  # › single right angle quote
    ord("→"): "->",  # → rightwards arrow
    ord("←"): "<-",  # ← leftwards arrow
    ord("↔"): "<->",  # ↔ left-right arrow
    ord("⇒"): "=>",  # ⇒ rightwards double arrow
}


def _normalize_punctuation(text: str) -> str:
    """Replace noisy typographic punctuation with ASCII equivalents."""
    return text.translate(_PUNCT_TABLE)


def _format_title_from_filename(stem: str) -> str:
    words = stem.replace("-", " ").replace("_", " ").title().split()
    return " ".join(_TITLE_KEYWORDS.get(word, word) for word in words)


def _split_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Return ``(frontmatter_dict, body)`` for a Markdown document."""
    match = re.match(r"^---\n(.*?)\n---\n?", text, re.DOTALL)
    if not match:
        return {}, text
    try:
        frontmatter = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        frontmatter = {}
    return frontmatter, text[match.end() :]


def _page_title(frontmatter: dict[str, str], stem: str) -> str:
    return frontmatter.get("title") or frontmatter.get("sidebarTitle") or _format_title_from_filename(stem)


def _clean_inline(text: str) -> str:
    """Strip Markdown adornments so a line reads as plain prose."""
    text = re.sub(r"\{\.external-link[^}]*\}", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # links -> link text
    text = text.replace("#!python ", "")
    # Drop inline-code ticks and bold/italic asterisks, but keep underscores so code
    # identifiers like ``on_tool_execution`` survive intact.
    text = re.sub(r"[`*]+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _description(body: str) -> str:
    """Extract a one-line description from the first prose paragraph of a page."""
    body = re.sub(r"^\s*#\s+.*\n", "", body, count=1)  # drop a leading H1 if present

    paragraph: list[str] = []
    in_code = False
    for line in body.split("\n"):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        if not stripped:
            if paragraph:
                break
            continue
        # Skip structural / non-prose lines until real prose starts.
        if stripped.startswith(("#", "<", "!!!", "===", "|", ">", "-", "*", "import ", "from ")):
            if paragraph:
                break
            continue
        paragraph.append(stripped)

    text = _clean_inline(" ".join(paragraph))
    if len(text) > _MAX_DESCRIPTION:
        text = text[:_MAX_DESCRIPTION].rsplit(" ", 1)[0].rstrip(",.;:") + "…"
    return text


def _clean_for_full(body: str) -> str:
    """Tidy a page body for inclusion in ``llms-full.txt``."""
    # Reduce code-fence info strings to just the language (drop linenums/hl_lines).
    body = re.sub(r"^(```[a-zA-Z0-9_+-]*)[^\n]*$", r"\1", body, flags=re.MULTILINE)
    body = re.sub(r"\{\.external-link[^}]*\}", "", body)
    body = body.replace("#!python ", "")
    return body.strip()


def _flatten_pages(group: dict) -> list[str]:
    """Return all leaf page paths within a (possibly nested) navigation group."""
    pages: list[str] = []
    for page in group["pages"]:
        if isinstance(page, str):
            pages.append(page)
        else:
            pages.extend(_flatten_pages(page))
    return pages


def _iter_sections(nav_group: dict) -> list[tuple[str, list[str]]]:
    """Split the nav group into ``(section_title, page_paths)`` pairs, preserving order."""
    sections: list[tuple[str, list[str]]] = []
    standalone: list[str] = []
    for page in nav_group["pages"]:
        if isinstance(page, str):
            standalone.append(page)
        else:
            sections.append((page["group"], _flatten_pages(page)))
    if standalone:
        sections.insert(0, (DEFAULT_SECTION, standalone))
    return sections


def _load_page(page_path: str, site_root: Path) -> tuple[str, str, str] | None:
    """Return ``(title, description, cleaned_body)`` for a page, or ``None`` if missing."""
    md_file = site_root / f"{page_path}.md"
    if not md_file.is_file():
        return None
    frontmatter, body = _split_frontmatter(md_file.read_text(encoding="utf-8"))
    title = _page_title(frontmatter, Path(page_path).stem)
    return title, _description(body), _clean_for_full(body)


def _render_index(sections: list[tuple[str, list[str]]], pages: dict[str, tuple[str, str, str]]) -> str:
    lines = [f"# {PROJECT_TITLE}", "", f"> {SUMMARY}", "", DETAILS, ""]
    for section_title, page_paths in sections:
        entries = []
        for page_path in page_paths:
            loaded = pages.get(page_path)
            if loaded is None:
                continue
            title, description, _ = loaded
            url = f"{BASE_URL}/{page_path}/"
            entries.append(f"- [{title}]({url})" + (f": {description}" if description else ""))
        if entries:
            lines.append(f"## {section_title}")
            lines.append("")
            lines.extend(entries)
            lines.append("")

    lines.append("## Optional")
    lines.append("")
    lines.append(
        f"- [llms-full.txt]({BASE_URL}/llms-full.txt): The entire AG2 documentation concatenated into a single file."
    )
    lines.append(
        "- [AG2 Skills](https://github.com/ag2ai/ag2-skills): Installable Agent Skills that "
        "teach coding assistants the AG2 API."
    )
    lines.append("")
    return _normalize_punctuation("\n".join(lines))


def _render_full(sections: list[tuple[str, list[str]]], pages: dict[str, tuple[str, str, str]]) -> str:
    lines = [
        f"# {PROJECT_TITLE} — Full Documentation",
        "",
        f"> {SUMMARY}",
        "",
        DETAILS,
        "",
    ]
    for _, page_paths in sections:
        for page_path in page_paths:
            loaded = pages.get(page_path)
            if loaded is None:
                continue
            title, _, body = loaded
            url = f"{BASE_URL}/{page_path}/"
            lines.append("---")
            lines.append("")
            lines.append(f"# {title}")
            lines.append("")
            lines.append(f"Source: {url}")
            lines.append("")
            lines.append(body)
            lines.append("")
    return _normalize_punctuation("\n".join(lines))


@require_optional_import(["yaml", "jinja2"], "docs")
def generate_llms_txt(website_dir: Path, site_root: Path) -> None:
    """Write ``llms.txt`` and ``llms-full.txt`` to ``site_root`` from the User Guide docs.

    Args:
        website_dir: The ``website/`` directory (holds ``mint-json-template.json.jinja``).
        site_root: The MkDocs ``docs_dir`` (``website/mkdocs/docs``); files written here
            are served at the site root, e.g. ``https://docs.ag2.ai/latest/llms.txt``.
    """
    template_path = website_dir / "mint-json-template.json.jinja"
    navigation = json.loads(Template(template_path.read_text(encoding="utf-8")).render())["navigation"]
    user_guide_group = next((group for group in navigation if group["group"] == "User Guide"), None)
    if user_guide_group is None:
        # Fail loudly rather than silently shipping no llms.txt — the most likely cause is
        # the "User Guide" navigation group being renamed in mint-json-template.json.jinja.
        raise RuntimeError("Could not find the 'User Guide' navigation group; llms.txt was not generated.")

    sections = _iter_sections(user_guide_group)
    pages: dict[str, tuple[str, str, str]] = {}
    for _, page_paths in sections:
        for page_path in page_paths:
            loaded = _load_page(page_path, site_root)
            if loaded is not None:
                pages[page_path] = loaded

    if not pages:
        # The User Guide docs were not converted to Markdown before this step ran.
        raise RuntimeError(
            f"No User Guide pages found under {site_root / 'docs' / 'user-guide'}; llms.txt was not generated."
        )

    (site_root / "llms.txt").write_text(_render_index(sections, pages), encoding="utf-8")
    (site_root / "llms-full.txt").write_text(_render_full(sections, pages), encoding="utf-8")
