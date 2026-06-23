# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import textwrap
from pathlib import Path

import pytest


@pytest.fixture
def skill_tree(tmp_path: Path) -> Path:
    """Minimal skill tree for testing.

    Structure::

        tmp_path/
          react-best-practices/
            SKILL.md  (has version, has scripts/ and a resource)
            scripts/
              scaffold.py
            references/
              guide.md
          markdown-guide/
            SKILL.md  (no version, no scripts/, no resources)
    """
    react_dir = tmp_path / "react-best-practices"
    react_dir.mkdir(parents=True)
    (react_dir / "SKILL.md").write_text(
        textwrap.dedent("""\
            ---
            name: react-best-practices
            description: Best practices for React development
            version: 1.2.0
            ---
            # React Best Practices
            Use functional components and hooks.
        """),
        encoding="utf-8",
    )
    scripts_dir = react_dir / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "scaffold.py").write_text('print("scaffold")\n', encoding="utf-8")
    refs_dir = react_dir / "references"
    refs_dir.mkdir()
    (refs_dir / "guide.md").write_text("# Guide\nDetailed React guidance.\n", encoding="utf-8")

    md_dir = tmp_path / "markdown-guide"
    md_dir.mkdir(parents=True)
    (md_dir / "SKILL.md").write_text(
        textwrap.dedent("""\
            ---
            name: markdown-guide
            description: Guide for writing Markdown
            ---
            # Markdown Guide
            Use headings, lists, and code blocks.
        """),
        encoding="utf-8",
    )

    return tmp_path
