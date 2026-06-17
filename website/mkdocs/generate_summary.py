# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path


def generate_summary(root_path: Path) -> None:
    """Build the literate-nav ``SUMMARY.md`` from ``navigation_template.txt``.

    Args:
        root_path: The root path of the mkdocs project (the directory containing ``docs/``).

    """
    docs_dir = root_path / "docs"

    summary = (docs_dir / "navigation_template.txt").read_text()
    summary = "\n".join(filter(bool, (x.rstrip() for x in summary.split("\n"))))

    (docs_dir / "SUMMARY.md").write_text(summary)


if __name__ == "__main__":
    generate_summary(Path(__file__).resolve().parent)
