from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional


def run_pydoc_markdown(config_file: Path) -> None:
    """Run pydoc-markdown with the specified config file.

    Args:
        config_file (Path): Path to the pydoc-markdown config file
    """
    try:
        subprocess.run(["pydoc-markdown"], check=True, capture_output=True, text=True)
        print(f"Successfully ran pydoc-markdown with config: {config_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error running pydoc-markdown: {e.stderr}")
        sys.exit(1)
    except FileNotFoundError:
        print("pydoc-markdown not found. Please install it with: pip install pydoc-markdown")
        sys.exit(1)


def escape_html_tags(content: str) -> str:
    """Escape all angle brackets < > in the content.

    Args:
        content (str): Input text content

    Returns:
        str: Content with all angle brackets escaped
    """
    return content.replace("<", r"\<").replace("{", r"\{")


def read_file_content(file_path: str) -> str:
    """Read content from a file.

    Args:
        file_path (str): Path to the file

    Returns:
        str: Content of the file
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def write_file_content(file_path: str, content: str) -> None:
    """Write content to a file.

    Args:
        file_path (str): Path to the file
        content (str): Content to write
    """
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)


def add_code_fences(content: str) -> str:
    """Add Python code fence markers to content starting with 'import'.

    Args:
        content: Content to process
    Returns:
        Content with Python code blocks properly fenced
    """
    lines = content.split("\n")
    output = []
    i = 0

    while i < len(lines):
        if lines[i].strip().startswith("import"):
            block = []
            # Collect lines until next markdown marker
            while i < len(lines) and not lines[i].strip().startswith(("---", "#", "-", "`")):
                block.append(lines[i])
                i += 1
            if block:
                output.extend(["```python", *block, "```"])
        else:
            output.append(lines[i])
            i += 1

    return "\n".join(output)


def convert_md_to_mdx(input_dir: Path) -> None:
    """Convert all .md files in directory to .mdx while preserving structure.

    Args:
        input_dir (Path): Directory containing .md files to convert
    """
    if not input_dir.exists():
        print(f"Directory not found: {input_dir}")
        sys.exit(1)

    for md_file in input_dir.rglob("*.md"):
        mdx_file = md_file.with_suffix(".mdx")

        # Read content from .md file
        content = md_file.read_text(encoding="utf-8")

        # Escape HTML tags
        processed_content = escape_html_tags(content)

        # Add code fences
        processed_content = add_code_fences(processed_content)

        # Write content to .mdx file
        mdx_file.write_text(processed_content, encoding="utf-8")

        # Remove original .md file
        md_file.unlink()
        print(f"Converted: {md_file} -> {mdx_file}")


def main() -> None:
    script_dir = Path(__file__).parent.absolute()

    parser = argparse.ArgumentParser(description="Process API reference documentation")
    parser.add_argument("--config", type=Path, help="Path to pydoc-markdown config file", default=script_dir)
    parser.add_argument(
        "--api-dir",
        type=Path,
        help="Directory containing API documentation to process",
        default=script_dir / "docs" / "reference",
    )

    args = parser.parse_args()

    # Run pydoc-markdown
    print("Running pydoc-markdown...")
    run_pydoc_markdown(args.config)

    # Convert MD to MDX
    print("Converting MD files to MDX...")
    convert_md_to_mdx(args.api_dir)

    print("API reference processing complete!")


if __name__ == "__main__":
    main()
