# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env python3

import subprocess
import sys
from pathlib import Path

from autogen._website.process_notebooks import EDIT_URL_HTML


def get_git_tracked_mdx_files() -> list[Path]:
    """Get list of git tracked mdx files using git command."""
    try:
        result = subprocess.run(
            ["git", "ls-files", "website/docs/**/*.mdx"],
            capture_output=True,
            text=True,
            check=True,
        )
        files = result.stdout.splitlines()
        return [Path(file) for file in files]
    except subprocess.CalledProcessError as e:
        print(f"Error getting git tracked mdx files: {e}")
        return []


def main() -> None:
    """Main function to check the edit URL in git tracked mdx files."""

    failed = False
    files_updated = False

    git_tracked_mdx_files = get_git_tracked_mdx_files()
    # print(f"Total git tracked mdx files: {len(git_tracked_mdx_files)}")

    html_placeholder = [line for line in EDIT_URL_HTML.splitlines() if line.strip() != ""][0]

    for mdx_file in git_tracked_mdx_files:
        try:
            with open(mdx_file, "r") as f:
                content = f.read()

            if html_placeholder not in content:
                print(f"Adding editUrl to {mdx_file}")
                with open(mdx_file, "a") as f:
                    f.write(EDIT_URL_HTML.format(file_path=str(mdx_file)))
                files_updated = True

        except Exception as e:
            print(f"Error processing {mdx_file}: {e}")
            failed = True

    if files_updated:
        print("Added editUrl in the missing mdx files. Please add them to git and commit.")
        failed = True

    sys.exit(int(failed))


if __name__ == "__main__":
    main()
