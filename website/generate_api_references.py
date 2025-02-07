# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Generator


@contextmanager
def _add_pdoc_placeholder_env() -> Generator[None, None, None]:
    """Add the environment variable ADD_PDOC_PLACEHOLDER_TO_MODULE to the current environment.

    The export_module decorator in doc_utils.py uses this environment variable, if set, the decorator modifies the __module__ attribute of the decorated symbol.
    If the environment variable is not set, the decorator does not modify the __module__ attribute.
    """
    os.environ["ADD_PDOC_PLACEHOLDER_TO_MODULE"] = "1"
    try:
        yield
    finally:
        os.environ.pop("ADD_PDOC_PLACEHOLDER_TO_MODULE")


if __name__ == "__main__":
    with _add_pdoc_placeholder_env():
        from autogen._website.generate_api_references import main

        main()
