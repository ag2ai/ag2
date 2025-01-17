# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from typing import Generator


@contextmanager
def check_for_missing_imports() -> Generator[None, None, None]:
    """
    A context manager to temporarily suppress ImportErrors.
    Use this to attempt imports without failing immediately on missing modules.
    """
    try:
        yield
    except ImportError:
        pass  # Ignore ImportErrors during this context
