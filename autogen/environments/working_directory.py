# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import contextlib
import os
import shutil
import tempfile
from contextvars import ContextVar
from typing import Optional

__all__ = ["WorkingDirectory"]


class WorkingDirectory:
    """Context manager for changing the current working directory."""

    _current_working_directory: ContextVar["WorkingDirectory"] = ContextVar("_current_working_directory")

    def __init__(self, path: str):
        """
        Initialize with a directory path.

        Args:
            path: The directory path to change to.
        """
        self.path = path
        self.original_path = None
        self.created_tmp = False
        self._token = None

    def __enter__(self):
        """Change to the specified directory and return self."""
        self.original_path = os.getcwd()
        if self.path:
            os.makedirs(self.path, exist_ok=True)
            os.chdir(self.path)

        # Set this as the current working directory in the context
        self._token = WorkingDirectory._current_working_directory.set(self)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Change back to the original directory and clean up if necessary."""
        # Reset the context variable if this was the active working directory
        if self._token is not None:
            WorkingDirectory._current_working_directory.reset(self._token)
            self._token = None

        if self.original_path:
            os.chdir(self.original_path)
        if self.created_tmp and self.path and os.path.exists(self.path):
            with contextlib.suppress(Exception):
                shutil.rmtree(self.path)

    @classmethod
    def create_tmp(cls):
        """Create a temporary directory and return a WorkingDirectory instance for it."""
        tmp_dir = tempfile.mkdtemp(prefix="ag2_work_dir_")
        instance = cls(tmp_dir)
        instance.created_tmp = True
        return instance

    @classmethod
    def get_current_working_directory(
        cls, working_directory: Optional["WorkingDirectory"] = None
    ) -> Optional["WorkingDirectory"]:
        """Get the current working directory or the specified one if provided."""
        if working_directory is not None:
            return working_directory
        try:
            return cls._current_working_directory.get()
        except LookupError:
            return None
