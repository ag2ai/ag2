# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest


def test_missing_optional_dependency_raises_import_error() -> None:
    """_missing_optional_dependency should return a Mock that raises ImportError when called."""
    from autogen.beta.debug import _missing_optional_dependency

    original_error = ImportError("no module named 'foo'")
    mock = _missing_optional_dependency("TestThing", original_error)

    with pytest.raises(ImportError, match='TestThing requires optional dependencies'):
        mock()


def test_missing_optional_dependency_raises_on_any_args() -> None:
    """The mock should raise regardless of what args/kwargs are passed."""
    from autogen.beta.debug import _missing_optional_dependency

    mock = _missing_optional_dependency("TestThing", ImportError("missing"))

    with pytest.raises(ImportError):
        mock("arg1", key="val")


def test_import_error_fallback_for_client() -> None:
    """When .client import fails, DebugClient should be a mock that raises ImportError."""
    with patch.dict("sys.modules", {"httpx": None}):
        from autogen.beta.debug import _missing_optional_dependency

        mock = _missing_optional_dependency("DebugClient", ImportError("no httpx"))
        with pytest.raises(ImportError, match="DebugClient requires optional dependencies"):
            mock()
