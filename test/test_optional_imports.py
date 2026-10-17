# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the module names accepted by the optional-dependency guard.

Scoped to what changed when the version-constraint machinery was removed
along with the ``packaging`` dependency: a constraint is now rejected
instead of being parsed and then ignored, and the name is matched in full
rather than truncated at the first dot.
"""

import json  # noqa: F401 — must be in sys.modules for the guard to observe it

import pytest

from ag2._import_utils import require_optional_import


@pytest.mark.parametrize("constraint", ["json>=1.0", "json>1.0", "json<=1.0", "json<1.0", "json>=1.0,<2.0"])
def test_version_constraint_is_rejected_rather_than_ignored(constraint: str) -> None:
    """Constraints were never honoured, so failing loudly beats a false sense of enforcement."""
    with pytest.raises(ValueError, match="Invalid package information"):
        require_optional_import(constraint, "test")


@pytest.mark.parametrize("name", ["", "..", ".json", "json.", "foo-bar", "1json"])
def test_names_that_can_never_be_imported_are_rejected(name: str) -> None:
    with pytest.raises(ValueError, match="Invalid package information"):
        require_optional_import(name, "test")


def test_missing_submodule_of_a_present_parent_still_raises() -> None:
    """A dotted name is checked in full, not truncated to its root package."""

    @require_optional_import("json.no_such_submodule", "somextra")
    def f() -> str:
        return "ran"

    with pytest.raises(ImportError, match="json.no_such_submodule"):
        f()
