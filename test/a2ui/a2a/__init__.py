# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

# The A2UI↔A2A bridge tests import the optional ``a2a-sdk``; skip the whole
# subpackage (mirroring ``test/a2a/__init__.py``) when it is not installed
# so collection doesn't hard-fail on minimal installs.
pytest.importorskip("a2a")
