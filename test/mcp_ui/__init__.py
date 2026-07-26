# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

# `mcp` is an optional extra; skip the whole package when it isn't installed
# (mirrors test/a2a/__init__.py). beta-test.yml runs without ag2[mcp].
pytest.importorskip("mcp")
