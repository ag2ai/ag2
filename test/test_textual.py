# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from ag2.textual import TUIAgent


def test_tui_agent_can_be_imported() -> None:
    assert TUIAgent.__name__ == "TUIAgent"
