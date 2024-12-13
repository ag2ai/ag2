# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

from autogen.tools import Tool


class TestTool:
    def test_init(self) -> None:
        tool = Tool(name="Test Tool", description="A test tool", func=lambda: None)
        assert tool.name == "Test Tool"

        assert False, "Test not implemented"
