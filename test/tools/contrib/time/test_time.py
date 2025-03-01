# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
from datetime import datetime
from typing import Callable

import pytest

from autogen.tools.contrib import TimeTool


class TestTimeTool:
    def test_time_tool_init(self) -> None:
        """Test initialization of TimeTool."""

        test_date_time_format = "%Y_%m_%d %H_%M_%S"

        time_tool = TimeTool(
            date_time_format=test_date_time_format,
        )

        assert time_tool.name == "date_time"
        assert time_tool._date_time_format == test_date_time_format
        assert time_tool.description == "Get the current computer's date and time."
        assert isinstance(time_tool.func, Callable)  # type: ignore[arg-type]

        expected_schema = {
            "description": time_tool.description,
            "name": time_tool.name,
            "parameters": {
                "properties": {
                    "date_time_format": {
                        "description": "date/time Python format",
                        "default": test_date_time_format,
                        "type": "string",
                    }
                },
                "required": [],
                "type": "object",
            },
        }
        assert time_tool.function_schema == expected_schema

    @pytest.mark.asyncio
    async def test_time_tool_call(self) -> None:
        """Test calling TimeTool returns a valid date/time."""

        time_tool = TimeTool()

        result = await time_tool.func()
        assert isinstance(result, str)

        try:
            datetime.strptime(result, "%Y-%m-%d %H:%M:%S")
        except ValueError as e:
            pytest.fail(f"Result is not in the expected format: {e}")
