# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from ag2.tools import tool
from ag2.tools.final.function_tool import FunctionToolSchema


def test_function_tool_schema_defaults_to_not_deferred():
    schema = FunctionToolSchema()
    assert schema.defer_loading is False


def test_tool_decorator_threads_defer_loading():
    @tool(defer_loading=True)
    def get_weather(location: str) -> str:
        """Get the weather at a location."""
        return location

    assert get_weather.schema.defer_loading is True


def test_tool_decorator_defaults_to_not_deferred():
    @tool
    def ping() -> str:
        """Ping."""
        return "pong"

    assert ping.schema.defer_loading is False
