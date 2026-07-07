# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from contextlib import AsyncExitStack, ExitStack
from dataclasses import dataclass, field

from ag2.annotations import Context, Variable
from ag2.events import BuiltinToolCallEvent, ToolCallEvent
from ag2.middleware import BaseMiddleware
from ag2.tools.schemas import ToolSchema
from ag2.tools.tool import Tool

from ._resolve import resolve_variable

GOOGLE_MAPS_TOOL_NAME = "google_maps"


@dataclass(slots=True)
class GoogleMapsToolSchema(ToolSchema):
    type: str = field(default=GOOGLE_MAPS_TOOL_NAME, init=False)
    latitude: float | None = None
    longitude: float | None = None
    language_code: str | None = None
    enable_widget: bool = False


class GoogleMapsTool(Tool):
    """Grounding with Google Maps (Gemini native ``google_maps`` tool).

    Provider support:

    - **Gemini** (Gemini 3 family) — enables Maps grounding. ``latitude`` /
      ``longitude`` bias results to a location via the request's
      ``tool_config.retrieval_config``; ``language_code`` localises results;
      ``enable_widget`` requests the interactive Maps widget context token.

    - All other providers raise
      :class:`~ag2.exceptions.UnsupportedToolError`.

    See:
    - https://ai.google.dev/gemini-api/docs/google-maps
    """

    __slots__ = (
        "_params",
        "name",
    )

    def __init__(
        self,
        *,
        latitude: float | Variable | None = None,
        longitude: float | Variable | None = None,
        language_code: str | Variable | None = None,
        enable_widget: bool | Variable = False,
    ) -> None:
        self._params: dict[str, object] = {}
        if latitude is not None:
            self._params["latitude"] = latitude
        if longitude is not None:
            self._params["longitude"] = longitude
        if language_code is not None:
            self._params["language_code"] = language_code
        if enable_widget:
            self._params["enable_widget"] = enable_widget

        self.name = GOOGLE_MAPS_TOOL_NAME

    async def schemas(self, context: "Context") -> list[GoogleMapsToolSchema]:
        resolved = {k: resolve_variable(v, context, param_name=k) for k, v in self._params.items()}
        return [GoogleMapsToolSchema(**resolved)]

    def register(
        self,
        stack: "ExitStack | AsyncExitStack",
        context: "Context",
        *,
        middleware: Iterable["BaseMiddleware"] = (),
    ) -> None:
        async def execute(event: "ToolCallEvent", context: "Context") -> None:
            pass

        stack.enter_context(
            context.stream.where(BuiltinToolCallEvent.name == GOOGLE_MAPS_TOOL_NAME).sub_scope(execute),
        )
