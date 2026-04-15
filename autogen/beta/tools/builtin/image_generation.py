# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Literal

from autogen.beta.annotations import Context, Variable
from autogen.beta.middleware import BaseMiddleware
from autogen.beta.tools.schemas import ToolSchema
from autogen.beta.tools.tool import Tool

from ._resolve import resolve_variable

IMAGE_GENERATION_TOOL_NAME = "image_generation"


@dataclass(slots=True)
class ImageGenerationToolSchema(ToolSchema):
    """Schema for the image_generation builtin tool (OpenAI Responses API)."""

    type: str = field(default=IMAGE_GENERATION_TOOL_NAME, init=False)
    quality: Literal["low", "medium", "high", "auto"] | None = None
    size: str | None = None
    background: Literal["transparent", "opaque", "auto"] | None = None
    output_format: Literal["png", "jpeg", "webp"] | None = None
    output_compression: int | None = None
    partial_images: int | None = None


class ImageGenerationTool(Tool):
    """Builtin image generation tool for the OpenAI Responses API.

    Instructs the model to generate images inline during the conversation.
    Only supported with ``OpenAIResponsesConfig`` — raises ``UnsupportedToolError``
    when used with ``OpenAIConfig`` (Chat Completions API).

    Generated images are returned as ``list[BinaryResult]`` via ``reply.files``.

    Args:
        quality: Image quality — ``"low"``, ``"medium"``, ``"high"``, or ``"auto"``.
        size: Image dimensions, e.g. ``"1024x1024"``, ``"1536x1024"``, ``"auto"``.
        background: Background type — ``"transparent"``, ``"opaque"``, or ``"auto"``.
        output_format: Output format — ``"png"``, ``"jpeg"``, or ``"webp"``.
        output_compression: Compression level 0–100 (webp/jpeg only).
        partial_images: Number of partial images to stream (1–3).
    """

    __slots__ = (
        "_params",
        "name",
    )

    def __init__(
        self,
        *,
        quality: Literal["low", "medium", "high", "auto"] | Variable | None = None,
        size: str | Variable | None = None,
        background: Literal["transparent", "opaque", "auto"] | Variable | None = None,
        output_format: Literal["png", "jpeg", "webp"] | Variable | None = None,
        output_compression: int | Variable | None = None,
        partial_images: int | Variable | None = None,
    ) -> None:
        self._params: dict[str, object] = {}
        if quality is not None:
            self._params["quality"] = quality
        if size is not None:
            self._params["size"] = size
        if background is not None:
            self._params["background"] = background
        if output_format is not None:
            self._params["output_format"] = output_format
        if output_compression is not None:
            self._params["output_compression"] = output_compression
        if partial_images is not None:
            self._params["partial_images"] = partial_images

        self.name = IMAGE_GENERATION_TOOL_NAME

    async def schemas(self, context: "Context") -> list[ImageGenerationToolSchema]:
        resolved = {k: resolve_variable(v, context, param_name=k) for k, v in self._params.items()}
        return [ImageGenerationToolSchema(**resolved)]

    def register(
        self,
        stack: "ExitStack",
        context: "Context",
        *,
        middleware: Iterable["BaseMiddleware"] = (),
    ) -> None:
        pass
