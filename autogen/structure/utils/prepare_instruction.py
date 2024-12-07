# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from https://github.com/lion-agi/lion-core are under the Apache-2.0 License
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Literal

from pydantic import BaseModel

from .break_down_pydantic import break_down_pydantic_annotation

DEFAULT_SYSTEM = "You are a helpful AI assistant. Let's think step by step."

CONTENT_KEYS = [
    "guidance",
    "instruction",
    "context",
    "request_response_format",
    "tool_schemas",
]


def prepare_request_response_format(request_fields: dict) -> str:
    """
    Prepare a standardized format for request responses.

    Args:
        request_fields: Dictionary of fields to include in response

    Returns:
        str: Formatted response template
    """
    return (
        "**MUST RETURN JSON-PARSEABLE RESPONSE ENCLOSED BY JSON CODE BLO" f"CKS.** \n```json\n{request_fields}\n```"
    ).strip()


def format_image_item(idx: str, x: str, /) -> dict[str, Any]:
    """
    Create an image_url dictionary for content formatting.

    Args:
        idx: Base64 encoded image data
        x: Image detail level

    Returns:
        dict: Formatted image item
    """
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{idx}",
            "detail": x,
        },
    }


def format_text_item(item: Any) -> str:
    """
    Format a text item or list of items into a string.

    Args:
        item: Text item(s) to format

    Returns:
        str: Formatted text
    """
    msg = ""
    item = [item] if not isinstance(item, list) else item
    for j in item:
        if isinstance(j, dict):
            for k, v in j.items():
                msg += f"- {k}: {v} \n\n"
        else:
            msg += f"{j}\n"
    return msg


def format_text_content(content: dict) -> str:
    """
    Format dictionary content into a structured text format.

    Args:
        content: Dictionary containing content sections

    Returns:
        str: Formatted text content
    """
    if "plain_content" in content and isinstance(content["plain_content"], str):
        return content["plain_content"]

    msg = "\n---\n # Task\n"
    for k, v in content.items():
        if k in CONTENT_KEYS:
            if k == "request_response_format":
                k = "response format"
            msg += f"## **Task {k}**\n{format_text_item(v)}\n\n"
    msg += "\n\n---\n"
    return msg


def format_image_content(
    text_content: str,
    images: list,
    image_detail: Literal["low", "high", "auto"],
) -> dict[str, Any]:
    """
    Format text content with images for message content.

    Args:
        text_content: The text content to format
        images: List of images to include
        image_detail: Level of detail for images

    Returns:
        dict: Formatted content with text and images
    """
    content = [{"type": "text", "text": text_content}]
    content.extend(format_image_item(i, image_detail) for i in images)
    return content


def prepare_instruction_content(
    guidance: str | None = None,
    instruction: str | None = None,
    context: str | dict | list | None = None,
    request_fields: dict | list[str] | None = None,
    plain_content: str | None = None,
    request_model: BaseModel = None,
    images: str | list | None = None,
    image_detail: Literal["low", "high", "auto"] | None = None,
    tool_schemas: dict | None = None,
) -> dict:
    """
    Prepare the content for an instruction message.

    Args:
        guidance: Optional guidance text
        instruction: Main instruction content
        context: Additional context information
        request_fields: Fields to request in response
        plain_content: Plain text content
        request_model: Pydantic model for structured requests
        images: Images to include
        image_detail: Level of detail for images
        tool_schemas: Tool schemas to include

    Returns:
        Note: Prepared instruction content

    Raises:
        ValueError: If both request_fields and request_model are provided
    """
    if request_fields and request_model:
        raise ValueError("only one of request_fields or request_model can be provided")

    out_ = {"context": []}
    if guidance:
        out_["guidance"] = guidance
    if instruction:
        out_["instruction"] = instruction
    if context:
        if isinstance(context, list):
            out_["context"].extend(context)
        else:
            out_["context"].append(context)
    if images:
        out_["images"] = images if isinstance(images, list) else [images]
        out_["image_detail"] = image_detail or "low"

    if tool_schemas:
        out_["tool_schemas"] = tool_schemas

    if request_model:
        out_["request_model"] = request_model
        request_fields = break_down_pydantic_annotation(request_model)
        out_["context"].append({"respond_schema_info": request_model.model_json_schema()})

    if request_fields:
        _fields = request_fields if isinstance(request_fields, dict) else {}
        if not isinstance(request_fields, dict):
            _fields = {i: "..." for i in request_fields}
        out_["request_fields"] = _fields
        out_["request_response_format"] = prepare_request_response_format(request_fields=_fields)

    if plain_content:
        out_["plain_content"] = plain_content

    return {k: v for k, v in out_.items() if v is not None}


def format_instruction_content(content: dict) -> dict[str, Any]:
    """Format the content of the instruction."""
    text_content = format_text_content(content)
    if "images" not in content:
        return {"role": "user", "content": text_content}
    else:
        content_ = format_image_content(
            text_content=text_content,
            images=content["images"],
            image_detail=content["image_detail"],
        )
        return {"role": "user", "content": content_}
