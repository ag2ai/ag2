# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from https://github.com/lion-agi/lion-core are under the Apache-2.0 License
# SPDX-License-Identifier: Apache-2.0


from pydantic import BaseModel


def prepare_assistant_response(assistant_response: BaseModel | list[BaseModel] | dict | str) -> dict:
    """
    Prepare an assistant's response for storage and transmission.

    This function handles various input formats including:
    - Single model outputs (response.choices[0].message.content)
    - Streaming responses (response[i].choices[0].delta.content)
    - Direct content in dictionaries or strings

    Args:
        assistant_response: The response content in any supported format

    Returns:
        Note: Formatted response content
    """
    if assistant_response:
        content = {}
        # Handle model.choices[0].message.content format
        if isinstance(assistant_response, BaseModel):
            content["assistant_response"] = assistant_response.choices[0].message.content or ""
            content["model_response"] = assistant_response.model_dump(exclude_none=True, exclude_unset=True)
        # Handle streaming response[i].choices[0].delta.content format
        elif isinstance(assistant_response, list):
            msg = "".join([i.choices[0].delta.content or "" for i in assistant_response])
            content["assistant_response"] = msg
            content["model_response"] = [
                i.model_dump(
                    exclude_none=True,
                    exclude_unset=True,
                )
                for i in assistant_response
            ]
        elif isinstance(assistant_response, dict) and "content" in assistant_response:
            content["assistant_response"] = assistant_response["content"]
        elif isinstance(assistant_response, str):
            content["assistant_response"] = assistant_response
        else:
            content["assistant_response"] = str(assistant_response)
        return content
    else:
        return {"assistant_response": "", "model_response": ""}
