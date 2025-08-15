# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
"""Utilities for client classes"""

import logging
import warnings
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class FormatterProtocol(Protocol):
    """Structured Output classes with a format method"""

    def format(self) -> str: ...


def validate_parameter(
    params: dict[str, Any],
    param_name: str,
    allowed_types: tuple[Any, ...],
    allow_None: bool,  # noqa: N803
    default_value: Any,
    numerical_bound: tuple[float | None, float | None] | None,
    allowed_values: list[Any] | None,
) -> Any:
    """Validates a given config parameter, checking its type, values, and setting defaults
    Parameters:
        params (Dict[str, Any]): Dictionary containing parameters to validate.
        param_name (str): The name of the parameter to validate.
        allowed_types (Tuple): Tuple of acceptable types for the parameter.
        allow_None (bool): Whether the parameter can be `None`.
        default_value (Any): The default value to use if the parameter is invalid or missing.
        numerical_bound (Optional[Tuple[Optional[float], Optional[float]]]):
            A tuple specifying the lower and upper bounds for numerical parameters.
            Each bound can be `None` if not applicable.
        allowed_values (Optional[List[Any]]): A list of acceptable values for the parameter.
            Can be `None` if no specific values are required.

    Returns:
        Any: The validated parameter value or the default value if validation fails.

    Raises:
        TypeError: If `allowed_values` is provided but is not a list.

    Example Usage:
    ```python
        # Validating a numerical parameter within specific bounds
        params = {"temperature": 0.5, "safety_model": "Meta-Llama/Llama-Guard-7b"}
        temperature = validate_parameter(params, "temperature", (int, float), True, 0.7, (0, 1), None)
        # Result: 0.5

        # Validating a parameter that can be one of a list of allowed values
        model = validate_parameter(
        params, "safety_model", str, True, None, None, ["Meta-Llama/Llama-Guard-7b", "Meta-Llama/Llama-Guard-13b"]
        )
        # If "safety_model" is missing or invalid in params, defaults to "default"
    ```
    """
    if allowed_values is not None and not isinstance(allowed_values, list):
        raise TypeError(f"allowed_values should be a list or None, got {type(allowed_values).__name__}")

    param_value = params.get(param_name, default_value)
    warning = ""

    if param_value is None and allow_None:
        pass
    elif param_value is None:
        if not allow_None:
            warning = "cannot be None"
    elif not isinstance(param_value, allowed_types):
        # Check types and list possible types if invalid
        if isinstance(allowed_types, tuple):
            formatted_types = "(" + ", ".join(f"{t.__name__}" for t in allowed_types) + ")"
        else:
            formatted_types = f"{allowed_types.__name__}"
        warning = f"must be of type {formatted_types}{' or None' if allow_None else ''}"
    elif numerical_bound:
        # Check the value fits in possible bounds
        lower_bound, upper_bound = numerical_bound
        if (lower_bound is not None and param_value < lower_bound) or (
            upper_bound is not None and param_value > upper_bound
        ):
            warning = "has numerical bounds"
            if lower_bound is not None:
                warning += f", >= {lower_bound!s}"
            if upper_bound is not None:
                if lower_bound is not None:
                    warning += " and"
                warning += f" <= {upper_bound!s}"
            if allow_None:
                warning += ", or can be None"

    elif allowed_values:  # noqa: SIM102
        # Check if the value matches any allowed values
        if not (allow_None and param_value is None) and param_value not in allowed_values:
            warning = f"must be one of these values [{allowed_values}]{', or can be None' if allow_None else ''}"

    # If we failed any checks, warn and set to default value
    if warning:
        warnings.warn(
            f"Config error - {param_name} {warning}, defaulting to {default_value}.",
            UserWarning,
        )
        param_value = default_value

    return param_value


def should_hide_tools(messages: list[dict[str, Any]], tools: list[dict[str, Any]], hide_tools_param: str) -> bool:
    """Determines if tools should be hidden. This function is used to hide tools when they have been run, minimising the chance of the LLM choosing them when they shouldn't.
    Parameters:
        messages (List[Dict[str, Any]]): List of messages
        tools (List[Dict[str, Any]]): List of tools
        hide_tools_param (str): "hide_tools" parameter value. Can be "if_all_run" (hide tools if all tools have been run), "if_any_run" (hide tools if any of the tools have been run), "never" (never hide tools). Default is "never".

    Returns:
        bool: Indicates whether the tools should be excluded from the response create request

    Example Usage:
    ```python
        # Validating a numerical parameter within specific bounds
        messages = params.get("messages", [])
        tools = params.get("tools", None)
        hide_tools = should_hide_tools(messages, tools, params["hide_tools"])
    """
    if hide_tools_param == "never" or tools is None or len(tools) == 0:
        return False
    elif hide_tools_param == "if_any_run":
        # Return True if any tool_call_id exists, indicating a tool call has been executed. False otherwise.
        return any(["tool_call_id" in dictionary for dictionary in messages])
    elif hide_tools_param == "if_all_run":
        # Return True if all tools have been executed at least once. False otherwise.

        # Get the list of tool names
        check_tool_names = [item["function"]["name"] for item in tools]

        # Prepare a list of tool call ids and related function names
        tool_call_ids = {}

        # Loop through the messages and check if the tools have been run, removing them as we go
        for message in messages:
            if "tool_calls" in message:
                # Register the tool ids and the function names (there could be multiple tool calls)
                for tool_call in message["tool_calls"]:
                    tool_call_ids[tool_call["id"]] = tool_call["function"]["name"]
            elif "tool_call_id" in message:
                # Tool called, get the name of the function based on the id
                tool_name_called = tool_call_ids[message["tool_call_id"]]

                # If we had not yet called the tool, check and remove it to indicate we have
                if tool_name_called in check_tool_names:
                    check_tool_names.remove(tool_name_called)

        # Return True if all tools have been called at least once (accounted for)
        return len(check_tool_names) == 0
    else:
        raise TypeError(
            f"hide_tools_param is not a valid value ['if_all_run','if_any_run','never'], got '{hide_tools_param}'"
        )


def validate_openai_client(client, client_type: str = "OpenAI") -> None:
    """Validates an OpenAI-compatible client instance.

    This function provides standardized validation for OpenAI and OpenAI-compatible
    clients across all AG2 client implementations.

    Args:
        client: The client instance to validate
        client_type: Name of the client type for error messages (default: "OpenAI")

    Raises:
        ValueError: If the client is invalid or misconfigured

    Example Usage:
    ```python
    from autogen.oai.client_utils import validate_openai_client

    # Validate OpenAI client
    validate_openai_client(openai_client, "OpenAI")

    # Validate custom client
    validate_openai_client(custom_client, "CustomAPI")
    ```
    """
    if client is None:
        raise ValueError(f"{client_type} client cannot be None")

    # Validate API key is present
    if not hasattr(client, "api_key") or not client.api_key:
        raise ValueError(f"{client_type} client must have a valid API key")

    # Test basic client functionality with validation call if available
    try:
        # This will fail immediately if API key is invalid or client is malformed
        # We don't actually make the call, just validate the client can be configured
        if hasattr(client, "_validate") and callable(client._validate):
            client._validate()
    except Exception as e:
        raise ValueError(f"Invalid {client_type} client configuration: {str(e)}")


def standardize_api_error(error: Exception, client_type: str = "API", model_name: str = "unknown") -> ValueError:
    """Standardizes API error messages across different client implementations.

    This function provides consistent error handling and messaging for common API
    errors encountered across different LLM providers.

    Args:
        error: The original exception raised by the API
        client_type: Name of the client/API for error messages (default: "API")
        model_name: Name of the model being used (default: "unknown")

    Returns:
        ValueError: Standardized error with consistent messaging

    Example Usage:
    ```python
    from autogen.oai.client_utils import standardize_api_error

    try:
        response = client.create(**params)
    except Exception as e:
        raise standardize_api_error(e, "OpenAI Responses API", "gpt-4o")
    ```
    """
    error_message = str(error).lower()

    # Standardize common error patterns
    if "api key" in error_message or "unauthorized" in error_message:
        return ValueError(f"Invalid API key for {client_type}: {str(error)}")
    elif "model" in error_message and ("not found" in error_message or "does not exist" in error_message):
        return ValueError(f"Model '{model_name}' not found or not supported by {client_type}: {str(error)}")
    elif "rate limit" in error_message:
        return ValueError(f"API rate limit exceeded for {client_type}: {str(error)}")
    elif "insufficient" in error_message and "quota" in error_message:
        return ValueError(f"Insufficient API quota for {client_type}: {str(error)}")
    elif "timeout" in error_message:
        return ValueError(f"API request timeout for {client_type}: {str(error)}")
    elif "connection" in error_message or "network" in error_message:
        return ValueError(f"Network connection error for {client_type}: {str(error)}")
    else:
        # Generic error with context
        return ValueError(f"{client_type} error: {str(error)}")


# Logging format (originally from FLAML)
logging_formatter = logging.Formatter(
    "[%(name)s: %(asctime)s] {%(lineno)d} %(levelname)s - %(message)s", "%m-%d %H:%M:%S"
)
