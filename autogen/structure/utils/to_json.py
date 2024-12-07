# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from https://github.com/lion-agi/lion-core are under the Apache-2.0 License
# SPDX-License-Identifier: Apache-2.0
import json
import re
from collections.abc import Callable, Iterable, Mapping
from typing import Any, Dict, List, Sequence, Union, overload

from pydantic_core import PydanticUndefinedType
from typing_extensions import Literal

from .xml_parser import xml_to_dict


def to_json(string: str | List[str], /, fuzzy_parse: bool = False) -> Union[List[Dict[str, Any]], Dict]:
    """Extract and parse JSON content from a string or markdown code blocks.

    This function attempts to parse JSON directly from the input string first.
    If that fails, it looks for JSON content within markdown code blocks
    (denoted by ```json).

    Args:
        string: Input string or list of strings to parse. If a list is provided,
               it will be joined with newlines.

    Returns:
        - A dictionary if a single JSON object is found
        - A list of dictionaries if multiple JSON objects are found
        - An empty list if no valid JSON is found

    Examples:
        >>> to_json('{"key": "value"}')
        {'key': 'value'}

        >>> to_json('''
        ... ```json
        ... {"key": "value"}
        ... ```
        ... ''')
        {'key': 'value'}

        >>> to_json('''
        ... ```json
        ... {"key1": "value1"}
        ... ```
        ... ```json
        ... {"key2": "value2"}
        ... ```
        ... ''')
        [{'key1': 'value1'}, {'key2': 'value2'}]
    """

    if isinstance(string, list):
        string = "\n".join(string)

    # Try direct JSON parsing first
    try:
        if fuzzy_parse:
            return fuzzy_parse_json(string)
        return json.loads(string)
    except Exception:
        pass

    # Look for JSON in markdown code blocks
    pattern = r"```json\s*(.*?)\s*```"
    matches = re.findall(pattern, string, re.DOTALL)

    if not matches:
        return []

    if len(matches) == 1:
        return json.loads(matches[0])

    if fuzzy_parse:
        return [fuzzy_parse_json(match) for match in matches]
    return [json.loads(match) for match in matches]


def fuzzy_parse_json(str_to_parse: str, /) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """Parse a JSON string with automatic fixing of common formatting issues.

    Args:
        str_to_parse: The JSON string to parse

    Returns:
        The parsed JSON object as a dictionary

    Raises:
        ValueError: If the string cannot be parsed as valid JSON
        TypeError: If the input is not a string or the result is not a dict
    """
    if not isinstance(str_to_parse, str):
        raise TypeError("Input must be a string")

    if not str_to_parse.strip():
        raise ValueError("Input string is empty")

    try:
        return json.loads(str_to_parse)
    except Exception:
        pass

    cleaned = _clean_json_string(str_to_parse)
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    try:
        fixed = fix_json_string(cleaned)
        return json.loads(fixed)
    except Exception as e:
        raise ValueError(f"Failed to parse JSON string after all fixing attempts: {e}") from e


def _clean_json_string(s: str) -> str:
    """Clean and standardize a JSON string."""
    s = re.sub(r"(?<!\\)'", '"', s)
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r'([{,])\s*([^"\s]+):', r'\1"\2":', s)
    return s.strip()


def fix_json_string(str_to_parse: str, /) -> str:
    """Fix a JSON string by ensuring all brackets are properly closed.

    Args:
        str_to_parse: JSON string to fix

    Returns:
        Fixed JSON string with proper bracket closure

    Raises:
        ValueError: If mismatched or extra closing brackets are found
    """
    if not str_to_parse:
        raise ValueError("Input string is empty")

    brackets = {"{": "}", "[": "]"}
    open_brackets = []
    pos = 0
    length = len(str_to_parse)

    while pos < length:
        char = str_to_parse[pos]

        # Handle escape sequences
        if char == "\\":
            pos += 2  # Skip escape sequence
            continue

        # Handle string content
        if char == '"':
            pos += 1
            # Skip until closing quote, accounting for escapes
            while pos < length:
                if str_to_parse[pos] == "\\":
                    pos += 2  # Skip escape sequence
                    continue
                if str_to_parse[pos] == '"':
                    break
                pos += 1
            pos += 1
            continue

        # Handle brackets
        if char in brackets:
            open_brackets.append(brackets[char])
        elif char in brackets.values():
            if not open_brackets:
                raise ValueError(f"Extra closing bracket '{char}' at position {pos}")
            if open_brackets[-1] != char:
                raise ValueError(f"Mismatched bracket '{char}' at position {pos}")
            open_brackets.pop()

        pos += 1

    # Add missing closing brackets
    closing_brackets = "".join(reversed(open_brackets))
    return str_to_parse + closing_brackets


@overload
def to_dict(input_: type[None] | PydanticUndefinedType, /) -> dict[str, Any]: ...


@overload
def to_dict(input_: Mapping, /) -> dict[str, Any]: ...


@overload
def to_dict(input_: set, /) -> dict[Any, Any]: ...


@overload
def to_dict(input_: Sequence, /) -> dict[str, Any]: ...


@overload
def to_dict(
    input_: Any,
    /,
    *,
    use_model_dump: bool = True,
    fuzzy_parse: bool = False,
    suppress: bool = False,
    str_type: Literal["json", "xml"] | None = "json",
    parser: Callable[[str], Any] | None = None,
    recursive: bool = False,
    max_recursive_depth: int = None,
    exclude_types: tuple = (),
    recursive_python_only: bool = True,
    **kwargs: Any,
) -> dict[str, Any]: ...


def to_dict(
    input_: Any,
    /,
    *,
    use_model_dump: bool = True,
    fuzzy_parse: bool = False,
    suppress: bool = False,
    str_type: Literal["json", "xml"] | None = "json",
    parser: Callable[[str], Any] | None = None,
    recursive: bool = False,
    max_recursive_depth: int = None,
    exclude_types: tuple = (),
    recursive_python_only: bool = True,
    **kwargs: Any,
):
    """
    Convert various input types to a dictionary, with optional recursive processing.

    Args:
        input_: The input to convert.
        use_model_dump: Use model_dump() for Pydantic models if available.
        fuzzy_parse: Use fuzzy parsing for string inputs.
        suppress: Return empty dict on errors if True.
        str_type: Input string type ("json" or "xml").
        parser: Custom parser function for string inputs.
        recursive: Enable recursive conversion of nested structures.
        max_recursive_depth: Maximum recursion depth (default 5, max 10).
        exclude_types: Tuple of types to exclude from conversion.
        recursive_python_only: If False, attempts to convert custom types recursively.
        **kwargs: Additional arguments for parsing functions.

    Returns:
        dict[str, Any]: A dictionary derived from the input.

    Raises:
        ValueError: If parsing fails and suppress is False.

    Examples:
        >>> to_dict({"a": 1, "b": [2, 3]})
        {'a': 1, 'b': [2, 3]}
        >>> to_dict('{"x": 10}', str_type="json")
        {'x': 10}
        >>> to_dict({"a": {"b": {"c": 1}}}, recursive=True, max_recursive_depth=2)
        {'a': {'b': {'c': 1}}}
    """
    try:
        if recursive:
            return recursive_to_dict(
                input_,
                use_model_dump=use_model_dump,
                fuzzy_parse=fuzzy_parse,
                str_type=str_type,
                parser=parser,
                max_recursive_depth=max_recursive_depth,
                exclude_types=exclude_types,
                recursive_custom_types=not recursive_python_only,
                **kwargs,
            )

        return _to_dict(
            input_,
            fuzzy_parse=fuzzy_parse,
            parser=parser,
            str_type=str_type,
            use_model_dump=use_model_dump,
            exclude_types=exclude_types,
            **kwargs,
        )
    except Exception as e:
        if suppress:
            return {}
        raise e


def _to_dict(
    input_: Any,
    /,
    *,
    use_model_dump: bool = True,
    fuzzy_parse: bool = False,
    str_type: Literal["json", "xml"] | None = "json",
    parser: Callable[[str], Any] | None = None,
    exclude_types: tuple = (),
    **kwargs: Any,
) -> dict[str, Any]:
    """Convert various input types to a dictionary.

    Handles multiple input types, including None, Mappings, strings, and more.

    Args:
        input_: The input to convert to a dictionary.
        use_model_dump: Use model_dump() for Pydantic models if available.
        fuzzy_parse: Use fuzzy parsing for string inputs.
        suppress: Return empty dict on parsing errors if True.
        str_type: Input string type, either "json" or "xml".
        parser: Custom parser function for string inputs.
        **kwargs: Additional arguments passed to parsing functions.

    Returns:
        A dictionary derived from the input.

    Raises:
        ValueError: If string parsing fails and suppress is False.

    Examples:
        >>> to_dict({"a": 1, "b": 2})
        {'a': 1, 'b': 2}
        >>> to_dict('{"x": 10}', str_type="json")
        {'x': 10}
        >>> to_dict("<root><a>1</a></root>", str_type="xml")
        {'a': '1'}
    """
    if isinstance(exclude_types, tuple) and len(exclude_types) > 0:
        if isinstance(input_, exclude_types):
            return input_

    if isinstance(input_, dict):
        return input_

    if use_model_dump and hasattr(input_, "model_dump"):
        return input_.model_dump(**kwargs)

    if isinstance(input_, type(None) | PydanticUndefinedType):
        return _undefined_to_dict(input_)
    if isinstance(input_, Mapping):
        return _mapping_to_dict(input_)

    if isinstance(input_, str):
        if fuzzy_parse:
            parser = fuzzy_parse_json
        try:
            a = _str_to_dict(
                input_,
                str_type=str_type,
                parser=parser,
                **kwargs,
            )
            if isinstance(a, dict):
                return a
        except Exception as e:
            raise ValueError("Failed to convert string to dictionary") from e

    if isinstance(input_, set):
        return _set_to_dict(input_)
    if isinstance(input_, Iterable):
        return _iterable_to_dict(input_)

    return _generic_type_to_dict(input_, **kwargs)


def _recursive_to_dict(
    input_: Any,
    /,
    *,
    max_recursive_depth: int,
    current_depth: int = 0,
    recursive_custom_types: bool = False,
    exclude_types: tuple = (),
    **kwargs: Any,
) -> Any:

    if current_depth >= max_recursive_depth:
        return input_

    if isinstance(input_, str):
        try:
            # Attempt to parse the string
            parsed = _to_dict(input_, **kwargs)
            # Recursively process the parsed result
            return _recursive_to_dict(
                parsed,
                max_recursive_depth=max_recursive_depth,
                current_depth=current_depth + 1,
                recursive_custom_types=recursive_custom_types,
                exclude_types=exclude_types,
                **kwargs,
            )
        except Exception:
            # Return the original string if parsing fails
            return input_

    elif isinstance(input_, dict):
        # Recursively process dictionary values
        return {
            key: _recursive_to_dict(
                value,
                max_recursive_depth=max_recursive_depth,
                current_depth=current_depth + 1,
                recursive_custom_types=recursive_custom_types,
                exclude_types=exclude_types,
                **kwargs,
            )
            for key, value in input_.items()
        }

    elif isinstance(input_, (list, tuple)):
        # Recursively process list or tuple elements
        processed = [
            _recursive_to_dict(
                element,
                max_recursive_depth=max_recursive_depth,
                current_depth=current_depth + 1,
                recursive_custom_types=recursive_custom_types,
                exclude_types=exclude_types,
                **kwargs,
            )
            for element in input_
        ]
        return type(input_)(processed)

    elif recursive_custom_types:
        # Process custom classes if enabled
        try:
            obj_dict = to_dict(input_, **kwargs)
            return _recursive_to_dict(
                obj_dict,
                max_recursive_depth=max_recursive_depth,
                current_depth=current_depth + 1,
                recursive_custom_types=recursive_custom_types,
                exclude_types=exclude_types,
                **kwargs,
            )
        except Exception:
            return input_

    else:
        # Return the input as is for other data types
        return input_


def recursive_to_dict(
    input_: Any,
    /,
    *,
    max_recursive_depth: int = None,
    exclude_types: tuple = (),
    recursive_custom_types: bool = False,
    **kwargs: Any,
) -> Any:

    if not isinstance(max_recursive_depth, int):
        max_recursive_depth = 5
    else:
        if max_recursive_depth < 0:
            raise ValueError("max_recursive_depth must be a non-negative integer")
        if max_recursive_depth == 0:
            return input_
        if max_recursive_depth > 10:
            raise ValueError("max_recursive_depth must be less than or equal to 10")

    return _recursive_to_dict(
        input_,
        max_recursive_depth=max_recursive_depth,
        current_depth=0,
        recursive_custom_types=recursive_custom_types,
        exclude_types=exclude_types,
        **kwargs,
    )


def _undefined_to_dict(
    input_: type[None] | PydanticUndefinedType,
    /,
) -> dict:
    return {}


def _mapping_to_dict(input_: Mapping, /) -> dict:
    return dict(input_)


def _str_to_dict(
    input_: str,
    /,
    *,
    str_type: Literal["json", "xml"] | None = "json",
    parser: Callable[[str], Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Handle string inputs."""
    if not input_:
        return {}

    if str_type == "json":
        try:
            return json.loads(input_, **kwargs) if parser is None else parser(input_, **kwargs)
        except json.JSONDecodeError as e:
            raise ValueError("Failed to parse JSON string") from e

    if str_type == "xml":
        try:
            if parser is None:
                return xml_to_dict(input_, **kwargs)
            return parser(input_, **kwargs)
        except Exception as e:
            raise ValueError("Failed to parse XML string") from e

    raise ValueError(f"Unsupported string type for `to_dict`: {str_type}, it should " "be 'json' or 'xml'.")


def _set_to_dict(input_: set, /) -> dict:
    return {value: value for value in input_}


def _iterable_to_dict(input_: Iterable, /) -> dict:
    return {idx: v for idx, v in enumerate(input_)}


def _generic_type_to_dict(
    input_,
    /,
    **kwargs: Any,
) -> dict[str, Any]:

    try:
        for method in ["to_dict", "dict", "json", "to_json"]:
            if hasattr(input_, method):
                result = getattr(input_, method)(**kwargs)
                return json.loads(result) if isinstance(result, str) else result
    except Exception:
        pass

    if hasattr(input_, "__dict__"):
        return input_.__dict__

    try:
        return dict(input_)
    except Exception as e:
        raise ValueError(f"Unable to convert input to dictionary: {e}")


__all__ = ["to_json", "fuzzy_parse_json", "to_dict"]
