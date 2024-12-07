# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from https://github.com/lion-agi/lion-core are under the Apache-2.0 License
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable, Sequence
from typing import Any, Literal

from .string_similarity import SIMILARITY_TYPE
from .to_json import to_dict, to_json
from .validate_keys import KeysDict, validate_keys


def validate_mapping(
    d: Any,
    keys: Sequence[str] | KeysDict,
    /,
    *,
    similarity_algo: SIMILARITY_TYPE | Callable[[str, str], float] = "jaro_winkler",
    similarity_threshold: float = 0.85,
    fuzzy_match: bool = True,
    handle_unmatched: Literal["ignore", "raise", "remove", "fill", "force"] = "ignore",
    fill_value: Any = None,
    fill_mapping: dict[str, Any] | None = None,
    strict: bool = False,
    suppress_conversion_errors: bool = False,
) -> dict[str, Any]:
    """
    Validate and correct any input into a dictionary with expected keys.

    Args:
        d: Input to validate. Can be:
            - Dictionary
            - JSON string or markdown code block
            - XML string
            - Object with to_dict/model_dump method
            - Any type convertible to dictionary
        keys: List of expected keys or dictionary mapping keys to types.
        similarity_algo: String similarity algorithm or custom function.
        similarity_threshold: Minimum similarity score for fuzzy matching.
        fuzzy_match: If True, use fuzzy matching for key correction.
        handle_unmatched: How to handle unmatched keys:
            - "ignore": Keep unmatched keys
            - "raise": Raise error for unmatched keys
            - "remove": Remove unmatched keys
            - "fill": Fill missing keys with default values
            - "force": Combine "fill" and "remove" behaviors
        fill_value: Default value for filling unmatched keys.
        fill_mapping: Dictionary mapping keys to default values.
        strict: Raise error if any expected key is missing.
        suppress_conversion_errors: Return empty dict on conversion errors.

    Returns:
        Validated and corrected dictionary.

    Raises:
        ValueError: If input cannot be converted or validation fails.
        TypeError: If input types are invalid.
    """
    if d is None:
        raise TypeError("Input cannot be None")

    # Try converting to dictionary
    try:
        if isinstance(d, str):
            # First try to_json for JSON strings and code blocks
            try:
                json_result = to_json(d)
                dict_input = json_result[0] if isinstance(json_result, list) else json_result
            except Exception:
                # Fall back to to_dict for other string formats
                dict_input = to_dict(d, str_type="json", fuzzy_parse=True, suppress=True)
        else:
            dict_input = to_dict(d, use_model_dump=True, fuzzy_parse=True, suppress=True)

        if not isinstance(dict_input, dict):
            if suppress_conversion_errors:
                dict_input = {}
            else:
                raise ValueError(f"Failed to convert input to dictionary: {type(dict_input)}")

    except Exception as e:
        if suppress_conversion_errors:
            dict_input = {}
        else:
            raise ValueError(f"Failed to convert input to dictionary: {e}")

    # Validate the dictionary
    return validate_keys(
        dict_input,
        keys,
        similarity_algo=similarity_algo,
        similarity_threshold=similarity_threshold,
        fuzzy_match=fuzzy_match,
        handle_unmatched=handle_unmatched,
        fill_value=fill_value,
        fill_mapping=fill_mapping,
        strict=strict,
    )


__all__ = ["validate_mapping"]
