# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
import json
import os
from pathlib import Path
from typing import Any


def config_list_from_json(
    env_or_file: str | Path,
    file_location: str | Path | None = "",
    filter_dict: dict[str, list[str | None] | set[str | None]] | None = None,
) -> list[dict[str, Any]]:
    """Retrieves a list of API configurations from a JSON stored in an environment variable or a file.

    This function attempts to parse JSON data from the given `env_or_file` parameter. If `env_or_file` is an
    environment variable containing JSON data, it will be used directly. Otherwise, it is assumed to be a filename,
    and the function will attempt to read the file from the specified `file_location`.

    The `filter_dict` parameter allows for filtering the configurations based on specified criteria. Each key in the
    `filter_dict` corresponds to a field in the configuration dictionaries, and the associated value is a list or set
    of acceptable values for that field. If a field is missing in a configuration and `None` is included in the list
    of acceptable values for that field, the configuration will still be considered a match.

    Args:
        env_or_file (str): The name of the environment variable, the filename, or the environment variable of the filename
            that containing the JSON data.
        file_location (str, optional): The directory path where the file is located, if `env_or_file` is a filename.
        filter_dict (dict, optional): A dictionary specifying the filtering criteria for the configurations, with
            keys representing field names and values being lists or sets of acceptable values for those fields.

    Example:
    ```python
    # Suppose we have an environment variable 'CONFIG_JSON' with the following content:
    # '[{"model": "gpt-3.5-turbo", "api_type": "azure"}, {"model": "gpt-4"}]'

    # We can retrieve a filtered list of configurations like this:
    filter_criteria = {"model": ["gpt-3.5-turbo"]}
    configs = config_list_from_json("CONFIG_JSON", filter_dict=filter_criteria)
    # The 'configs' variable will now contain only the configurations that match the filter criteria.
    ```

    Returns:
        List[Dict]: A list of configuration dictionaries that match the filtering criteria specified in `filter_dict`.

    Raises:
        FileNotFoundError: if env_or_file is neither found as an environment variable nor a file
    """
    env_str = os.environ.get(str(env_or_file))

    if env_str:
        # The environment variable exists. We should use information from it.
        if os.path.exists(env_str):  # noqa: SIM108
            # It is a file location, and we need to load the json from the file.
            json_str = Path(env_str).read_text()
        else:
            # Else, it should be a JSON string by itself.
            json_str = env_str
        config_list = json.loads(json_str)

    else:
        # The environment variable does not exist.
        # So, `env_or_file` is a filename. We should use the file location.
        config_list_path = Path(file_location) / env_or_file if file_location else Path(env_or_file)

        with open(config_list_path) as json_file:
            config_list = json.load(json_file)

    return filter_config(config_list, filter_dict)


def filter_config(
    config_list: list[dict[str, Any]],
    filter_dict: dict[str, list[str | None] | set[str | None]] | None,
    exclude: bool = False,
) -> list[dict[str, Any]]:
    """Filter configuration dictionaries based on specified criteria.

    This function filters a list of configuration dictionaries by applying ALL criteria specified in `filter_dict`.
    A configuration is included in the result if it satisfies every key-value constraint in the filter dictionary.
    For each filter key, the configuration's corresponding field value must match at least one of the acceptable
    values (OR logic within each criteria, AND logic between different criteria).

    Args:
        config_list (list of dict): A list of configuration dictionaries to be filtered.

        filter_dict (dict, optional): A dictionary specifying filter criteria where:
            - Keys are field names to check in each configuration dictionary
            - Values are lists/sets of acceptable values for that field
            - A configuration matches if ALL filter keys are satisfied AND for each key,
              the config's field value matches at least one acceptable value
            - If a filter value includes None, configurations missing that field will match
            - If None, no filtering is applied

        exclude (bool, optional): If False (default), return configurations that match the filter.
                                If True, return configurations that do NOT match the filter.

    Returns:
        list of dict: Filtered list of configuration dictionaries.

    Matching Logic:
        - **Between different filter keys**: AND logic (all criteria must be satisfied)
        - **Within each filter key's values**: OR logic (any acceptable value can match)
        - **For list-type config values**: Match if there's any intersection with acceptable values
        - **For scalar config values**: Match if the value is in the list of acceptable values
        - **Missing fields**: Only match if None is included in the acceptable values for that field

    Examples:
        ```python
        configs = [
            {"model": "gpt-3.5-turbo", "api_type": "openai"},
            {"model": "gpt-4", "api_type": "openai"},
            {"model": "gpt-3.5-turbo", "api_type": "azure", "api_version": "2024-02-01"},
            {"model": "gpt-4", "tags": ["premium", "latest"]},
        ]

        # Example 1: Single criterion - matches any model in the list
        filter_dict = {"model": ["gpt-4", "gpt-4o"]}
        result = filter_config(configs, filter_dict)
        # Returns: [{"model": "gpt-4", "api_type": "openai"}, {"model": "gpt-4", "tags": ["premium", "latest"]}]

        # Example 2: Multiple criteria - must satisfy ALL conditions
        filter_dict = {"model": ["gpt-3.5-turbo"], "api_type": ["azure"]}
        result = filter_config(configs, filter_dict)
        # Returns: [{"model": "gpt-3.5-turbo", "api_type": "azure", "api_version": "2024-02-01"}]

        # Example 3: Tag filtering with list intersection
        filter_dict = {"tags": ["premium"]}
        result = filter_config(configs, filter_dict)
        # Returns: [{"model": "gpt-4", "tags": ["premium", "latest"]}]

        # Example 4: Exclude matching configurations
        filter_dict = {"api_type": ["openai"]}
        result = filter_config(configs, filter_dict, exclude=True)
        # Returns configs that do NOT have api_type="openai"
        ```
    Note:
        - If `filter_dict` is empty or None, no filtering is applied and `config_list` is returned as is.
        - If a configuration dictionary in `config_list` does not contain a key specified in `filter_dict`,
          it is considered a non-match and is excluded from the result.

    """
    if filter_dict:
        return [
            item
            for item in config_list
            if all(_satisfies_criteria(item.get(key), values) != exclude for key, values in filter_dict.items())
        ]

    return config_list


def _satisfies_criteria(config_value: Any, criteria_values: Any) -> bool:
    """Check if a configuration field value satisfies the filter criteria.

    This helper function implements the matching logic between a single configuration
    field value and the acceptable values specified in the filter criteria. It handles
    both scalar and list-type configuration values with appropriate matching strategies.

    Args:
        config_value (Any): The value from a configuration dictionary field.
                           Can be None, a scalar value, or a list of values.
        criteria_values (Any): The acceptable values from the filter dictionary.
                              Can be a single value or a list/set of acceptable values.

    Returns:
        bool: True if the config_value satisfies the criteria, False otherwise.

    Matching Logic:
        - **None config values**: Always return False (missing fields don't match)
        - **List config values**:
            - If criteria is a list: Match if there's any intersection (set overlap)
            - If criteria is scalar: Match if the scalar is contained in the config list
        - **Scalar config values**:
            - If criteria is a list: Match if the config value is in the criteria list
            - If criteria is scalar: Match if the values are exactly equal

    Examples:
        ```python
        # List config value with list criteria (intersection matching)
        _satisfies_criteria(["gpt-4", "gpt-3.5"], ["gpt-4", "claude"])  # True (gpt-4 intersects)
        _satisfies_criteria(["tag1", "tag2"], ["tag3", "tag4"])  # False (no intersection)

        # List config value with scalar criteria (containment matching)
        _satisfies_criteria(["premium", "latest"], "premium")  # True (premium is in list)
        _satisfies_criteria(["tag1", "tag2"], "tag3")  # False (tag3 not in list)

        # Scalar config value with list criteria (membership matching)
        _satisfies_criteria("gpt-4", ["gpt-4", "gpt-3.5"])  # True (gpt-4 in criteria)
        _satisfies_criteria("claude", ["gpt-4", "gpt-3.5"])  # False (claude not in criteria)

        # Scalar config value with scalar criteria (equality matching)
        _satisfies_criteria("openai", "openai")  # True (exact match)
        _satisfies_criteria("openai", "azure")  # False (different values)

        # None config values (missing fields)
        _satisfies_criteria(None, ["gpt-4"])  # False (missing field)
        _satisfies_criteria(None, "gpt-4")  # False (missing field)
        ```

    Note:
        This is an internal helper function used by `filter_config()`. The function
        assumes that both parameters can be of various types and handles type
        checking internally to determine the appropriate matching strategy.
    """
    if config_value is None:
        return False

    if isinstance(config_value, list):
        if isinstance(criteria_values, list):
            return bool(set(config_value) & set(criteria_values))  # Non-empty intersection
        else:
            return criteria_values in config_value
    else:
        # In filter_dict, filter could be either a list of values or a single value.
        # For example, filter_dict = {"model": ["gpt-3.5-turbo"]} or {"model": "gpt-3.5-turbo"}
        if isinstance(criteria_values, list):
            return config_value in criteria_values
        return bool(config_value == criteria_values)
