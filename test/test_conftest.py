# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT


import pytest

from .conftest import Credentials, credentials_all_llms


@pytest.mark.parametrize("credentials_from_test_param", credentials_all_llms, indirect=True)
def test_credentials_from_test_param_fixture(
    credentials_from_test_param: Credentials,
    request: pytest.FixtureRequest,
) -> None:
    # Get the parameter name request node
    current_llm = request.node.callspec.id

    assert current_llm is not None
    assert isinstance(credentials_from_test_param, Credentials)

    first_config = credentials_from_test_param.config_list[0]
    if "gpt_4" in current_llm:
        if "api_type" in first_config:
            assert first_config["api_type"] == "openai"
    elif "gemini" in current_llm:
        assert first_config["api_type"] == "google"
    elif "anthropic" in current_llm:
        assert first_config["api_type"] == "anthropic"
    else:
        assert False, f"Unknown LLM fixture: {current_llm}"
