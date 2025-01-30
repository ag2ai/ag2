# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT


import os
import subprocess

import pytest

from .conftest import Credentials, Secrets, credentials_all_llms, suppress_gemini_resource_exhausted


@pytest.mark.parametrize("credentials_from_test_param", credentials_all_llms, indirect=True)
@suppress_gemini_resource_exhausted
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


class TestSecrets:
    def test_sanitize_secrets(self):
        Secrets.add_secret("mysecret")
        data = "This contains mysecret and ysecre and somemysecreand should be sanitized."
        sanitized = Secrets.sanitize_secrets(data)
        assert sanitized == "This contains ***** and ***** and some*****and should be sanitized."

    @pytest.mark.skipif(
        not os.getenv("RUN_SANITIZATION_TEST"),
        reason="Skipping sensitive tests. Set RUN_SANITIZATION_TEST=1 to run.",
    )
    def test_raise_exception_with_secret(self):
        Secrets.add_secret("mysecret")
        raise Exception("This is a test exception. mysecret exposed!!!")

    def test_sensitive_output_is_sanitized(self):
        # Run pytest for the sensitive tests and capture the output
        result = subprocess.run(
            [
                "pytest",
                "-s",
                "test/test_conftest.py::TestSecrets::test_raise_exception_with_secret",
            ],
            env={**os.environ, "RUN_SANITIZATION_TEST": "1"},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Combine stdout and stderr to search for secrets
        output = result.stdout + result.stderr

        assert "mysecret" not in output, "Secret exposed in test output!"
        assert "*****" in output, "Sanitization is not working as expected!"
