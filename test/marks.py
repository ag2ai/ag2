# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT

import pytest

credentials_all_llms = [
    pytest.param(
        "credentials_gpt_4o_mini",
        marks=[pytest.mark.openai, pytest.mark.aux_neg_flag],
    ),
    pytest.param(
        "credentials_gemini_flash",
        marks=[pytest.mark.gemini, pytest.mark.aux_neg_flag],
    ),
    pytest.param(
        "credentials_anthropic_claude_sonnet",
        marks=[pytest.mark.anthropic, pytest.mark.aux_neg_flag],
    ),
]

credentials_browser_use = [
    pytest.param(
        "credentials_gpt_4o_mini",
        marks=[pytest.mark.openai, pytest.mark.aux_neg_flag],
    ),
    pytest.param(
        "credentials_anthropic_claude_sonnet",
        marks=[pytest.mark.anthropic, pytest.mark.aux_neg_flag],
    ),
    pytest.param(
        "credentials_gemini_flash_exp",
        marks=[pytest.mark.gemini, pytest.mark.aux_neg_flag],
    ),
    # Deeseek currently does not work too well with the browser-use
    pytest.param(
        "credentials_deepseek_chat",
        marks=[pytest.mark.deepseek, pytest.mark.aux_neg_flag],
    ),
]
