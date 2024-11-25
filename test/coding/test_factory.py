# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
import pytest

from autogen.coding.factory import CodeExecutorFactory


def test_create_unknown() -> None:
    config = {"executor": "unknown"}
    with pytest.raises(ValueError, match="Unknown code executor unknown"):
        CodeExecutorFactory.create(config)

    config = {}
    with pytest.raises(ValueError, match="Unknown code executor None"):
        CodeExecutorFactory.create(config)
