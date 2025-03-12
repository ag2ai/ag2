# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/https://github.com/Lancetnik/FastDepends are under the MIT License.
# SPDX-License-Identifier: MIT

import pytest

from autogen.fast_depends._compat import PYDANTIC_V2

pydanticV1 = pytest.mark.skipif(PYDANTIC_V2, reason="requires PydanticV2")  # noqa: N816

pydanticV2 = pytest.mark.skipif(not PYDANTIC_V2, reason="requires PydanticV1")  # noqa: N816
