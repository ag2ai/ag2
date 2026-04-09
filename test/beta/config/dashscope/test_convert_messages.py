# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from autogen.beta.config.dashscope.mappers import convert_messages
from autogen.beta.events import ImageInput
from autogen.beta.exceptions import UnsupportedInputError


def test_image_input_raises() -> None:
    with pytest.raises(UnsupportedInputError, match="ImageInput.*dashscope"):
        convert_messages([], [ImageInput(url="https://example.com/img.png")])
