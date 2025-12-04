# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

# Taken over from https://github.com/openai/openai-python/blob/main/src/openai/_models.py


import pydantic
import pydantic.generics
from pydantic import ConfigDict

__all__ = ["BaseModel"]


class BaseModel(pydantic.BaseModel):
    model_config = ConfigDict(extra="allow", defer_build=True)
