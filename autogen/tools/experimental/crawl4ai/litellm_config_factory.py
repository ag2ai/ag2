# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC

from ....oai import get_first_llm_config  # noqa


class LiteLLmConfigFactory(ABC):
    _factories: set["LiteLLmConfigFactory"] = set()
