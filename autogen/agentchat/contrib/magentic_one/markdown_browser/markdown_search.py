# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT

import logging
import os
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, cast
from urllib.parse import quote, quote_plus, unquote, urlparse, urlunparse

import requests

# TODO: Fix these types
from .mdconvert import MarkdownConverter  # type: ignore

logger = logging.getLogger(__name__)


class AbstractMarkdownSearch(ABC):
    """
    An abstract class for providing search capabilities to a Markdown browser.
    """

    @abstractmethod
    def search(self, query: str) -> str:
        pass


class BingMarkdownSearch(AbstractMarkdownSearch):
    """
    Provides Bing web search capabilities to Markdown browsers.
    """

    pass
