# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
import textwrap

import pytest

import ag2.knowledge
import ag2.tools
from ag2.knowledge.redis import RedisKnowledgeStore
from ag2.tools.search import TavilySearchTool
from ag2.tools.toolkits import MCPToolkit

HEAVY_PACKAGES = ("ddgs", "mcp", "redis", "tavily")


def _run(code: str) -> str:
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def test_importing_ag2_loads_no_search_mcp_or_redis_client() -> None:
    loaded = _run(
        f"""
        import sys
        import ag2
        import ag2.tools
        import ag2.knowledge
        print(sorted(m for m in {HEAVY_PACKAGES!r} if m in sys.modules))
        """
    )

    assert loaded == "[]"


def test_mcp_client_loads_on_first_access() -> None:
    loaded = _run(
        """
        import sys
        from ag2.tools import MCPToolkit
        print("mcp" in sys.modules, "redis" in sys.modules)
        """
    )

    assert loaded == "True False"


@pytest.mark.parametrize(
    ("module", "name", "expected"),
    [
        (ag2.tools, "MCPToolkit", MCPToolkit),
        (ag2.tools, "TavilySearchTool", TavilySearchTool),
        (ag2.knowledge, "RedisKnowledgeStore", RedisKnowledgeStore),
    ],
)
def test_lazy_export_is_the_implementation(module: object, name: str, expected: object) -> None:
    assert getattr(module, name) is expected


@pytest.mark.parametrize("module", [ag2.tools, ag2.knowledge])
def test_every_export_resolves(module: object) -> None:
    for name in module.__all__:  # type: ignore[attr-defined]
        assert getattr(module, name) is not None


def test_unknown_name_raises_attribute_error() -> None:
    with pytest.raises(AttributeError, match="module 'ag2.tools' has no attribute 'NoSuchTool'"):
        ag2.tools.NoSuchTool  # noqa: B018
