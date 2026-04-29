# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from dirty_equals import IsPartialDict

pytest.importorskip("perplexity")

from autogen.beta import Agent, DataInput, ImageInput, Variable
from autogen.beta.context import ConversationContext
from autogen.beta.events import ModelResponse, ToolCallEvent, ToolCallsEvent, ToolResultsEvent
from autogen.beta.testing import TestConfig, TrackingConfig
from autogen.beta.tools.search.perplexity import ImageMeta, PerplexitySearchTool, SearchResponse, SearchResult

SAMPLE_RAW = SimpleNamespace(
    id="chatcmpl-xxx",
    model="sonar",
    created=1700000000,
    choices=[
        SimpleNamespace(
            index=0,
            finish_reason="stop",
            message=SimpleNamespace(
                role="assistant",
                content="AG2 is an open-source multi-agent framework.",
            ),
        ),
    ],
    search_results=[
        SimpleNamespace(
            title="AG2 Framework",
            url="https://ag2.ai",
            date="2026-01-01",
            snippet="AG2 is an agent framework.",
        ),
        SimpleNamespace(
            title="GitHub - AG2",
            url="https://github.com/ag2ai/ag2",
            date=None,
            snippet=None,
        ),
    ],
    citations=["https://ag2.ai", "https://github.com/ag2ai/ag2"],
)


def _make_config(query: str, final_reply: str = "done", tool_name: str = "perplexity_search") -> TestConfig:
    call = ToolCallEvent(arguments=json.dumps({"query": query}), name=tool_name)
    return TestConfig(ModelResponse(tool_calls=ToolCallsEvent([call])), final_reply)


def _empty_response() -> SimpleNamespace:
    return SimpleNamespace(
        id="chatcmpl-empty",
        model="sonar",
        created=0,
        choices=[
            SimpleNamespace(
                index=0,
                finish_reason="stop",
                message=SimpleNamespace(role="assistant", content=""),
            ),
        ],
        search_results=None,
        citations=None,
    )


@pytest.mark.asyncio
class TestSchema:
    async def test_default_schema(self, context: ConversationContext) -> None:
        perp = PerplexitySearchTool(client=MagicMock())

        [schema] = await perp.schemas(context)

        assert schema.function.name == "perplexity_search"
        assert schema.function.parameters == IsPartialDict({
            "required": ["query"],
            "properties": IsPartialDict({"query": IsPartialDict({"type": "string"})}),
        })

    async def test_custom_schema(self, context: ConversationContext) -> None:
        perp = PerplexitySearchTool(client=MagicMock(), name="my_search", description="Custom search tool.")

        [schema] = await perp.schemas(context)

        assert schema.function.name == "my_search"
        assert schema.function.description == "Custom search tool."
        assert schema.function.parameters == IsPartialDict({
            "required": ["query"],
            "properties": IsPartialDict({"query": IsPartialDict({"type": "string"})}),
        })


@pytest.mark.asyncio
class TestSearchExecution:
    async def test_search_returns_structured_results(self, mock: MagicMock) -> None:
        # arrange tool
        mock.chat.completions.create.return_value = SAMPLE_RAW
        perp = PerplexitySearchTool(client=mock)

        # arrange agent
        config = TrackingConfig(_make_config("AG2 framework"))
        agent = Agent("a", config=config, tools=[perp])

        # act
        await agent.ask("search")

        # assert: framework forwards a single ToolResult with a single DataInput payload
        tool_results_event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        [tool_result] = tool_results_event.results
        [part] = tool_result.result.parts
        assert part == DataInput(
            SearchResponse(
                query="AG2 framework",
                content="AG2 is an open-source multi-agent framework.",
                search_results=[
                    SearchResult(
                        title="AG2 Framework",
                        url="https://ag2.ai",
                        date="2026-01-01",
                        snippet="AG2 is an agent framework.",
                    ),
                    SearchResult(
                        title="GitHub - AG2",
                        url="https://github.com/ag2ai/ag2",
                        date=None,
                        snippet=None,
                    ),
                ],
                citations=["https://ag2.ai", "https://github.com/ag2ai/ag2"],
            )
        )

    async def test_search_empty_results(self, mock: MagicMock) -> None:
        # arrange tool
        mock.chat.completions.create.return_value = _empty_response()
        perp = PerplexitySearchTool(client=mock)

        # arrange agent
        config = TrackingConfig(_make_config("nothing"))
        agent = Agent("a", config=config, tools=[perp])

        # act
        await agent.ask("search")

        # assert
        tool_results_event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        [tool_result] = tool_results_event.results
        [part] = tool_result.result.parts
        assert part == DataInput(
            SearchResponse(
                query="nothing",
                content="",
                search_results=[],
                citations=[],
            )
        )

    async def test_all_params_forwarded_to_client(self, mock: MagicMock) -> None:
        mock.chat.completions.create.return_value = _empty_response()

        perp = PerplexitySearchTool(
            client=mock,
            model="sonar-pro",
            max_tokens=2000,
            search_domain_filter=["arxiv.org", "-medium.com"],
            search_context_size="medium",
            search_mode="academic",
            search_recency_filter="week",
            return_images=True,
            return_related_questions=True,
        )
        agent = Agent("a", config=_make_config("q"), tools=[perp])

        await agent.ask("search")

        mock.chat.completions.create.assert_called_once_with(
            model="sonar-pro",
            messages=[
                {"role": "system", "content": "Be precise and concise."},
                {"role": "user", "content": "q"},
            ],
            max_tokens=2000,
            web_search_options={"search_context_size": "medium"},
            search_domain_filter=["arxiv.org", "-medium.com"],
            search_mode="academic",
            search_recency_filter="week",
            return_images=True,
            return_related_questions=True,
        )

    async def test_defaults_applied_when_params_omitted(self, mock: MagicMock) -> None:
        # When optional params aren't set, defaults (sonar/1000/high) are used and
        # other Perplexity-specific kwargs are NOT forwarded so the API uses its own defaults.
        mock.chat.completions.create.return_value = _empty_response()

        perp = PerplexitySearchTool(client=mock)
        agent = Agent("a", config=_make_config("q"), tools=[perp])

        await agent.ask("search")

        mock.chat.completions.create.assert_called_once_with(
            model="sonar",
            messages=[
                {"role": "system", "content": "Be precise and concise."},
                {"role": "user", "content": "q"},
            ],
            max_tokens=1000,
            web_search_options={"search_context_size": "high"},
        )

    async def test_custom_tool_name_in_agent(self, mock: MagicMock) -> None:
        # arrange tool
        mock.chat.completions.create.return_value = _empty_response()
        perp = PerplexitySearchTool(client=mock, name="web_search")

        # arrange agent
        config = TrackingConfig(_make_config("AG2 framework", tool_name="web_search"))
        agent = Agent("a", config=config, tools=[perp])

        # act
        await agent.ask("search")

        # assert tool was actually invoked under the custom name
        mock.chat.completions.create.assert_called_once()

    async def test_returns_image_parts_when_api_yields_images(self, mock: MagicMock) -> None:
        # The Perplexity SDK delivers `images` as plain dicts (the field is not
        # declared on StreamChunk's pydantic model). Mirror that here.
        raw = SimpleNamespace(
            id="chatcmpl-img",
            model="sonar",
            created=0,
            choices=[
                SimpleNamespace(
                    index=0,
                    finish_reason="stop",
                    message=SimpleNamespace(role="assistant", content="See attached images."),
                ),
            ],
            search_results=None,
            citations=None,
            images=[
                {
                    "image_url": "https://example.com/a.jpg",
                    "origin_url": "https://example.com/a",
                    "title": "Image A",
                    "width": 800,
                    "height": 600,
                },
                {
                    "image_url": "https://example.com/b.png",
                    "origin_url": "https://example.com/b",
                    "title": "Image B",
                    "width": 1024,
                    "height": 768,
                },
            ],
        )
        mock.chat.completions.create.return_value = raw
        perp = PerplexitySearchTool(client=mock, return_images=True)

        # arrange agent
        config = TrackingConfig(_make_config("show me images"))
        agent = Agent("a", config=config, tools=[perp])

        # act
        await agent.ask("search")

        # assert: parts == [DataInput(SearchResponse w/ image metadata), UrlInput(IMAGE), UrlInput(IMAGE)]
        tool_results_event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        [tool_result] = tool_results_event.results
        data_part, image_a, image_b = tool_result.result.parts

        assert data_part == DataInput(
            SearchResponse(
                query="show me images",
                content="See attached images.",
                search_results=[],
                citations=[],
                images=[
                    ImageMeta(
                        image_url="https://example.com/a.jpg",
                        origin_url="https://example.com/a",
                        title="Image A",
                        width=800,
                        height=600,
                    ),
                    ImageMeta(
                        image_url="https://example.com/b.png",
                        origin_url="https://example.com/b",
                        title="Image B",
                        width=1024,
                        height=768,
                    ),
                ],
            )
        )
        assert image_a == ImageInput(url="https://example.com/a.jpg")
        assert image_b == ImageInput(url="https://example.com/b.png")

    async def test_skips_image_entries_without_image_url(self, mock: MagicMock) -> None:
        # API may return image objects without an image_url — those should be silently dropped.
        raw = SimpleNamespace(
            id="chatcmpl-img",
            model="sonar",
            created=0,
            choices=[
                SimpleNamespace(
                    index=0,
                    finish_reason="stop",
                    message=SimpleNamespace(role="assistant", content=""),
                ),
            ],
            search_results=None,
            citations=None,
            images=[
                {"origin_url": None, "title": None, "width": None, "height": None},
                {"image_url": "https://example.com/ok.jpg"},
            ],
        )
        mock.chat.completions.create.return_value = raw
        perp = PerplexitySearchTool(client=mock, return_images=True)

        config = TrackingConfig(_make_config("q"))
        agent = Agent("a", config=config, tools=[perp])

        await agent.ask("search")

        tool_results_event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        [tool_result] = tool_results_event.results
        data_part, image = tool_result.result.parts
        assert image == ImageInput(url="https://example.com/ok.jpg")


@pytest.mark.asyncio
class TestPerplexitySearchToolVariable:
    async def test_resolved(self, mock: MagicMock) -> None:
        # arrange tool
        mock.chat.completions.create.return_value = _empty_response()
        perp = PerplexitySearchTool(
            client=mock,
            model=Variable("user_model"),
            search_recency_filter=Variable(),
        )

        # arrange agent with variables
        agent = Agent(
            "a",
            config=_make_config("test query"),
            tools=[perp],
            variables={
                "user_model": "sonar-pro",
                "search_recency_filter": "day",
            },
        )

        # act
        await agent.ask("search")

        # assert variables were resolved into the SDK call
        mock.chat.completions.create.assert_called_once_with(
            model="sonar-pro",
            messages=[
                {"role": "system", "content": "Be precise and concise."},
                {"role": "user", "content": "test query"},
            ],
            max_tokens=1000,
            web_search_options={"search_context_size": "high"},
            search_recency_filter="day",
        )

    async def test_missing_raises(self, mock: MagicMock) -> None:
        mock.chat.completions.create.return_value = _empty_response()
        perp = PerplexitySearchTool(
            client=mock,
            search_mode=Variable(),
        )

        agent = Agent("a", config=_make_config("test query"), tools=[perp])

        with pytest.raises(KeyError, match="search_mode"):
            await agent.ask("search")
