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
from autogen.beta.tools.search.perplexity import (
    PerplexityImageMeta,
    PerplexitySearchResponse,
    PerplexitySearchResult,
    PerplexitySearchToolkit,
)


def _tool_call_config(
    arguments: dict,
    *,
    tool_name: str,
    final_reply: str = "done",
) -> TestConfig:
    return TestConfig(
        ModelResponse(
            tool_calls=ToolCallsEvent([
                ToolCallEvent(arguments=json.dumps(arguments), name=tool_name),
            ]),
        ),
        final_reply,
    )


SEARCH_API_RAW = SimpleNamespace(
    id="search-xxx",
    results=[
        SimpleNamespace(
            title="AG2 Framework",
            url="https://ag2.ai",
            snippet="AG2 is an agent framework.",
            date="2026-01-01",
        ),
        SimpleNamespace(
            title="GitHub - AG2",
            url="https://github.com/ag2ai/ag2",
            snippet="Open source repo.",
            date=None,
        ),
    ],
)


def _empty_search_api_response() -> SimpleNamespace:
    return SimpleNamespace(id="search-empty", results=[])


CHAT_RAW = SimpleNamespace(
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


def _empty_chat_response() -> SimpleNamespace:
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
    async def test_default_schemas(self, context: ConversationContext) -> None:
        toolkit = PerplexitySearchToolkit(client=MagicMock())

        schemas = list(await toolkit.schemas(context))

        names = [s.function.name for s in schemas]
        assert names == ["perplexity_search", "perplexity_answer"]

    async def test_search_schema_has_query_param(self, context: ConversationContext) -> None:
        toolkit = PerplexitySearchToolkit(client=MagicMock())

        schemas = list(await toolkit.schemas(context))
        search_schema = next(s for s in schemas if s.function.name == "perplexity_search")

        assert search_schema.function.parameters == IsPartialDict({
            "required": ["query"],
            "properties": IsPartialDict({"query": IsPartialDict({"type": "string"})}),
        })

    async def test_answer_schema_has_query_param(self, context: ConversationContext) -> None:
        toolkit = PerplexitySearchToolkit(client=MagicMock())

        schemas = list(await toolkit.schemas(context))
        answer_schema = next(s for s in schemas if s.function.name == "perplexity_answer")

        assert answer_schema.function.parameters == IsPartialDict({
            "required": ["query"],
            "properties": IsPartialDict({"query": IsPartialDict({"type": "string"})}),
        })

    async def test_custom_search_name_and_description(self, context: ConversationContext) -> None:
        toolkit = PerplexitySearchToolkit(client=MagicMock())
        custom = toolkit.search(name="web_search", description="Custom search.")

        [schema] = list(await custom.schemas(context))

        assert schema.function.name == "web_search"
        assert schema.function.description == "Custom search."

    async def test_custom_answer_name_and_description(self, context: ConversationContext) -> None:
        toolkit = PerplexitySearchToolkit(client=MagicMock())
        custom = toolkit.answer(name="ask_perplexity", description="Custom answer.")

        [schema] = list(await custom.schemas(context))

        assert schema.function.name == "ask_perplexity"
        assert schema.function.description == "Custom answer."


@pytest.mark.asyncio
class TestSearch:
    async def test_returns_structured_results(self, mock: MagicMock) -> None:
        mock.search.create.return_value = SEARCH_API_RAW
        toolkit = PerplexitySearchToolkit(client=mock)

        config = TrackingConfig(_tool_call_config({"query": "AG2 framework"}, tool_name="perplexity_search"))
        agent = Agent("a", config=config, tools=[toolkit])

        await agent.ask("search")

        tool_results_event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        [tool_result] = tool_results_event.results
        [part] = tool_result.result.parts
        assert part == DataInput(
            PerplexitySearchResponse(
                query="AG2 framework",
                results=[
                    PerplexitySearchResult(
                        title="AG2 Framework",
                        url="https://ag2.ai",
                        snippet="AG2 is an agent framework.",
                        date="2026-01-01",
                    ),
                    PerplexitySearchResult(
                        title="GitHub - AG2",
                        url="https://github.com/ag2ai/ag2",
                        snippet="Open source repo.",
                        date=None,
                    ),
                ],
            )
        )

    async def test_empty_results(self, mock: MagicMock) -> None:
        mock.search.create.return_value = _empty_search_api_response()
        toolkit = PerplexitySearchToolkit(client=mock)

        config = TrackingConfig(_tool_call_config({"query": "nothing"}, tool_name="perplexity_search"))
        agent = Agent("a", config=config, tools=[toolkit])

        await agent.ask("search")

        tool_results_event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        [tool_result] = tool_results_event.results
        [part] = tool_result.result.parts
        assert part == DataInput(PerplexitySearchResponse(query="nothing", results=[]))

    async def test_none_params_omitted(self, mock: MagicMock) -> None:
        mock.search.create.return_value = _empty_search_api_response()
        toolkit = PerplexitySearchToolkit(client=mock)

        agent = Agent("a", config=_tool_call_config({"query": "q"}, tool_name="perplexity_search"), tools=[toolkit])

        await agent.ask("search")

        mock.search.create.assert_called_once_with(query="q")

    async def test_all_params_forwarded(self, mock: MagicMock) -> None:
        mock.search.create.return_value = _empty_search_api_response()
        toolkit = PerplexitySearchToolkit(client=mock)
        search_tool = toolkit.search(
            max_results=5,
            max_tokens_per_page=512,
            search_domain_filter=["arxiv.org", "-medium.com"],
            search_recency_filter="week",
            search_after_date_filter="1/1/2025",
            search_before_date_filter="12/31/2025",
        )

        agent = Agent(
            "a",
            config=_tool_call_config({"query": "q"}, tool_name="perplexity_search"),
            tools=[search_tool],
        )
        await agent.ask("search")

        mock.search.create.assert_called_once_with(
            query="q",
            max_results=5,
            max_tokens_per_page=512,
            search_domain_filter=["arxiv.org", "-medium.com"],
            search_recency_filter="week",
            search_after_date_filter="1/1/2025",
            search_before_date_filter="12/31/2025",
        )

    async def test_custom_tool_name_in_agent(self, mock: MagicMock) -> None:
        mock.search.create.return_value = _empty_search_api_response()
        toolkit = PerplexitySearchToolkit(client=mock)
        search_tool = toolkit.search(name="web_search")

        agent = Agent(
            "a",
            config=_tool_call_config({"query": "q"}, tool_name="web_search"),
            tools=[search_tool],
        )
        await agent.ask("search")

        mock.search.create.assert_called_once()


@pytest.mark.asyncio
class TestAnswer:
    async def test_returns_structured_results(self, mock: MagicMock) -> None:
        mock.chat.completions.create.return_value = CHAT_RAW
        toolkit = PerplexitySearchToolkit(client=mock)

        config = TrackingConfig(_tool_call_config({"query": "AG2 framework"}, tool_name="perplexity_answer"))
        agent = Agent("a", config=config, tools=[toolkit])

        await agent.ask("answer")

        tool_results_event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        [tool_result] = tool_results_event.results
        [part] = tool_result.result.parts
        assert part == DataInput(
            PerplexitySearchResponse(
                query="AG2 framework",
                results=[
                    PerplexitySearchResult(
                        title="AG2 Framework",
                        url="https://ag2.ai",
                        snippet="AG2 is an agent framework.",
                        date="2026-01-01",
                    ),
                    PerplexitySearchResult(
                        title="GitHub - AG2",
                        url="https://github.com/ag2ai/ag2",
                        snippet=None,
                        date=None,
                    ),
                ],
                content="AG2 is an open-source multi-agent framework.",
                citations=["https://ag2.ai", "https://github.com/ag2ai/ag2"],
            )
        )

    async def test_empty_results(self, mock: MagicMock) -> None:
        mock.chat.completions.create.return_value = _empty_chat_response()
        toolkit = PerplexitySearchToolkit(client=mock)

        config = TrackingConfig(_tool_call_config({"query": "nothing"}, tool_name="perplexity_answer"))
        agent = Agent("a", config=config, tools=[toolkit])

        await agent.ask("answer")

        tool_results_event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        [tool_result] = tool_results_event.results
        [part] = tool_result.result.parts
        assert part == DataInput(
            PerplexitySearchResponse(
                query="nothing",
                results=[],
                content="",
                citations=[],
            )
        )

    async def test_defaults_applied_when_params_omitted(self, mock: MagicMock) -> None:
        mock.chat.completions.create.return_value = _empty_chat_response()
        toolkit = PerplexitySearchToolkit(client=mock)

        agent = Agent("a", config=_tool_call_config({"query": "q"}, tool_name="perplexity_answer"), tools=[toolkit])

        await agent.ask("answer")

        mock.chat.completions.create.assert_called_once_with(
            model="sonar",
            messages=[
                {"role": "system", "content": "Be precise and concise."},
                {"role": "user", "content": "q"},
            ],
            max_tokens=1000,
            web_search_options={"search_context_size": "high"},
        )

    async def test_all_params_forwarded(self, mock: MagicMock) -> None:
        mock.chat.completions.create.return_value = _empty_chat_response()
        toolkit = PerplexitySearchToolkit(client=mock)
        answer_tool = toolkit.answer(
            model="sonar-pro",
            max_tokens=2000,
            search_domain_filter=["arxiv.org", "-medium.com"],
            search_context_size="medium",
            search_mode="academic",
            search_recency_filter="week",
            return_images=True,
            return_related_questions=True,
        )

        agent = Agent(
            "a",
            config=_tool_call_config({"query": "q"}, tool_name="perplexity_answer"),
            tools=[answer_tool],
        )
        await agent.ask("answer")

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

    async def test_custom_tool_name_in_agent(self, mock: MagicMock) -> None:
        mock.chat.completions.create.return_value = _empty_chat_response()
        toolkit = PerplexitySearchToolkit(client=mock)
        answer_tool = toolkit.answer(name="ask_perplexity")

        agent = Agent(
            "a",
            config=_tool_call_config({"query": "q"}, tool_name="ask_perplexity"),
            tools=[answer_tool],
        )
        await agent.ask("answer")

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
        toolkit = PerplexitySearchToolkit(client=mock)
        answer_tool = toolkit.answer(return_images=True)

        config = TrackingConfig(_tool_call_config({"query": "show me images"}, tool_name="perplexity_answer"))
        agent = Agent("a", config=config, tools=[answer_tool])

        await agent.ask("answer")

        tool_results_event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        [tool_result] = tool_results_event.results
        data_part, image_a, image_b = tool_result.result.parts

        assert data_part == DataInput(
            PerplexitySearchResponse(
                query="show me images",
                results=[],
                content="See attached images.",
                citations=[],
                images=[
                    PerplexityImageMeta(
                        image_url="https://example.com/a.jpg",
                        origin_url="https://example.com/a",
                        title="Image A",
                        width=800,
                        height=600,
                    ),
                    PerplexityImageMeta(
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
        toolkit = PerplexitySearchToolkit(client=mock)
        answer_tool = toolkit.answer(return_images=True)

        config = TrackingConfig(_tool_call_config({"query": "q"}, tool_name="perplexity_answer"))
        agent = Agent("a", config=config, tools=[answer_tool])

        await agent.ask("answer")

        tool_results_event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        [tool_result] = tool_results_event.results
        _data_part, image = tool_result.result.parts
        assert image == ImageInput(url="https://example.com/ok.jpg")


@pytest.mark.asyncio
class TestSearchVariable:
    async def test_resolved(self, mock: MagicMock) -> None:
        mock.search.create.return_value = _empty_search_api_response()
        toolkit = PerplexitySearchToolkit(client=mock)
        search_tool = toolkit.search(
            max_results=Variable("user_max"),
            search_recency_filter=Variable(),
        )

        agent = Agent(
            "a",
            config=_tool_call_config({"query": "test query"}, tool_name="perplexity_search"),
            tools=[search_tool],
            variables={
                "user_max": 7,
                "search_recency_filter": "day",
            },
        )

        await agent.ask("search")

        mock.search.create.assert_called_once_with(
            query="test query",
            max_results=7,
            search_recency_filter="day",
        )

    async def test_missing_raises(self, mock: MagicMock) -> None:
        mock.search.create.return_value = _empty_search_api_response()
        toolkit = PerplexitySearchToolkit(client=mock)
        search_tool = toolkit.search(max_results=Variable())

        agent = Agent(
            "a",
            config=_tool_call_config({"query": "test query"}, tool_name="perplexity_search"),
            tools=[search_tool],
        )

        with pytest.raises(KeyError, match="max_results"):
            await agent.ask("search")


@pytest.mark.asyncio
class TestAnswerVariable:
    async def test_resolved(self, mock: MagicMock) -> None:
        mock.chat.completions.create.return_value = _empty_chat_response()
        toolkit = PerplexitySearchToolkit(client=mock)
        answer_tool = toolkit.answer(
            model=Variable("user_model"),
            search_recency_filter=Variable(),
        )

        agent = Agent(
            "a",
            config=_tool_call_config({"query": "test query"}, tool_name="perplexity_answer"),
            tools=[answer_tool],
            variables={
                "user_model": "sonar-pro",
                "search_recency_filter": "day",
            },
        )

        await agent.ask("answer")

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
        mock.chat.completions.create.return_value = _empty_chat_response()
        toolkit = PerplexitySearchToolkit(client=mock)
        answer_tool = toolkit.answer(search_mode=Variable())

        agent = Agent(
            "a",
            config=_tool_call_config({"query": "test query"}, tool_name="perplexity_answer"),
            tools=[answer_tool],
        )

        with pytest.raises(KeyError, match="search_mode"):
            await agent.ask("answer")


@pytest.mark.asyncio
class TestIndividualTools:
    async def test_search_tool_passed_alone(self, mock: MagicMock) -> None:
        mock.search.create.return_value = _empty_search_api_response()
        toolkit = PerplexitySearchToolkit(client=mock)

        agent = Agent(
            "a",
            config=_tool_call_config({"query": "q"}, tool_name="perplexity_search"),
            tools=[toolkit.search()],
        )
        await agent.ask("search")

        mock.search.create.assert_called_once_with(query="q")

    async def test_answer_tool_passed_alone(self, mock: MagicMock) -> None:
        mock.chat.completions.create.return_value = _empty_chat_response()
        toolkit = PerplexitySearchToolkit(client=mock)

        agent = Agent(
            "a",
            config=_tool_call_config({"query": "q"}, tool_name="perplexity_answer"),
            tools=[toolkit.answer()],
        )
        await agent.ask("answer")

        mock.chat.completions.create.assert_called_once()

    async def test_whole_toolkit_registers_both_tools(self, mock: MagicMock) -> None:
        mock.search.create.return_value = _empty_search_api_response()
        toolkit = PerplexitySearchToolkit(client=mock)

        agent = Agent(
            "a",
            config=_tool_call_config({"query": "q"}, tool_name="perplexity_search"),
            tools=[toolkit],
        )
        await agent.ask("search")

        mock.search.create.assert_called_once_with(query="q")
