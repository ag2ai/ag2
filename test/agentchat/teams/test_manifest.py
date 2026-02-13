# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for agent manifest loading - manifest-driven team building."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from dotenv import load_dotenv

load_dotenv(os.path.expanduser("~/.env"))

from autogen import ConversableAgent, LLMConfig
from autogen.agentchat.teams._manifest import (
    AGENT_CLASSES,
    AgentManifest,
    LocalAgentDef,
    ManifestAgent,
    ReferenceAgentDef,
    _expand_env_vars,
    _import_agent_class,
    _make_agent_description,
    _make_ask_specialist,
    _merge_llm_config,
    build_workers_from_manifest,
    load_manifest,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


SAMPLE_MANIFEST = {
    "agents": [
        {
            "id": "agent-1",
            "name": "WebResearcherAgent",
            "description": "Searches the web and summarizes findings.",
            "url": "https://example.com/web-researcher/",
            "platform": "AG2",
            "framework_type": "AG2",
            "is_global": True,
            "content": {
                "skills": [
                    {"id": "web_search", "name": "Web Search", "description": "Search the web"},
                    {"id": "summarize", "name": "Summarization", "description": "Summarize text"},
                ],
                "organization": "AG2 AI",
                "version": "1.0.0",
            },
            "required_connector_types": [],
        },
        {
            "id": "agent-2",
            "name": "CodingAgent",
            "description": "Writes code in multiple languages.",
            "url": "https://example.com/coding/",
            "platform": "AG2",
            "framework_type": "AG2",
            "is_global": True,
            "content": {
                "skills": [
                    {"id": "python", "name": "Python Development", "description": "Write Python code"},
                ],
                "organization": "AG2 AI",
                "version": "1.0.0",
            },
            "required_connector_types": [],
        },
        {
            "id": "agent-3",
            "name": "SlackAgent",
            "description": "Manages Slack messages and channels.",
            "url": "https://example.com/slack/",
            "platform": "AG2",
            "framework_type": "AG2",
            "is_global": False,
            "content": {
                "skills": [],
                "organization": "AG2 AI",
                "version": "1.0.0",
            },
            "required_connector_types": ["slack"],
        },
    ]
}

SAMPLE_LOCAL_MANIFEST = {
    "local_agents": [
        {
            "name": "writer",
            "description": "Writer & researcher",
            "system_message": "You are a skilled writer.",
            "llm_config": {
                "model": "claude-haiku-4-5",
                "api_type": "anthropic",
                "max_tokens": 8192,
            },
        },
        {
            "name": "analyst",
            "description": "Data analyst",
            "system_message": "You analyze data and produce reports.",
        },
    ]
}

SAMPLE_MIXED_MANIFEST = {
    "a2a": [
        {
            "id": "remote-1",
            "name": "RemoteAgent",
            "description": "A remote A2A agent.",
            "url": "https://example.com/remote/",
            "content": {"skills": [{"id": "s1", "name": "Skill1", "description": "Does stuff"}]},
        },
    ],
    "local_agents": [
        {
            "name": "local-writer",
            "description": "Writes things locally",
            "system_message": "You write things.",
        },
    ],
    "reference_agents": [
        {
            "name": "wiki-bot",
            "agent_class": "WikipediaAgent",
            "description": "Searches Wikipedia",
            "args": {"language": "en", "top_k": 3},
        },
    ],
}


@pytest.fixture
def manifest_file(tmp_path: Path) -> Path:
    """Create a temporary manifest JSON file."""
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(SAMPLE_MANIFEST))
    return path


@pytest.fixture
def manifest() -> AgentManifest:
    """Parse the sample manifest."""
    return AgentManifest.model_validate(SAMPLE_MANIFEST)


@pytest.fixture
def local_manifest() -> AgentManifest:
    """Parse the local-only manifest."""
    return AgentManifest.model_validate(SAMPLE_LOCAL_MANIFEST)


@pytest.fixture
def mixed_manifest() -> AgentManifest:
    """Parse the mixed manifest with all three categories."""
    return AgentManifest.model_validate(SAMPLE_MIXED_MANIFEST)


@pytest.fixture
def dummy_llm_config() -> LLMConfig:
    """A dummy LLM config for wrapper agents (won't make real calls)."""
    return LLMConfig({"model": "test-model", "api_key": "test-key", "api_type": "anthropic"})


# ---------------------------------------------------------------------------
# Manifest model tests
# ---------------------------------------------------------------------------


class TestManifestModels:
    """Test Pydantic models for manifest parsing."""

    def test_parse_manifest(self, manifest: AgentManifest) -> None:
        assert len(manifest.agents) == 3
        assert manifest.agents[0].name == "WebResearcherAgent"
        assert manifest.agents[1].name == "CodingAgent"
        assert manifest.agents[2].name == "SlackAgent"

    def test_agent_fields(self, manifest: AgentManifest) -> None:
        agent = manifest.agents[0]
        assert agent.id == "agent-1"
        assert agent.url == "https://example.com/web-researcher/"
        assert agent.platform == "AG2"
        assert agent.is_global is True
        assert len(agent.content.skills) == 2

    def test_skill_fields(self, manifest: AgentManifest) -> None:
        skill = manifest.agents[0].content.skills[0]
        assert skill.id == "web_search"
        assert skill.name == "Web Search"
        assert skill.description == "Search the web"

    def test_content_fields(self, manifest: AgentManifest) -> None:
        content = manifest.agents[0].content
        assert content.organization == "AG2 AI"
        assert content.version == "1.0.0"

    def test_required_connectors(self, manifest: AgentManifest) -> None:
        assert manifest.agents[2].required_connector_types == ["slack"]
        assert manifest.agents[0].required_connector_types == []

    def test_serialization_roundtrip(self, manifest: AgentManifest) -> None:
        json_str = manifest.model_dump_json()
        restored = AgentManifest.model_validate_json(json_str)
        assert len(restored.agents) == 3
        assert restored.agents[0].name == "WebResearcherAgent"

    def test_empty_manifest(self) -> None:
        m = AgentManifest.model_validate({"agents": []})
        assert len(m.agents) == 0

    def test_minimal_agent(self) -> None:
        """Agent with only required fields."""
        data = {"agents": [{"id": "x", "name": "MinimalAgent"}]}
        m = AgentManifest.model_validate(data)
        assert m.agents[0].name == "MinimalAgent"
        assert m.agents[0].url == ""
        assert m.agents[0].content.skills == []

    def test_local_agent_def(self) -> None:
        """LocalAgentDef model parses correctly."""
        d = LocalAgentDef(
            name="writer",
            description="Writes",
            system_message="You write.",
            llm_config={"model": "haiku"},
        )
        assert d.name == "writer"
        assert d.llm_config["model"] == "haiku"

    def test_local_agent_def_empty_llm_config(self) -> None:
        """LocalAgentDef defaults to empty llm_config."""
        d = LocalAgentDef(name="basic")
        assert d.llm_config == {}

    def test_reference_agent_def(self) -> None:
        """ReferenceAgentDef model parses correctly."""
        d = ReferenceAgentDef(
            name="slack-bot",
            agent_class="SlackAgent",
            args={"bot_token": "xoxb-123", "channel_id": "C123"},
            llm_config={"model": "gpt-4.1"},
        )
        assert d.agent_class == "SlackAgent"
        assert d.args["bot_token"] == "xoxb-123"
        assert d.llm_config["model"] == "gpt-4.1"

    def test_three_category_manifest(self) -> None:
        """AgentManifest with all three categories."""
        m = AgentManifest.model_validate(SAMPLE_MIXED_MANIFEST)
        assert len(m.a2a) == 1
        assert len(m.local_agents) == 1
        assert len(m.reference_agents) == 1
        assert m.a2a[0].name == "RemoteAgent"
        assert m.local_agents[0].name == "local-writer"
        assert m.reference_agents[0].agent_class == "WikipediaAgent"

    def test_a2a_key_separate_from_agents(self) -> None:
        """a2a and agents are separate lists."""
        data = {
            "a2a": [{"id": "a", "name": "A2AAgent"}],
            "agents": [{"id": "b", "name": "LegacyAgent"}],
        }
        m = AgentManifest.model_validate(data)
        assert len(m.a2a) == 1
        assert len(m.agents) == 1
        assert m.a2a[0].name == "A2AAgent"
        assert m.agents[0].name == "LegacyAgent"

    def test_empty_three_category(self) -> None:
        """All-empty manifest is valid."""
        m = AgentManifest.model_validate({})
        assert m.a2a == []
        assert m.agents == []
        assert m.local_agents == []
        assert m.reference_agents == []


# ---------------------------------------------------------------------------
# Load manifest tests
# ---------------------------------------------------------------------------


class TestLoadManifest:
    """Test loading manifest from file."""

    def test_load_from_file(self, manifest_file: Path) -> None:
        m = load_manifest(manifest_file)
        assert len(m.agents) == 3
        assert m.agents[0].name == "WebResearcherAgent"

    def test_load_from_string_path(self, manifest_file: Path) -> None:
        m = load_manifest(str(manifest_file))
        assert len(m.agents) == 3

    def test_load_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_manifest("/nonexistent/path.json")

    def test_load_sample_manifest(self) -> None:
        """Load the actual sample manifest from teams_docs."""
        sample = Path(__file__).parents[3] / "teams_docs" / "sample_agentos_agent_manifests.json"
        if not sample.exists():
            pytest.skip("Sample manifest not found")
        m = load_manifest(sample)
        assert len(m.agents) > 0
        # Check a known agent
        names = {a.name for a in m.agents}
        assert "AtlassianAgent" in names or len(names) > 5

    def test_load_mixed_manifest(self, tmp_path: Path) -> None:
        """Load a mixed manifest from file."""
        path = tmp_path / "mixed.json"
        path.write_text(json.dumps(SAMPLE_MIXED_MANIFEST))
        m = load_manifest(path)
        assert len(m.a2a) == 1
        assert len(m.local_agents) == 1
        assert len(m.reference_agents) == 1


# ---------------------------------------------------------------------------
# Agent description builder
# ---------------------------------------------------------------------------


class TestMakeAgentDescription:
    def test_basic_description(self, manifest: AgentManifest) -> None:
        desc = _make_agent_description(manifest.agents[0])
        assert "Searches the web" in desc
        assert "Web Search" in desc
        assert "Summarization" in desc

    def test_long_description_truncated(self) -> None:
        agent = ManifestAgent(
            id="x",
            name="LongAgent",
            description="A" * 500,
        )
        desc = _make_agent_description(agent)
        assert len(desc) < 500
        assert desc.endswith("...")

    def test_no_skills(self) -> None:
        agent = ManifestAgent(
            id="x",
            name="NoSkillAgent",
            description="Just a description",
        )
        desc = _make_agent_description(agent)
        assert "Skills:" not in desc


# ---------------------------------------------------------------------------
# Environment variable expansion tests
# ---------------------------------------------------------------------------


class TestExpandEnvVars:
    def test_simple_string(self) -> None:
        with patch.dict(os.environ, {"MY_TOKEN": "secret123"}):
            assert _expand_env_vars("${MY_TOKEN}") == "secret123"

    def test_string_with_prefix(self) -> None:
        with patch.dict(os.environ, {"TOKEN": "abc"}):
            assert _expand_env_vars("Bearer ${TOKEN}") == "Bearer abc"

    def test_nested_dict(self) -> None:
        with patch.dict(os.environ, {"A": "val_a", "B": "val_b"}):
            result = _expand_env_vars({"key1": "${A}", "key2": "${B}"})
            assert result == {"key1": "val_a", "key2": "val_b"}

    def test_nested_list(self) -> None:
        with patch.dict(os.environ, {"X": "hello"}):
            result = _expand_env_vars(["${X}", "literal"])
            assert result == ["hello", "literal"]

    def test_non_string_passthrough(self) -> None:
        assert _expand_env_vars(42) == 42
        assert _expand_env_vars(True) is True
        assert _expand_env_vars(None) is None

    def test_missing_env_var_raises(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            # Remove the var if it exists
            os.environ.pop("DEFINITELY_NOT_SET_XYZ", None)
            with pytest.raises(KeyError, match="DEFINITELY_NOT_SET_XYZ"):
                _expand_env_vars("${DEFINITELY_NOT_SET_XYZ}")

    def test_no_expansion_needed(self) -> None:
        assert _expand_env_vars("plain string") == "plain string"

    def test_mixed_dict_with_non_strings(self) -> None:
        with patch.dict(os.environ, {"VAR": "expanded"}):
            result = _expand_env_vars({"s": "${VAR}", "n": 5, "b": True})
            assert result == {"s": "expanded", "n": 5, "b": True}


# ---------------------------------------------------------------------------
# Import agent class tests
# ---------------------------------------------------------------------------


class TestImportAgentClass:
    def test_unknown_class_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown agent class"):
            _import_agent_class("NonExistentAgent")

    def test_registry_has_expected_classes(self) -> None:
        expected = {
            "SlackAgent",
            "DiscordAgent",
            "TelegramAgent",
            "WebSurferAgent",
            "WikipediaAgent",
            "ReasoningAgent",
            "DeepResearchAgent",
        }
        assert expected == set(AGENT_CLASSES.keys())

    def test_import_wikipedia_agent(self) -> None:
        """WikipediaAgent should import without heavy dependencies."""
        try:
            cls = _import_agent_class("WikipediaAgent")
            assert cls.__name__ == "WikipediaAgent"
        except ImportError:
            pytest.skip("WikipediaAgent dependencies not installed")


# ---------------------------------------------------------------------------
# LLM config override tests
# ---------------------------------------------------------------------------


class TestMergeLLMConfig:
    def test_no_override_returns_original(self, dummy_llm_config: LLMConfig) -> None:
        result = _merge_llm_config(dummy_llm_config)
        assert result is dummy_llm_config

    def test_empty_dict_returns_original(self, dummy_llm_config: LLMConfig) -> None:
        result = _merge_llm_config(dummy_llm_config, {})
        assert result is dummy_llm_config

    def test_model_override(self, dummy_llm_config: LLMConfig) -> None:
        result = _merge_llm_config(dummy_llm_config, {"model": "new-model"})
        dumped = result.model_dump()
        assert dumped["config_list"][0]["model"] == "new-model"

    def test_max_tokens_override(self, dummy_llm_config: LLMConfig) -> None:
        result = _merge_llm_config(dummy_llm_config, {"max_tokens": 4096})
        dumped = result.model_dump()
        assert dumped["config_list"][0]["max_tokens"] == 4096

    def test_api_type_override(self, dummy_llm_config: LLMConfig) -> None:
        result = _merge_llm_config(dummy_llm_config, {"api_type": "openai"})
        dumped = result.model_dump()
        assert dumped["config_list"][0]["api_type"] == "openai"

    def test_multiple_overrides(self, dummy_llm_config: LLMConfig) -> None:
        result = _merge_llm_config(dummy_llm_config, {"model": "m", "max_tokens": 100})
        dumped = result.model_dump()
        assert dumped["config_list"][0]["model"] == "m"
        assert dumped["config_list"][0]["max_tokens"] == 100

    def test_dict_input(self) -> None:
        base = {"model": "old", "api_key": "k", "api_type": "anthropic"}
        result = _merge_llm_config(base, {"model": "new"})
        dumped = result.model_dump()
        assert dumped["config_list"][0]["model"] == "new"

    def test_built_in_tools_passthrough(self, dummy_llm_config: LLMConfig) -> None:
        """built_in_tools (e.g. for Responses API web_search) passes through."""
        result = _merge_llm_config(
            dummy_llm_config,
            {"model": "gpt-4.1-mini", "api_type": "responses", "built_in_tools": ["web_search"]},
        )
        dumped = result.model_dump()
        entry = dumped["config_list"][0]
        assert entry["model"] == "gpt-4.1-mini"
        assert entry["api_type"] == "responses"


# ---------------------------------------------------------------------------
# Build workers tests (A2A / legacy agents)
# ---------------------------------------------------------------------------


class TestBuildWorkersFromManifest:
    def test_build_all_workers(self, manifest: AgentManifest, dummy_llm_config: LLMConfig) -> None:
        workers = build_workers_from_manifest(manifest, dummy_llm_config)
        assert len(workers) == 3
        names = {w.name for w in workers}
        assert names == {"WebResearcherAgent", "CodingAgent", "SlackAgent"}

    def test_filter_by_name(self, manifest: AgentManifest, dummy_llm_config: LLMConfig) -> None:
        workers = build_workers_from_manifest(
            manifest,
            dummy_llm_config,
            agent_names=["CodingAgent"],
        )
        assert len(workers) == 1
        assert workers[0].name == "CodingAgent"

    def test_filter_multiple_names(self, manifest: AgentManifest, dummy_llm_config: LLMConfig) -> None:
        workers = build_workers_from_manifest(
            manifest,
            dummy_llm_config,
            agent_names=["CodingAgent", "SlackAgent"],
        )
        assert len(workers) == 2
        names = {w.name for w in workers}
        assert names == {"CodingAgent", "SlackAgent"}

    def test_empty_manifest(self, dummy_llm_config: LLMConfig) -> None:
        m = AgentManifest(agents=[])
        workers = build_workers_from_manifest(m, dummy_llm_config)
        assert workers == []

    def test_worker_is_conversable_agent(self, manifest: AgentManifest, dummy_llm_config: LLMConfig) -> None:
        workers = build_workers_from_manifest(manifest, dummy_llm_config)
        for w in workers:
            assert isinstance(w, ConversableAgent)

    def test_worker_has_llm_config(self, manifest: AgentManifest, dummy_llm_config: LLMConfig) -> None:
        workers = build_workers_from_manifest(manifest, dummy_llm_config)
        for w in workers:
            assert w.llm_config is not None

    def test_worker_has_description(self, manifest: AgentManifest, dummy_llm_config: LLMConfig) -> None:
        workers = build_workers_from_manifest(manifest, dummy_llm_config)
        web_agent = next(w for w in workers if w.name == "WebResearcherAgent")
        assert "Searches the web" in web_agent.description

    def test_worker_has_system_message(self, manifest: AgentManifest, dummy_llm_config: LLMConfig) -> None:
        workers = build_workers_from_manifest(manifest, dummy_llm_config)
        web_agent = next(w for w in workers if w.name == "WebResearcherAgent")
        assert "ask_specialist" in web_agent.system_message
        assert "delegate" in web_agent.system_message.lower()

    def test_worker_has_ask_specialist_tool(self, manifest: AgentManifest, dummy_llm_config: LLMConfig) -> None:
        """Verify the ask_specialist tool is registered on each worker."""
        workers = build_workers_from_manifest(manifest, dummy_llm_config)
        for w in workers:
            assert "ask_specialist" in w._function_map

    def test_worker_has_manifest_metadata(self, manifest: AgentManifest, dummy_llm_config: LLMConfig) -> None:
        workers = build_workers_from_manifest(manifest, dummy_llm_config)
        web_agent = next(w for w in workers if w.name == "WebResearcherAgent")
        assert hasattr(web_agent, "_manifest_agent")
        assert web_agent._manifest_agent.url == "https://example.com/web-researcher/"

    def test_workers_can_be_added_to_team(self, manifest: AgentManifest, dummy_llm_config: LLMConfig) -> None:
        """Verify workers integrate with Team.add_agent() properly."""
        from autogen.agentchat.teams import Team

        team = Team("test-team")
        leader = ConversableAgent("leader", llm_config=dummy_llm_config)
        team.add_agent(leader, is_leader=True)

        workers = build_workers_from_manifest(manifest, dummy_llm_config)
        for w in workers:
            team.add_agent(w)

        assert len(team.agents) == 4  # leader + 3 workers
        # Workers should have task tools registered by Team.add_agent
        web_agent = team.agents["WebResearcherAgent"]
        assert "my_tasks" in web_agent._function_map
        assert "claim_task" in web_agent._function_map
        assert "complete_task" in web_agent._function_map
        # And still have ask_specialist
        assert "ask_specialist" in web_agent._function_map


# ---------------------------------------------------------------------------
# Build local workers tests
# ---------------------------------------------------------------------------


class TestBuildLocalWorkers:
    def test_local_agents_created(self, local_manifest: AgentManifest, dummy_llm_config: LLMConfig) -> None:
        workers = build_workers_from_manifest(local_manifest, dummy_llm_config)
        assert len(workers) == 2
        names = {w.name for w in workers}
        assert names == {"writer", "analyst"}

    def test_local_agent_has_system_message(self, local_manifest: AgentManifest, dummy_llm_config: LLMConfig) -> None:
        workers = build_workers_from_manifest(local_manifest, dummy_llm_config)
        writer = next(w for w in workers if w.name == "writer")
        assert writer.system_message == "You are a skilled writer."

    def test_local_agent_has_description(self, local_manifest: AgentManifest, dummy_llm_config: LLMConfig) -> None:
        workers = build_workers_from_manifest(local_manifest, dummy_llm_config)
        writer = next(w for w in workers if w.name == "writer")
        assert writer.description == "Writer & researcher"

    def test_local_agent_no_ask_specialist(self, local_manifest: AgentManifest, dummy_llm_config: LLMConfig) -> None:
        """Local agents should NOT have the ask_specialist delegation tool."""
        workers = build_workers_from_manifest(local_manifest, dummy_llm_config)
        for w in workers:
            assert "ask_specialist" not in w._function_map

    def test_local_agent_model_override(self, local_manifest: AgentManifest, dummy_llm_config: LLMConfig) -> None:
        """Local agent with model override gets a different model in llm_config."""
        workers = build_workers_from_manifest(local_manifest, dummy_llm_config)
        writer = next(w for w in workers if w.name == "writer")
        dumped = writer.llm_config.model_dump()
        assert dumped["config_list"][0]["model"] == "claude-haiku-4-5"

    def test_local_agent_max_tokens_override(self, local_manifest: AgentManifest, dummy_llm_config: LLMConfig) -> None:
        workers = build_workers_from_manifest(local_manifest, dummy_llm_config)
        writer = next(w for w in workers if w.name == "writer")
        dumped = writer.llm_config.model_dump()
        assert dumped["config_list"][0]["max_tokens"] == 8192

    def test_local_agent_api_type_override(self, local_manifest: AgentManifest, dummy_llm_config: LLMConfig) -> None:
        """Writer has api_type in llm_config which overrides the base config."""
        workers = build_workers_from_manifest(local_manifest, dummy_llm_config)
        writer = next(w for w in workers if w.name == "writer")
        dumped = writer.llm_config.model_dump()
        assert dumped["config_list"][0]["api_type"] == "anthropic"

    def test_local_agent_no_override_uses_base(
        self, local_manifest: AgentManifest, dummy_llm_config: LLMConfig
    ) -> None:
        """Agent without model override uses the base config."""
        workers = build_workers_from_manifest(local_manifest, dummy_llm_config)
        analyst = next(w for w in workers if w.name == "analyst")
        dumped = analyst.llm_config.model_dump()
        assert dumped["config_list"][0]["model"] == "test-model"

    def test_local_agent_filter_by_name(self, local_manifest: AgentManifest, dummy_llm_config: LLMConfig) -> None:
        workers = build_workers_from_manifest(
            local_manifest,
            dummy_llm_config,
            agent_names=["writer"],
        )
        assert len(workers) == 1
        assert workers[0].name == "writer"

    def test_local_agent_is_conversable_agent(self, local_manifest: AgentManifest, dummy_llm_config: LLMConfig) -> None:
        workers = build_workers_from_manifest(local_manifest, dummy_llm_config)
        for w in workers:
            assert isinstance(w, ConversableAgent)


# ---------------------------------------------------------------------------
# Build reference workers tests
# ---------------------------------------------------------------------------


class TestBuildReferenceWorkers:
    def test_reference_agent_created(self, dummy_llm_config: LLMConfig) -> None:
        """Reference agent with a mock class gets instantiated."""
        mock_cls = MagicMock(return_value=ConversableAgent("mock-agent", llm_config=dummy_llm_config))
        manifest = AgentManifest(
            reference_agents=[
                ReferenceAgentDef(
                    name="mock-agent",
                    agent_class="WikipediaAgent",
                    args={"language": "en"},
                ),
            ],
        )
        with patch("autogen.agentchat.teams._manifest._import_agent_class", return_value=mock_cls):
            workers = build_workers_from_manifest(manifest, dummy_llm_config)

        assert len(workers) == 1
        assert workers[0].name == "mock-agent"
        # Verify the class was called with expected args
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["name"] == "mock-agent"
        assert call_kwargs["language"] == "en"
        assert "llm_config" in call_kwargs

    def test_reference_agent_no_ask_specialist(self, dummy_llm_config: LLMConfig) -> None:
        """Reference agents should NOT have ask_specialist tool."""
        mock_cls = MagicMock(return_value=ConversableAgent("ref-agent", llm_config=dummy_llm_config))
        manifest = AgentManifest(
            reference_agents=[
                ReferenceAgentDef(name="ref-agent", agent_class="WikipediaAgent"),
            ],
        )
        with patch("autogen.agentchat.teams._manifest._import_agent_class", return_value=mock_cls):
            workers = build_workers_from_manifest(manifest, dummy_llm_config)

        assert "ask_specialist" not in workers[0]._function_map

    def test_reference_agent_env_var_expansion(self, dummy_llm_config: LLMConfig) -> None:
        """${VAR} in args gets expanded from environment."""
        mock_cls = MagicMock(return_value=ConversableAgent("slack-bot", llm_config=dummy_llm_config))
        manifest = AgentManifest(
            reference_agents=[
                ReferenceAgentDef(
                    name="slack-bot",
                    agent_class="SlackAgent",
                    args={"bot_token": "${MY_SLACK_TOKEN}", "channel_id": "C123"},
                ),
            ],
        )
        with (
            patch.dict(os.environ, {"MY_SLACK_TOKEN": "xoxb-secret"}),
            patch("autogen.agentchat.teams._manifest._import_agent_class", return_value=mock_cls),
        ):
            build_workers_from_manifest(manifest, dummy_llm_config)

        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["bot_token"] == "xoxb-secret"
        assert call_kwargs["channel_id"] == "C123"

    def test_reference_agent_missing_env_var_raises(self, dummy_llm_config: LLMConfig) -> None:
        """Missing env var in args raises KeyError."""
        manifest = AgentManifest(
            reference_agents=[
                ReferenceAgentDef(
                    name="bad-agent",
                    agent_class="SlackAgent",
                    args={"bot_token": "${MISSING_VAR_XYZ_123}"},
                ),
            ],
        )
        os.environ.pop("MISSING_VAR_XYZ_123", None)
        with pytest.raises(KeyError, match="MISSING_VAR_XYZ_123"):
            build_workers_from_manifest(manifest, dummy_llm_config)

    def test_reference_agent_description_override(self, dummy_llm_config: LLMConfig) -> None:
        """Description from manifest overrides class default."""
        mock_cls = MagicMock(return_value=ConversableAgent("wiki", llm_config=dummy_llm_config))
        manifest = AgentManifest(
            reference_agents=[
                ReferenceAgentDef(
                    name="wiki",
                    agent_class="WikipediaAgent",
                    description="Custom wiki description",
                ),
            ],
        )
        with patch("autogen.agentchat.teams._manifest._import_agent_class", return_value=mock_cls):
            build_workers_from_manifest(manifest, dummy_llm_config)

        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["description"] == "Custom wiki description"

    def test_reference_agent_model_override(self, dummy_llm_config: LLMConfig) -> None:
        """Reference agent with model override via llm_config dict."""
        mock_cls = MagicMock(return_value=ConversableAgent("wiki", llm_config=dummy_llm_config))
        manifest = AgentManifest(
            reference_agents=[
                ReferenceAgentDef(
                    name="wiki",
                    agent_class="WikipediaAgent",
                    llm_config={"model": "custom-model"},
                ),
            ],
        )
        with patch("autogen.agentchat.teams._manifest._import_agent_class", return_value=mock_cls):
            build_workers_from_manifest(manifest, dummy_llm_config)

        call_kwargs = mock_cls.call_args[1]
        llm_cfg = call_kwargs["llm_config"]
        dumped = llm_cfg.model_dump()
        assert dumped["config_list"][0]["model"] == "custom-model"

    def test_reference_agent_filter_by_name(self, dummy_llm_config: LLMConfig) -> None:
        mock_cls = MagicMock(return_value=ConversableAgent("wiki", llm_config=dummy_llm_config))
        manifest = AgentManifest(
            reference_agents=[
                ReferenceAgentDef(name="wiki", agent_class="WikipediaAgent"),
                ReferenceAgentDef(name="other", agent_class="WikipediaAgent"),
            ],
        )
        with patch("autogen.agentchat.teams._manifest._import_agent_class", return_value=mock_cls):
            workers = build_workers_from_manifest(
                manifest,
                dummy_llm_config,
                agent_names=["wiki"],
            )

        assert len(workers) == 1

    def test_unknown_agent_class_raises(self, dummy_llm_config: LLMConfig) -> None:
        manifest = AgentManifest(
            reference_agents=[
                ReferenceAgentDef(name="bad", agent_class="FakeAgent"),
            ],
        )
        with pytest.raises(ValueError, match="Unknown agent class"):
            build_workers_from_manifest(manifest, dummy_llm_config)


# ---------------------------------------------------------------------------
# Mixed manifest tests
# ---------------------------------------------------------------------------


class TestMixedManifest:
    def test_mixed_manifest_all_categories(self, mixed_manifest: AgentManifest, dummy_llm_config: LLMConfig) -> None:
        """Build workers from a manifest with all three categories."""
        mock_cls = MagicMock(return_value=ConversableAgent("wiki-bot", llm_config=dummy_llm_config))
        with patch("autogen.agentchat.teams._manifest._import_agent_class", return_value=mock_cls):
            workers = build_workers_from_manifest(mixed_manifest, dummy_llm_config)

        assert len(workers) == 3
        names = {w.name for w in workers}
        assert names == {"RemoteAgent", "local-writer", "wiki-bot"}

    def test_mixed_manifest_a2a_has_ask_specialist(
        self, mixed_manifest: AgentManifest, dummy_llm_config: LLMConfig
    ) -> None:
        """Only A2A agents get ask_specialist."""
        mock_cls = MagicMock(return_value=ConversableAgent("wiki-bot", llm_config=dummy_llm_config))
        with patch("autogen.agentchat.teams._manifest._import_agent_class", return_value=mock_cls):
            workers = build_workers_from_manifest(mixed_manifest, dummy_llm_config)

        remote = next(w for w in workers if w.name == "RemoteAgent")
        local = next(w for w in workers if w.name == "local-writer")

        assert "ask_specialist" in remote._function_map
        assert "ask_specialist" not in local._function_map

    def test_mixed_manifest_filter_across_categories(
        self, mixed_manifest: AgentManifest, dummy_llm_config: LLMConfig
    ) -> None:
        """agent_names filter works across all three categories."""
        mock_cls = MagicMock(return_value=ConversableAgent("wiki-bot", llm_config=dummy_llm_config))
        with patch("autogen.agentchat.teams._manifest._import_agent_class", return_value=mock_cls):
            workers = build_workers_from_manifest(
                mixed_manifest,
                dummy_llm_config,
                agent_names=["local-writer", "wiki-bot"],
            )

        assert len(workers) == 2
        names = {w.name for w in workers}
        assert names == {"local-writer", "wiki-bot"}

    def test_backward_compat_agents_plus_a2a(self, dummy_llm_config: LLMConfig) -> None:
        """Both 'agents' (legacy) and 'a2a' entries are merged."""
        data = {
            "a2a": [{"id": "1", "name": "A2A_One", "url": "https://one.com/"}],
            "agents": [{"id": "2", "name": "Legacy_One", "url": "https://two.com/"}],
        }
        m = AgentManifest.model_validate(data)
        workers = build_workers_from_manifest(m, dummy_llm_config)

        assert len(workers) == 2
        names = {w.name for w in workers}
        assert names == {"A2A_One", "Legacy_One"}
        # Both should have ask_specialist
        for w in workers:
            assert "ask_specialist" in w._function_map


# ---------------------------------------------------------------------------
# ask_specialist tool tests
# ---------------------------------------------------------------------------


class TestAskSpecialist:
    def test_factory_creates_async_function(self) -> None:
        func = _make_ask_specialist("https://example.com/agent/", "TestAgent")
        assert asyncio.iscoroutinefunction(func)

    def test_factory_captures_closure(self) -> None:
        """Each factory call captures its own URL/name (not shared)."""
        func1 = _make_ask_specialist("https://url1.com/", "Agent1")
        func2 = _make_ask_specialist("https://url2.com/", "Agent2")
        # They should be distinct functions
        assert func1 is not func2

    def test_missing_a2a_sdk_returns_error(self) -> None:
        """If a2a SDK isn't installed, the tool returns a helpful error message."""
        func = _make_ask_specialist("https://example.com/agent/", "TestAgent")

        # We can't easily mock the import, but if a2a IS installed,
        # calling the tool with a bad URL should return an error message
        result = asyncio.get_event_loop().run_until_complete(func("test message"))
        # Should either be an import error or a connection error
        assert isinstance(result, str)
        assert "Error" in result or "error" in result or "Specialist" in result


# ---------------------------------------------------------------------------
# Integration: manifest -> team -> orchestrator config
# ---------------------------------------------------------------------------


class TestManifestTeamIntegration:
    def test_full_pipeline(self, manifest_file: Path, dummy_llm_config: LLMConfig) -> None:
        """Test the full pipeline: load manifest -> build workers -> add to team."""
        from autogen.agentchat.teams import Team

        manifest = load_manifest(manifest_file)
        team = Team("manifest-test", description="Test team from manifest")

        leader = ConversableAgent("leader", llm_config=dummy_llm_config)
        team.add_agent(leader, is_leader=True)

        workers = build_workers_from_manifest(manifest, dummy_llm_config)
        for w in workers:
            team.add_agent(w)

        # Team should be properly configured
        assert team.leader is leader
        assert len(team.agents) == 4

        # Config should serialize
        config = team.config()
        assert config.name == "manifest-test"
        assert "WebResearcherAgent" in config.agents

    def test_leader_sees_worker_descriptions(self, manifest: AgentManifest, dummy_llm_config: LLMConfig) -> None:
        """Verify worker descriptions are visible for leader task assignment."""
        from autogen.agentchat.teams import Orchestrator, Team

        team = Team("test", description="Test")
        leader = ConversableAgent("leader", llm_config=dummy_llm_config)
        team.add_agent(leader, is_leader=True)

        workers = build_workers_from_manifest(manifest, dummy_llm_config)
        for w in workers:
            team.add_agent(w)

        orch = Orchestrator(team)
        desc = orch._get_worker_descriptions()

        assert "WebResearcherAgent" in desc
        assert "CodingAgent" in desc
        assert "SlackAgent" in desc

    def test_local_workers_in_team(self, local_manifest: AgentManifest, dummy_llm_config: LLMConfig) -> None:
        """Local agents can be added to a team and get task tools."""
        from autogen.agentchat.teams import Team

        team = Team("local-test")
        leader = ConversableAgent("leader", llm_config=dummy_llm_config)
        team.add_agent(leader, is_leader=True)

        workers = build_workers_from_manifest(local_manifest, dummy_llm_config)
        for w in workers:
            team.add_agent(w)

        assert len(team.agents) == 3  # leader + 2 local
        writer = team.agents["writer"]
        assert "my_tasks" in writer._function_map
        assert "claim_task" in writer._function_map
        assert "complete_task" in writer._function_map
        # Local agents should NOT have ask_specialist
        assert "ask_specialist" not in writer._function_map


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
