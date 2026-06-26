# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from a2a.client.client_factory import TransportProtocol
from a2a.types import (
    AgentCard,
    AgentProvider,
    AgentSkill,
    ClientCredentialsOAuthFlow,
    HTTPAuthSecurityScheme,
    OAuthFlows,
    SecurityScheme,
)

from ag2 import Agent
from ag2.a2a import build_card
from ag2.a2a.security import bearer_scheme, oauth2_scheme, require
from ag2.testing import TestConfig
from ag2.tools.skills import LocalRuntime, SkillsToolkit


def _agent() -> Agent:
    return Agent("agent-x", config=TestConfig("ok"))


class TestCapabilities:
    def test_push_notifications_default_false(self) -> None:
        card = build_card(_agent(), url="http://test")

        assert card.capabilities.push_notifications is False

    def test_push_notifications_flag_propagates(self) -> None:
        card = build_card(_agent(), url="http://test", push_notifications=True)

        assert card.capabilities.push_notifications is True

    def test_streaming_always_true(self) -> None:
        card = build_card(_agent(), url="http://test")

        assert card.capabilities.streaming is True

    def test_round_trip_via_protobuf(self) -> None:
        card = build_card(_agent(), url="http://test", push_notifications=True)

        decoded = AgentCard.FromString(card.SerializeToString())

        assert decoded.capabilities.push_notifications is True


class TestSecurity:
    def test_no_security_by_default(self) -> None:
        card = build_card(_agent(), url="http://test")

        assert dict(card.security_schemes) == {}
        assert list(card.security_requirements) == []

    def test_schemes_auto_derived_from_requirements(self) -> None:
        bearer = bearer_scheme(name="bearer", description="JWT auth")

        card = build_card(_agent(), url="http://test", security=[require(bearer)])

        assert dict(card.security_schemes) == {
            "bearer": SecurityScheme(
                http_auth_security_scheme=HTTPAuthSecurityScheme(
                    scheme="bearer",
                    bearer_format="JWT",
                    description="JWT auth",
                ),
            ),
        }
        assert list(card.security_requirements) == [require(bearer).to_proto()]

    def test_round_trip_preserves_schemes(self) -> None:
        bearer = bearer_scheme(name="bearer")

        card = build_card(_agent(), url="http://test", security=[require(bearer)])

        decoded = AgentCard.FromString(card.SerializeToString())

        assert decoded == card

    def test_oauth2_scoped_requirement(self) -> None:
        flows = OAuthFlows(client_credentials=ClientCredentialsOAuthFlow(token_url="https://x/token"))
        oauth = oauth2_scheme(name="oauth", flows=flows)

        card = build_card(
            _agent(),
            url="http://test",
            security=[require(oauth.with_scopes("read", "write"))],
        )

        assert list(card.security_requirements) == [require(oauth.with_scopes("read", "write")).to_proto()]
        assert dict(card.security_schemes) == {"oauth": oauth.scheme}

    def test_scheme_deduped_across_requirements(self) -> None:
        bearer = bearer_scheme(name="bearer")
        oauth = oauth2_scheme(name="oauth", flows=OAuthFlows())

        card = build_card(
            _agent(),
            url="http://test",
            security=[
                require(bearer),
                require(oauth.with_scopes("read")),
                require(bearer, oauth.with_scopes("write")),
            ],
        )

        assert set(card.security_schemes.keys()) == {"bearer", "oauth"}
        assert len(card.security_requirements) == 3


class TestProviderAndBranding:
    def test_provider_passthrough(self) -> None:
        provider = AgentProvider(organization="AG2ai", url="https://ag2.dev")
        card = build_card(_agent(), url="http://test", provider=provider)

        assert card.provider == provider

    def test_documentation_and_icon_urls(self) -> None:
        card = build_card(
            _agent(),
            url="http://test",
            documentation_url="https://docs.example/agent-x",
            icon_url="https://docs.example/agent-x.png",
        )

        assert card.documentation_url == "https://docs.example/agent-x"
        assert card.icon_url == "https://docs.example/agent-x.png"

    def test_omitted_by_default(self) -> None:
        card = build_card(_agent(), url="http://test")

        assert card.documentation_url == ""
        assert card.icon_url == ""
        assert card.provider == AgentProvider()


class TestSkills:
    def test_default_single_skill_from_agent(self) -> None:
        card = build_card(_agent(), url="http://test")

        assert list(card.skills) == [
            AgentSkill(id="agent-x", name="agent-x", description="agent-x"),
        ]

    def test_custom_skills_replace_default(self) -> None:
        skills = [
            AgentSkill(
                id="search",
                name="Search",
                description="Search the web",
                tags=["search", "web"],
                examples=["Find recent papers on X"],
                input_modes=["text/plain"],
                output_modes=["application/json"],
            ),
            AgentSkill(id="summarize", name="Summarize", description="Summarize docs"),
        ]
        card = build_card(_agent(), url="http://test", skills=skills)

        assert list(card.skills) == skills

    def test_empty_skills_list_replaces_default(self) -> None:
        card = build_card(_agent(), url="http://test", skills=[])

        assert list(card.skills) == []


class TestSkillsAutoDiscovery:
    def test_skills_picked_up_from_skills_toolkit(self, local_skills_dir: Path) -> None:
        agent = Agent(
            "agent-x",
            config=TestConfig("ok"),
            tools=[SkillsToolkit(LocalRuntime(str(local_skills_dir)))],
        )

        card = build_card(agent, url="http://test")

        assert list(card.skills) == [
            AgentSkill(id="code-review", name="code-review", description="Review code for bugs and style"),
            AgentSkill(id="data-analysis", name="data-analysis", description="Analyse CSV/JSON datasets"),
        ]

    def test_explicit_skills_override_auto_discovery(self, local_skills_dir: Path) -> None:
        agent = Agent(
            "agent-x",
            config=TestConfig("ok"),
            tools=[SkillsToolkit(LocalRuntime(str(local_skills_dir)))],
        )
        override = [AgentSkill(id="override", name="override", description="custom")]

        card = build_card(agent, url="http://test", skills=override)

        assert list(card.skills) == override

    def test_falls_back_to_default_skill_when_no_toolkit(self) -> None:
        agent = Agent("agent-x", config=TestConfig("ok"))

        card = build_card(agent, url="http://test")

        assert list(card.skills) == [
            AgentSkill(id="agent-x", name="agent-x", description="agent-x"),
        ]

    def test_falls_back_when_toolkit_has_no_skills(self, tmp_path: Path) -> None:
        agent = Agent(
            "agent-x",
            config=TestConfig("ok"),
            tools=[SkillsToolkit(LocalRuntime(str(tmp_path)))],
        )

        card = build_card(agent, url="http://test")

        assert list(card.skills) == [
            AgentSkill(id="agent-x", name="agent-x", description="agent-x"),
        ]


class TestInterfaceTenants:
    def test_no_tenant_by_default(self) -> None:
        card = build_card(_agent(), url="http://test")

        [iface] = list(card.supported_interfaces)
        assert iface.tenant == ""

    def test_tenant_propagates_to_jsonrpc_interface(self) -> None:
        card = build_card(_agent(), url="http://test", tenants={"jsonrpc": "tenant-A"})

        [iface] = list(card.supported_interfaces)
        assert iface.tenant == "tenant-A"
        assert iface.protocol_binding == TransportProtocol.JSONRPC.value

    def test_tenant_per_transport(self) -> None:
        card = build_card(
            _agent(),
            url="http://j",
            transports=("jsonrpc", "rest", "grpc"),
            rest_url="http://r",
            grpc_url="grpc.example:50051",
            tenants={"jsonrpc": "t-j", "grpc": "t-g"},
        )

        tenants = {iface.protocol_binding: iface.tenant for iface in card.supported_interfaces}
        assert tenants == {
            TransportProtocol.JSONRPC.value: "t-j",
            TransportProtocol.HTTP_JSON.value: "",
            TransportProtocol.GRPC.value: "t-g",
        }
