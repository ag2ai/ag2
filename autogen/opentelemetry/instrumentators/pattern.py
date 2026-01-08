# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any, Optional

from opentelemetry.sdk.trace import TracerProvider

from autogen import ConversableAgent
from autogen.agentchat import Agent
from autogen.agentchat.group import ContextVariables
from autogen.agentchat.group.group_tool_executor import GroupToolExecutor
from autogen.agentchat.group.patterns.pattern import Pattern
from autogen.agentchat.group.targets.transition_target import TransitionTarget
from autogen.agentchat.groupchat import GroupChat, GroupChatManager
from autogen.doc_utils import export_module
from autogen.opentelemetry.consts import SpanType
from autogen.opentelemetry.setup import get_tracer

from .agent import instrument_agent


@export_module("autogen.opentelemetry")
def instrument_pattern(pattern: Pattern, *, tracer_provider: TracerProvider) -> Pattern:
    """Instrument a Pattern with OpenTelemetry tracing.

    Instruments the pattern's prepare_group_chat method to automatically
    instrument all agents and group chats created by the pattern.

    Args:
        pattern: The pattern instance to instrument.
        tracer_provider: The OpenTelemetry tracer provider to use for creating spans.

    Returns:
        The instrumented pattern instance (same object, modified in place).

    Usage:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from autogen.opentelemetry import instrument_pattern

        resource = Resource.create(attributes={"service.name": "my-service"})
        tracer_provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint="http://127.0.0.1:4317")
        processor = BatchSpanProcessor(exporter)
        tracer_provider.add_span_processor(processor)
        trace.set_tracer_provider(tracer_provider)

        pattern = SomePattern()
        instrument_pattern(pattern, tracer_provider=tracer_provider)
    """
    old_prepare_group_chat = pattern.prepare_group_chat

    def prepare_group_chat_traced(
        max_rounds: int,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[
        list["ConversableAgent"],
        list["ConversableAgent"],
        Optional["ConversableAgent"],
        ContextVariables,
        "ConversableAgent",
        TransitionTarget,
        "GroupToolExecutor",
        "GroupChat",
        "GroupChatManager",
        list[dict[str, Any]],
        "ConversableAgent",
        list[str],
        list["Agent"],
    ]:
        (
            agents,
            wrapped_agents,
            user_agent,
            context_variables,
            initial_agent,
            group_after_work,
            tool_executor,
            groupchat,
            manager,
            processed_messages,
            last_agent,
            group_agent_names,
            temp_user_list,
        ) = old_prepare_group_chat(max_rounds, *args, **kwargs)

        for agent in groupchat.agents:
            instrument_agent(agent, tracer_provider=tracer_provider)

        instrument_agent(manager, tracer_provider=tracer_provider)
        instrument_groupchat(groupchat, tracer_provider=tracer_provider)

        # IMPORTANT: register_reply() in GroupChatManager.__init__ creates a shallow copy of groupchat
        # (via copy.copy). We need to also instrument that copy which is stored in manager._reply_func_list
        # so that we can trace the "auto" speaker selection internal chats.
        for reply_func_entry in manager._reply_func_list:
            config = reply_func_entry.get("config")
            if isinstance(config, GroupChat) and config is not groupchat:
                instrument_groupchat(config, tracer_provider=tracer_provider)

        return (
            agents,
            wrapped_agents,
            user_agent,
            context_variables,
            initial_agent,
            group_after_work,
            tool_executor,
            groupchat,
            manager,
            processed_messages,
            last_agent,
            group_agent_names,
            temp_user_list,
        )

    pattern.prepare_group_chat = prepare_group_chat_traced

    return pattern


def instrument_groupchat(groupchat: GroupChat, *, tracer_provider: TracerProvider) -> GroupChat:
    tracer = get_tracer(tracer_provider)
    # Wrap _create_internal_agents to instrument temporary agents for auto speaker selection
    old_create_internal_agents = groupchat._create_internal_agents

    def create_internal_agents_traced(
        agents: list[Agent],
        max_attempts: int,
        messages: list[dict[str, Any]],
        validate_speaker_name: Any,
        selector: ConversableAgent | None = None,
    ) -> tuple[ConversableAgent, ConversableAgent]:
        checking_agent, speaker_selection_agent = old_create_internal_agents(
            agents, max_attempts, messages, validate_speaker_name, selector
        )
        # Instrument the temporary agents so their chats are traced
        instrument_agent(checking_agent, tracer_provider=tracer_provider)
        instrument_agent(speaker_selection_agent, tracer_provider=tracer_provider)
        return checking_agent, speaker_selection_agent

    groupchat._create_internal_agents = create_internal_agents_traced

    # Wrap a_auto_select_speaker with a parent span
    old_a_auto_select_speaker = groupchat.a_auto_select_speaker

    async def a_auto_select_speaker_traced(
        last_speaker: Agent,
        selector: ConversableAgent,
        messages: list[dict[str, Any]] | None,
        agents: list[Agent] | None,
    ) -> Agent:
        with tracer.start_as_current_span("speaker_selection") as span:
            span.set_attribute("ag2.span.type", SpanType.SPEAKER_SELECTION.value)
            span.set_attribute("gen_ai.operation.name", "speaker_selection")

            # Record candidate agents
            candidate_agents = agents if agents is not None else groupchat.agents
            span.set_attribute(
                "ag2.speaker_selection.candidates",
                json.dumps([a.name for a in candidate_agents]),
            )

            result = await old_a_auto_select_speaker(last_speaker, selector, messages, agents)

            # Record selected speaker
            span.set_attribute("ag2.speaker_selection.selected", result.name)
            return result

    groupchat.a_auto_select_speaker = a_auto_select_speaker_traced

    # Wrap _auto_select_speaker (sync version) with a parent span
    old_auto_select_speaker = groupchat._auto_select_speaker

    def auto_select_speaker_traced(
        last_speaker: Agent,
        selector: ConversableAgent,
        messages: list[dict[str, Any]] | None,
        agents: list[Agent] | None,
    ) -> Agent:
        with tracer.start_as_current_span("speaker_selection") as span:
            span.set_attribute("ag2.span.type", SpanType.SPEAKER_SELECTION.value)
            span.set_attribute("gen_ai.operation.name", "speaker_selection")

            # Record candidate agents
            candidate_agents = agents if agents is not None else groupchat.agents
            span.set_attribute(
                "ag2.speaker_selection.candidates",
                json.dumps([a.name for a in candidate_agents]),
            )

            result = old_auto_select_speaker(last_speaker, selector, messages, agents)

            # Record selected speaker
            span.set_attribute("ag2.speaker_selection.selected", result.name)
            return result

    groupchat._auto_select_speaker = auto_select_speaker_traced

    return groupchat
