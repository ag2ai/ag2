from collections.abc import Sequence
from typing import Any, Optional

from a2a.utils.telemetry import SpanKind
from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Attributes, Resource
from opentelemetry.sdk.trace import Tracer, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.sampling import Decision, Sampler, SamplingResult
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from autogen import ConversableAgent
from autogen.a2a import A2aAgentServer
from autogen.agentchat import Agent
from autogen.agentchat.group import ContextVariables
from autogen.agentchat.group.group_tool_executor import GroupToolExecutor
from autogen.agentchat.group.patterns.pattern import Pattern
from autogen.agentchat.group.targets.transition_target import TransitionTarget
from autogen.agentchat.groupchat import GroupChat, GroupChatManager
from autogen.version import __version__ as AG2_VERSION

OTEL_SCHEMA = "https://opentelemetry.io/schemas/1.11.0"
INSTRUMENTING_MODULE_NAME = "opentelemetry.instrumentation.ag2"
INSTRUMENTING_LIBRARY_VERSION = AG2_VERSION

_TRACE_PROPAGATOR = TraceContextTextMapPropagator()


class DropNoiseSampler(Sampler):
    def should_sample(
        self,
        parent_context: Optional["Context"],
        trace_id: int,
        name: str,
        kind: SpanKind | None = None,
        attributes: Attributes = None,
        links: Sequence["trace.Link"] | None = None,
        trace_state: trace.TraceState | None = None,
    ) -> "SamplingResult":
        decision = Decision.RECORD_ONLY if name.startswith("a2a.") else Decision.RECORD_AND_SAMPLE
        return SamplingResult(decision, attributes=None, trace_state=trace_state)

    def get_description(self) -> str:
        return "Drop a2a.server noisy spans"


def setup_instrumentation(service_name: str, endpoint: str = "http://127.0.0.1:4317") -> Tracer:
    resource = Resource.create(attributes={"service.name": service_name})
    tracer_provider = TracerProvider(
        resource=resource,
        # sampler=DropNoiseSampler(),
    )
    exporter = OTLPSpanExporter(endpoint=endpoint)
    processor = BatchSpanProcessor(exporter)
    tracer_provider.add_span_processor(processor)
    trace.set_tracer_provider(tracer_provider)

    return tracer_provider.get_tracer(
        instrumenting_module_name=INSTRUMENTING_MODULE_NAME,
        instrumenting_library_version=INSTRUMENTING_LIBRARY_VERSION,
        schema_url=OTEL_SCHEMA,
    )


def instrument_a2a_server(server: A2aAgentServer, tracer: Tracer) -> A2aAgentServer:
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request

    class OTELMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            span_context = _TRACE_PROPAGATOR.extract(request.headers)
            with tracer.start_as_current_span("a2a-execution", context=span_context):
                return await call_next(request)

    server.add_middleware(OTELMiddleware)

    instrument_agent(server.agent, tracer)
    return server


def instrument_pattern(pattern: Pattern, tracer: Tracer) -> Pattern:
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
            instrument_agent(agent, tracer)

        instrument_agent(manager, tracer)

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


def instrument_agent(agent: Agent, tracer: Tracer) -> Agent:
    # Instrument `a_generate_reply`
    old_a_generate_reply = agent.a_generate_reply

    async def a_generate_traced_reply(
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        with tracer.start_as_current_span(f"{agent.name}.a_generate_reply"):
            return await old_a_generate_reply(*args, **kwargs)

    agent.a_generate_reply = a_generate_traced_reply

    # Instrument `a_initiate_traced_chat`
    if hasattr(agent, "a_initiate_chat"):
        old_a_initiate_chat = agent.a_initiate_chat

        async def a_initiate_traced_chat(
            *args: Any,
            max_turns: int | None = None,
            **kwargs: Any,
        ) -> Any:
            with tracer.start_as_current_span(f"{agent.name}.a_initiate_chat") as rollspan:
                if max_turns:
                    rollspan.set_attribute("chat.max_turns", max_turns)

                result = await old_a_initiate_chat(*args, **kwargs)

                rollspan.set_attribute("chat.turns", len(result.chat_history))

                usage_including_cached_inference = result.cost["usage_including_cached_inference"]
                total_cost = usage_including_cached_inference.pop("total_cost")
                rollspan.set_attribute("chat.cost", total_cost)

                if usage_including_cached_inference:
                    model, cost_data = next(iter(usage_including_cached_inference.items()))

                    rollspan.set_attributes({
                        "chat.model": model,
                        "chat.cost.total_tokens": cost_data["total_tokens"],
                        "chat.cost.prompt_tokens": cost_data["prompt_tokens"],
                        "chat.cost.completion_tokens": cost_data["completion_tokens"],
                    })

                return result

        agent.a_initiate_chat = a_initiate_traced_chat

    # Instrument `a_resume`
    if hasattr(agent, "a_resume"):
        old_a_resume = agent.a_resume

        async def a_resume_traced(
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            with tracer.start_as_current_span(f"{agent.name}.a_resume"):
                return await old_a_resume(*args, **kwargs)

        agent.a_resume = a_resume_traced

    # Instrument `a_run_chat`
    if hasattr(agent, "a_run_chat"):
        old_a_run_chat = agent.a_run_chat

        async def a_run_chat_traced(
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            with tracer.start_as_current_span(f"{agent.name}.a_run_chat"):
                return await old_a_run_chat(*args, **kwargs)

        agent.a_run_chat = a_run_chat_traced

    # Instrument `a_generate_remote_reply`
    if hasattr(agent, "a_generate_remote_reply"):
        old_httpx_client_factory = agent._httpx_client_factory

        def httpx_client_factory_traced():
            httpx_client = old_httpx_client_factory()
            _TRACE_PROPAGATOR.inject(httpx_client.headers)
            return httpx_client

        agent._httpx_client_factory = httpx_client_factory_traced

        old_a_generate_remote_reply = agent.a_generate_remote_reply

        async def a_generate_remote_reply_traced(
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            with tracer.start_as_current_span(f"{agent.name}.a_generate_remote_reply"):
                return await old_a_generate_remote_reply(*args, **kwargs)

        agent.a_generate_remote_reply = a_generate_remote_reply_traced

    return agent
