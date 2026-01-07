"""Test file for initiate_chats (sequential multi-chat) tracing."""

import os

from dotenv import load_dotenv
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from autogen import ConversableAgent, LLMConfig
from autogen.opentelemetry import instrument_agent, instrument_chats, instrument_llm_wrapper

load_dotenv()

llm_config = LLMConfig(
    config_list={
        "api_type": "openai",
        "model": "gpt-4o-mini",
        "api_key": os.getenv("OPENAI_API_KEY"),
    }
)

curriculum_message = """You are a curriculum designer for a fourth grade class.
Nominate an appropriate a topic for a lesson, based on the given subject. Be very brief."""

planner_message = """You are a classroom lesson agent.
Given a topic, write a brief lesson plan for a fourth grade class in bullet points.
Include the title and 2-3 learning objectives. Keep it short."""

formatter_message = """You are a lesson plan formatter. Format the complete plan as follows:
<title>Lesson plan title</title>
<objectives>Key learning objectives</objectives>
Say DONE when complete."""

teacher_message = """You are a classroom teacher.
You decide topics for lessons and work with specialists. Be concise."""

lesson_curriculum = ConversableAgent(
    name="curriculum_agent",
    system_message=curriculum_message,
    llm_config=llm_config,
    human_input_mode="NEVER",
)

lesson_planner = ConversableAgent(
    name="planner_agent",
    system_message=planner_message,
    llm_config=llm_config,
    human_input_mode="NEVER",
)

lesson_formatter = ConversableAgent(
    name="formatter_agent",
    system_message=formatter_message,
    llm_config=llm_config,
    human_input_mode="NEVER",
)

teacher = ConversableAgent(
    name="teacher_agent",
    system_message=teacher_message,
    llm_config=llm_config,
    human_input_mode="NEVER",
)


def main():
    """Run sequential chats using initiate_chats."""
    resource = Resource.create(attributes={"service.name": "local-initiate-chats"})
    tracer_provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint="http://127.0.0.1:14317")
    processor = BatchSpanProcessor(exporter)
    tracer_provider.add_span_processor(processor)
    trace.set_tracer_provider(tracer_provider)
    instrument_llm_wrapper(tracer_provider=tracer_provider)

    # Instrument the initiate_chats function (adds parent span)
    instrument_chats(tracer_provider=tracer_provider)

    # Instrument all agents
    instrument_agent(teacher, tracer_provider=tracer_provider)
    instrument_agent(lesson_curriculum, tracer_provider=tracer_provider)
    instrument_agent(lesson_planner, tracer_provider=tracer_provider)
    instrument_agent(lesson_formatter, tracer_provider=tracer_provider)

    # Run sequential chats
    chat_results = teacher.initiate_chats([
        {
            "recipient": lesson_curriculum,
            "message": "Let's create a science lesson, what's a good topic?",
            "max_turns": 1,
            "summary_method": "last_msg",
        },
        {
            "recipient": lesson_planner,
            "message": "Create a brief lesson plan.",
            "max_turns": 1,
            "summary_method": "last_msg",
        },
        {
            "recipient": lesson_formatter,
            "message": "Format the lesson plan.",
            "max_turns": 1,
            "summary_method": "last_msg",
        },
    ])

    print("\n\n=== RESULTS ===")
    print("\nCurriculum summary:\n", chat_results[0].summary)
    print("\nLesson Planner summary:\n", chat_results[1].summary)
    print("\nFormatter summary:\n", chat_results[2].summary)


if __name__ == "__main__":
    main()
