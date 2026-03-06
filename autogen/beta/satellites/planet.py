# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable
from contextlib import ExitStack
from typing import Any
from uuid import uuid4

from autogen.beta.agent import Agent, Conversation, PromptType
from autogen.beta.annotations import Context
from autogen.beta.config import LLMClient, ModelConfig
from autogen.beta.events import (
    BaseEvent,
    HumanInputRequest,
    ModelRequest,
    ModelResponse,
    ToolResults,
)
from autogen.beta.hitl import HumanHook, default_hitl_hook, wrap_hitl
from autogen.beta.stream import MemoryStream, Stream
from autogen.beta.tools import Tool, ToolExecutor, tool

from .events import (
    SatelliteCompleted,
    SatelliteFlag,
    SatelliteStarted,
    TaskSatelliteProgress,
    TaskSatelliteRequest,
    TaskSatelliteResult,
)
from .satellite import Satellite


class PlanetAgent(Agent):
    """An agent that orchestrates satellite observers and task workers.

    Extends :class:`Agent` with:

    * **Natural-born satellites** – plug-in observers that monitor the event
      stream and flag issues (cost, loops, safety, …).
    * **Task satellites** – on-demand sub-agents spawned via the built-in
      ``spawn_task`` / ``spawn_tasks`` tools.
    * **Flag injection** – satellite flags are drained into the LLM system
      prompt before each model call so the planet can react.

    Parameters
    ----------
    name:
        Agent display name.
    prompt:
        System prompt(s) – static strings or async callables.
    config:
        LLM configuration for the planet agent.
    satellites:
        Iterable of :class:`Satellite` plug-ins to attach.
    satellite_config:
        Default :class:`ModelConfig` used by spawned task satellites.
        Falls back to *config* if not provided.
    satellite_prompt:
        System prompt for task satellites.
    hitl_hook:
        Human-in-the-loop callback.
    tools:
        Tools available to the planet agent.
    dependencies:
        Dependency injection values.
    variables:
        Serializable variables.
    """

    def __init__(
        self,
        name: str,
        prompt: PromptType | Iterable[PromptType] = (),
        *,
        config: ModelConfig | None = None,
        satellites: Iterable[Satellite] = (),
        satellite_config: ModelConfig | None = None,
        satellite_prompt: str = (
            "You are a task satellite. Complete the assigned task "
            "thoroughly and concisely. Return only the result."
        ),
        hitl_hook: HumanHook | None = None,
        tools: Iterable[Callable[..., Any] | Tool] = (),
        dependencies: dict[Any, Any] | None = None,
        variables: dict[Any, Any] | None = None,
    ) -> None:
        super().__init__(
            name,
            prompt,
            config=config,
            hitl_hook=hitl_hook,
            tools=tools,
            dependencies=dependencies,
            variables=variables,
        )
        self._satellites: list[Satellite] = list(satellites)
        self._satellite_config: ModelConfig | None = satellite_config or config
        self._satellite_prompt = satellite_prompt

        # Own copies – avoids accessing name-mangled parent attrs.
        self._planet_hitl = wrap_hitl(hitl_hook) if hitl_hook else default_hitl_hook
        self._planet_executor = ToolExecutor()

    # ------------------------------------------------------------------
    # Satellite management
    # ------------------------------------------------------------------

    def add_satellite(self, satellite: Satellite) -> None:
        """Register a satellite plug-in (before calling :meth:`ask`)."""
        self._satellites.append(satellite)

    # ------------------------------------------------------------------
    # Internal: spawn helpers
    # ------------------------------------------------------------------

    def _build_spawn_tools(self) -> list[Tool]:
        """Create the ``spawn_task`` and ``spawn_tasks`` tools."""
        planet = self

        @tool
        async def spawn_task(task: str, ctx: Context) -> str:
            """Spawn a task satellite to handle a subtask autonomously.

            Use this when a sub-problem can be researched or solved
            independently.  The satellite runs with its own LLM context
            and returns the result.
            """
            return await planet._run_task_satellite(task, ctx)

        @tool
        async def spawn_tasks(ctx: Context, tasks: list[str], parallel: bool = True) -> str:
            """Spawn multiple task satellites at once.

            Args:
                tasks: List of subtask descriptions.
                parallel: Run concurrently when True (default), sequentially
                    when False.
            """
            if parallel:
                results = await asyncio.gather(
                    *(planet._run_task_satellite(t, ctx) for t in tasks)
                )
            else:
                results = [await planet._run_task_satellite(t, ctx) for t in tasks]

            parts = []
            for task_desc, result in zip(tasks, results):
                parts.append(f"## {task_desc}\n\n{result}")
            return "\n\n---\n\n".join(parts)

        return [spawn_task, spawn_tasks]

    async def _run_task_satellite(self, task: str, ctx: Context) -> str:
        """Run a single task satellite and emit lifecycle events.

        Bridges ``ModelMessageChunk`` events from the satellite's stream
        to the parent stream as ``TaskSatelliteProgress``, giving callers
        real-time visibility into satellite work.
        """
        from autogen.beta.events import ModelMessageChunk

        satellite_name = f"task-{uuid4().hex[:8]}"

        await ctx.send(TaskSatelliteRequest(task=task, satellite_name=satellite_name))

        sat_stream = MemoryStream()
        parent_ctx = ctx  # capture for the bridge closure

        # Bridge: forward streaming chunks to parent as progress events
        async def _bridge_chunks(chunk: ModelMessageChunk) -> None:
            await parent_ctx.send(
                TaskSatelliteProgress(
                    satellite_name=satellite_name,
                    content=chunk.content,
                )
            )

        sat_stream.where(ModelMessageChunk).subscribe(_bridge_chunks)

        satellite = Agent(
            satellite_name,
            prompt=self._satellite_prompt,
            config=self._satellite_config,
        )
        conversation = await satellite.ask(task, stream=sat_stream)

        result_text = ""
        if conversation.message and conversation.message.message:
            result_text = conversation.message.message.content

        usage = conversation.message.usage if conversation.message else {}
        await ctx.send(
            TaskSatelliteResult(
                task=task,
                satellite_name=satellite_name,
                result=result_text,
                usage=usage,
            )
        )
        return result_text

    # ------------------------------------------------------------------
    # Flag helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_flags(flags: list[SatelliteFlag]) -> str:
        header = "[SATELLITE MONITORING ALERTS]"
        lines = []
        for f in flags:
            lines.append(f"- [{f.severity.upper()}] ({f.source}): {f.message}")
        return f"{header}\n" + "\n".join(lines)

    # ------------------------------------------------------------------
    # Core execution override
    # ------------------------------------------------------------------

    async def _execute(
        self,
        event: BaseEvent,
        *,
        ctx: Context,
        client: LLMClient,
        additional_tools: Iterable[Tool] = (),
    ) -> Conversation:
        spawn_tools = self._build_spawn_tools()
        all_tools = self.tools + list(additional_tools) + spawn_tools

        # Flag collection queue
        flag_queue: list[SatelliteFlag] = []

        async def _collect_flag(event: SatelliteFlag) -> None:
            flag_queue.append(event)

        flag_sub = ctx.stream.where(SatelliteFlag).subscribe(_collect_flag)

        # Attach natural-born satellites
        for sat in self._satellites:
            sat.attach(ctx.stream, ctx)
            await ctx.send(SatelliteStarted(name=sat.name))

        try:
            # LLM caller that drains flags before each call
            async def _call_client(event: BaseEvent, ctx: Context) -> None:
                if flag_queue:
                    flags = flag_queue[:]
                    flag_queue.clear()
                    ctx.prompt.append(self._format_flags(flags))

                await client(
                    *await ctx.stream.history.get_events(),
                    ctx=ctx,
                    tools=all_tools,
                )

                # Remove temporary flag prompt so it doesn't persist
                if ctx.prompt and ctx.prompt[-1].startswith("[SATELLITE"):
                    ctx.prompt.pop()

            with ExitStack() as stack:
                stack.enter_context(
                    ctx.stream.where(ModelRequest | ToolResults).sub_scope(
                        _call_client
                    ),
                )
                stack.enter_context(
                    ctx.stream.where(HumanInputRequest).sub_scope(
                        self._planet_hitl
                    ),
                )
                self._planet_executor.register(stack, ctx, tools=all_tools)

                async with ctx.stream.get(ModelResponse) as result:
                    await ctx.send(event)
                    message = await result

                while message.tool_calls and not message.response_force:
                    async with ctx.stream.get(ModelResponse) as result:
                        await ctx.send(message.tool_calls)
                        message = await result

                return Conversation(
                    message,
                    ctx=ctx,
                    agent=self,
                    client=client,
                )
        finally:
            for sat in self._satellites:
                sat.detach()
                await ctx.send(SatelliteCompleted(name=sat.name))
            ctx.stream.unsubscribe(flag_sub)
