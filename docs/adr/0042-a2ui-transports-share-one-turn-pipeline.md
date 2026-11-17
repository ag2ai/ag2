---
status: accepted
date: 2026-11-18
---

# 0042. Every A2UI transport runs its turn through one pipeline; transports only translate their wire

## Context

REST and AG-UI ran an A2UI turn through `ag2.a2ui.dispatch`, and A2A ran it through its own copy in
`A2UIAgentExecutor`. Each copy did the same work:

- decoded the renderer's envelopes, capabilities and data model;
- negotiated the turn's offer (ADR 0041);
- answered the interactions the server handles itself;
- built the A2UI prompt fragments and middleware;
- filtered what may be sent.

The renderer-side decoding was repeated in each transport's request mapping as well. The copies drifted twice: the
same no-UI guard ended up in three places, and an A2A-only fallback differed for v1.0. Every rule change about what a
turn may do became a multi-file edit.

## Decision

- `ag2.a2ui.incoming.read_renderer_input` decodes what the renderer sent with one request into an
  `A2UIRendererInput`. A transport says only where the envelopes, capabilities and data model sit on its wire. A2A
  passes one envelope group per DataPart, so each part's prompts still take its place.
- `ag2.a2ui._turn.open_turn(runtime, renderer_input)` returns an `A2UITurn`, which owns the A2UI rules of one turn:
  - the negotiated offer;
  - its prompt fragments (`system_prompt`) and `middleware`;
  - the server-side answers (`answer_interactions`);
  - the filter on the agent's messages (`deliverable`).

  No transport negotiates offers, builds A2UI prompt fragments or filters messages itself.
- Running the agent stays with each transport: `dispatch` for REST and AG-UI, and the base `AgentExecutor` for A2A,
  which owns the task lifecycle, client tools and HITL. The pipeline does not run the agent.

## Consequences

- A rule about what a turn may send is changed in one module and holds on every transport.
- A new transport writes a wire adapter and picks an agent runner. It does not copy turn logic.
- The pipeline cannot assume how the agent runs, so it exposes `system_prompt`, `middleware` and `collect` rather
  than a single `run()`. Merging the two agent runners would need a new ADR.
