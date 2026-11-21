---
status: accepted
date: 2026-11-17
---

# 0041. A2UI offers are negotiated per turn, and a turn with no common catalog runs without A2UI

## Context

An `A2UIServer` or `A2UIAgentExecutor` used to build one prompt and one validator at construction and use them on
every turn. A renderer's `supportedCatalogIds` only added a note to the prompt ("the client did NOT list this agent's
catalog"), so the model could still build surfaces on a catalog the renderer cannot draw, and the server sent them.

Upstream A2UI (`6e79b2e`) and its reference SDK say more:

- The official SDK's `resolve_catalogs` (`python/a2ui_agent/src/a2ui/utils/catalog_resolver.py` L50) activates
  only the registered catalogs the renderer names, and the prompt and the validator both use only those. When none is
  in common it raises `A2uiCatalogError`. Without capabilities it activates every registered catalog. Inline catalogs
  become active only when the agent opts in (`accepts_inline_catalogs`).
- v1.0 `a2ui_protocol.md` L1630 ("Renderer capabilities"): "Capabilities carried per message are scoped to that
  message's turn, and omitting them withdraws A2UI for that turn; capabilities carried at session setup are scoped
  to the session. Each transport binding states which applies." The A2A extension states it for A2A
  (`specification/v1_0/extensions/a2a/docs/a2ui_extension_specification.md` L144, "Capability scoping"): "An agent
  MUST treat a message that carries no `a2uiRendererCapabilities` entry for its version as a turn in which the
  renderer offers no A2UI, and MUST NOT send `application/a2ui+json` DataParts in reply to it." Every ag2 transport
  (REST, AG-UI, A2A) carries capabilities per request. v0.9 has no such rule.
- v1.0 L1486 ("callAgentFunction"): the agent "MUST return an `agentFunctionResponse`" to every call.

## Decision

Each turn gets an `A2UIOffer`, built from the renderer's capabilities for the served version:

- Without capabilities, v1.0 offers no A2UI on any transport, per L1630: the renderer has withdrawn it, so not even
  an `agentFunctionResponse` is sent. Such a turn offers no catalog, so a `callAgentFunction` naming one does not
  select a handler: nobody waits for the answer, and a side-effecting handler should not run for nothing. Before v1.0 every catalog the agent serves is offered, as the SDK does.
- With capabilities, only the agent's catalogs the renderer names are offered. The prompt, the validator, the catalog
  `createSurface` names in the prompt and the catalogs a `callAgentFunction` may name all narrow to them. Offers are
  cached per catalog subset, so the cost is paid once per subset.
- With no catalog in common the turn **runs without A2UI** and logs a warning, where the SDK raises. The prompt has no
  A2UI section, server-side handlers still run but their surface messages are dropped, and any A2UI the model emits
  anyway is extracted from the prose and dropped. A `callAgentFunction` still gets its `agentFunctionResponse` (L1486):
  the renderer has not withdrawn A2UI, only shares no catalog, and it is waiting for the answer. The call can only
  name one of the renderer's catalogs, so its handler is matched by `@call` name alone (ADR 0040), and
  `UNKNOWN_FUNCTION` means only that no handler has that name (L1486). The user still gets a text answer, which a
  request error would not give them.
- Inline catalogs are never offered. ag2 has no opt-in for them, and serving a catalog the renderer defines would
  validate UI against a schema the deployment did not review.

## Consequences

- Breaking: a renderer that lists catalogs the agent does not serve, and none it does, used to get UI with a warning in
  the prompt and now gets none. A v1.0 client that sends no `a2uiRendererCapabilities`, over any transport, gets no
  A2UI.
- The capabilities prompt section still lists every catalog the renderer names, including ones the agent does not
  serve; only the served ones are in the A2UI section.
- Supporting inline catalogs later needs an explicit opt-in and a new ADR.
