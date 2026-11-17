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
- The A2A extension (`specification/v1_0/extensions/a2a/docs/a2ui_extension_specification.md` L144, "Capability
  scoping"): "An agent MUST treat a message that carries no `a2uiRendererCapabilities` entry for its version as a turn
  in which the renderer offers no A2UI, and MUST NOT send `application/a2ui+json` DataParts in reply to it." Earlier
  versions and the REST and AG-UI transports set no such rule.

## Decision

Each turn gets an `A2UIOffer`, built from the renderer's capabilities for the served version:

- Without capabilities, every catalog the agent serves is offered, as the SDK does. Exception: A2A v1.0, where the
  turn offers no A2UI, per the MUST above.
- With capabilities, only the agent's catalogs the renderer names are offered. The prompt, the validator, the catalog
  `createSurface` names in the prompt and the catalogs a `callAgentFunction` may name all narrow to them. Offers are
  cached per catalog subset, so the cost is paid once per subset.
- With no catalog in common the turn **runs without A2UI** and logs a warning, where the SDK raises. The prompt has no
  A2UI section, server-side handlers still run but their messages are dropped, and any A2UI the model emits anyway is
  extracted from the prose and dropped. The user still gets a text answer, which a request error would not give them.
- Inline catalogs are never offered. ag2 has no opt-in for them, and serving a catalog the renderer defines would
  validate UI against a schema the deployment did not review.

## Consequences

- Breaking: a renderer that lists catalogs the agent does not serve, and none it does, used to get UI with a warning in
  the prompt and now gets none. An A2A v1.0 client that sends no `a2uiRendererCapabilities` gets no A2UI.
- The capabilities prompt section still lists every catalog the renderer names, including ones the agent does not
  serve; only the served ones are in the A2UI section.
- Supporting inline catalogs later needs an explicit opt-in and a new ADR.
