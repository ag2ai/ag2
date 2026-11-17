---
status: accepted
date: 2026-11-17
---

# 0040. A2UI custom catalogs follow the spec instead of being merged into the basic one

## Context

`A2UIServer(custom_catalog=...)` used to merge the custom catalog over the basic one and serve the union under the
custom `$id`. A surface created with that id accepted every basic component without `@catalogId`. A renderer that
registers the custom catalog as written does not have those components, so ag2 accepted UI the renderer could not
draw.

Upstream A2UI (`6e79b2e`) says something different in each version:

- v0.9 (`specification/v0_9/docs/a2ui_protocol.md` L157): to use your own catalog, "include it in the prompt in
  place of the basic catalog". A surface has one catalog. A v0.9 catalog may still `$ref` basic definitions.
- v1.0 (`specification/v1_0/docs/a2ui_protocol.md` L511-522): catalogs are mixable. A component resolves against its
  own `@catalogId` or its surface's `catalogId`, with "no fallback". A catalog may not `$ref` another catalog.

## Decision

Each catalog is validated as written; nothing is merged.

- v0.9 and v0.9.1: a custom catalog replaces the basic one. It is the only catalog a surface may name, the prompt
  lists only its components, and its rules replace the basic catalog's. The prompt example keeps only the messages
  whose components the catalog has.
- v1.0: the basic and the custom catalog are both served. The basic catalog is the surface default in the prompt,
  and the prompt tells the model to give custom components `@catalogId` and custom function calls `catalogId`.
- A `callAgentFunction` is matched to an `@a2ui_action` handler by its `@call` name. Its `catalogId`, which the
  renderer MUST set to the catalog the call resolved to (L1495), may be absent or any catalog the turn offers
  (ADR 0041). This is ag2's choice: an `@a2ui_action` handler is not declared in any catalog, and the renderer
  routes undeclared functions to the agent (L1484), so the spec ties the handler to no catalog.

## Consequences

- Breaking for users of `custom_catalog`. In v0.9 a catalog that relied on the merge must declare the basic
  components it uses (`"Text": {"$ref": "basic_catalog.json#/components/Text"}`). In v1.0 basic components on a
  surface created with the custom id need `@catalogId`.
- The component graph reads references from every served catalog, following `$ref`s into `basic_catalog.json`; when
  both v1.0 catalogs define a component type, the custom definition decides which properties are references.
- The capability note in the prompt checks only the catalog surfaces are created with, not every served catalog.
