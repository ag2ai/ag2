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
- Identity: "Each catalog is identified by a `catalogId` string" (v0.9 L317), and "catalog definitions should include
  both `$id` (used by JSON Schema tooling) and `catalogId` (used by A2UI SDKs and catalog negotiation), setting both
  fields to the same URI" (v1.0 L543; v0.9 L319). v1.0 `catalog_definition.json` requires `catalogId`, and the
  reference SDK (`a2ui_agent/processor/catalog_providers.py` `_settle_catalog_id`) takes the id from it only.
- Version: `protocolVersion` is "optional for backward compatibility; defaults to `"0.9"` if omitted, required for
  catalogs targeting `1.0` and beyond" (v1.0 L635).

## Decision

Each catalog is validated as written; nothing is merged.

- A custom catalog is identified by its `catalogId`, a non-empty string, in every version. A `$id` is optional but
  must equal it: the spec's SHOULD becomes a MUST, because the catalog's `$ref`s resolve against `$id` and it is
  served under `catalogId`, and two names for one catalog would split them. Internally a catalog without `$id` is
  keyed by its `catalogId`.
- Its `protocolVersion`, `"0.9"` when absent, must have the served version's major.minor: a v1.0 agent needs
  `"protocolVersion": "1.0"`, and a v0.9 agent rejects a 1.0 catalog. The SDK rejects a mismatch too, but takes the
  agent's version when the catalog states none; ag2 follows L635 instead (ADR 0039).
- A v1.0 custom catalog must satisfy the vendored `catalog_definition.json`, resolved with `common_types.json`. Its
  `\p{XID_Start}` extension-key pattern does not compile in Python `re`, so those keys are checked with
  `str.isidentifier` instead. The UAX #31 name and reserved `Surface` checks run first, for clearer messages; a
  schema violation is reported with its JSON pointer.

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

- Breaking for users of `custom_catalog`: it needs a `catalogId` (a `$id`-only catalog is rejected), and a v1.0 one
  also `"protocolVersion": "1.0"`, a schema-valid shape (`returnType` and an `@call` `const` on every function,
  `anyComponent` and `anyFunction` in any `$defs`) and only the root keys `catalog_definition.json` allows. In v0.9 a catalog that relied on the merge must declare the basic
  components it uses (`"Text": {"$ref": "basic_catalog.json#/components/Text"}`). In v1.0 basic components on a
  surface created with the custom id need `@catalogId`.
- The component graph reads references from every served catalog, following `$ref`s into `basic_catalog.json`; when
  both v1.0 catalogs define a component type, the custom definition decides which properties are references.
- The capability note in the prompt checks only the catalog surfaces are created with, not every served catalog.
