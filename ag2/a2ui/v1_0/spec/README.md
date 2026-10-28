# A2UI v1.0 vendored spec

These JSON schemas are vendored from the upstream A2UI project.

- **Source repository:** https://github.com/a2ui-project/a2ui
- **Source paths:** `specification/v1_0/json/*.json` and
  `catalogs/basic/v1/catalog.json`
- **Pinned commit:** `ba4c47baba5663037217f107d5bcc51f6d35c536` (`main`, 2026-10-19)
- **Spec status:** v1.0 is an upstream **release candidate** — re-pin and re-vendor when it stabilizes.

## Local adaptations

Only the catalog file is renamed on disk (`catalogs/basic/v1/catalog.json` →
`basic_catalog.json`) to match this module's per-version layout. The file
**contents are byte-faithful** to upstream — `$id`
(`https://a2ui.org/specification/v1_0/catalogs/basic/catalog.json`),
`catalogId`, and all `$ref`s are unchanged. The canonical catalog id resolves
on a2ui.org and must match what renderers advertise in `supportedCatalogIds`.

Unlike v0.9, the schema files use the upstream *agent / renderer* names
(`agent_to_renderer.json`, `renderer_to_agent.json`, `agent_capabilities.json`,
`renderer_capabilities.json`, `renderer_data_model.json`) rather than
*server / client*.

## What v1.0 changes over v0.9

See upstream `specification/v1_0/docs/evolution_guide.md`. The ones that shape
this integration:

- **Agent → renderer** messages are `createSurface`, `updateComponents`,
  `updateDataModel`, `deleteSurface`, `callRendererFunction` and
  `agentFunctionResponse`. `createSurface` may carry the initial `components`
  and `dataModel`; `theme` is gone and `catalogId` is optional.
- **Renderer → agent** messages are `action`, `callAgentFunction`,
  `rendererFunctionResponse` and `error`. The v0.9.x-draft `wantResponse` /
  `actionId` / `actionResponse` mechanism does not exist: request/response goes
  through `callAgentFunction` → `agentFunctionResponse`, correlated by
  `functionCallId`.
- Data bindings and function calls use the reserved `@path` / `@call` keys.
- `updateDataModel.value` is required; `null` deletes the key.
- Catalog guidance lives in the catalog's `instructions` field; there is no
  `basic_catalog_rules.txt`.
