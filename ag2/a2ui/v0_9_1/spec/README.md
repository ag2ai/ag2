# A2UI v0.9.1 vendored spec

These JSON schemas are vendored **as-is** from the upstream A2UI project.

- **Source repository:** https://github.com/a2ui-project/a2ui
- **Source paths:** `specification/v0_9_1/json/*.json` and
  `specification/v0_9_1/catalogs/basic/catalog.json` (+ `rules.txt`)
- **Pinned commit:** `d8af0c56cd8e211b4f7d5c168efb56828d7aae5f`
- **Spec status:** v0.9.1 is the upstream **current/stable** release.

## Local adaptations

Only the catalog file is renamed on disk (`catalogs/basic/catalog.json` →
`basic_catalog.json`) to match this module's per-version layout. The file
**contents are byte-faithful** to upstream — `$id`, `catalogId`, and all
`$ref`s are unchanged.

## Relationship to v0.9

v0.9.1 is a backward-compatible patch over v0.9:

- The catalog is **identical** to v0.9 (same 18 components, same canonical
  `$id` `https://a2ui.org/specification/v0_9/catalogs/basic/catalog.json`).
- The schema `$id`s point at `v0_9` (not `v0_9_1`) — v0.9.1 reuses v0.9's
  identifiers, so `schema_base_uri` and `default_catalog_id` are shared.
- The only behavioural change: the `version` field accepts the enum
  `["v0.9", "v0.9.1"]` (v0.9 used `const: "v0.9"`). This module emits and
  prompts for `"v0.9.1"`.
