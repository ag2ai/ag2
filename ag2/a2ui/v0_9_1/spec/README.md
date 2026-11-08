# A2UI v0.9.1 vendored spec

These JSON schemas are vendored **as-is** from the upstream A2UI project.

- **Source repository:** https://github.com/a2ui-project/a2ui
- **Source paths:** `specification/v0_9_1/json/*.json` and
  `specification/v0_9_1/catalogs/basic/catalog.json` (+ `rules.txt`)
- **Pinned commit:** `078b3d5c4fe5541e4390baad1ec5995c6bba6857` (upstream `main`,
  2026-11-08)
- **Spec status:** v0.9.1 is a **previous** upstream version; v1.0 is current
  (`specification/versions.json`).

## Local adaptations

Only the catalog file is renamed on disk (`catalogs/basic/catalog.json` →
`basic_catalog.json`) to match this module's per-version layout. The file
**contents are byte-faithful** to upstream — `$id`, `catalogId`, and all
`$ref`s are unchanged.

## Relationship to v0.9

v0.9.1 is a backward-compatible patch over v0.9:

- The catalog has the same 18 components and the same canonical `$id`
  `https://a2ui.org/specification/v0_9/catalogs/basic/catalog.json` as v0.9;
  only its descriptions differ.
- The schema `$id`s point at `v0_9` (not `v0_9_1`) — v0.9.1 reuses v0.9's
  identifiers, so `schema_base_uri` and `default_catalog_id` are shared.
- The only behavioural change: the `version` field accepts the enum
  `["v0.9", "v0.9.1"]` (v0.9 used `const: "v0.9"`). This module emits and
  prompts for `"v0.9.1"`.
