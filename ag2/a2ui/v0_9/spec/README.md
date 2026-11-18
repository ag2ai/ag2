# A2UI v0.9 vendored spec

These JSON schemas are vendored **as-is** from the upstream A2UI project.

- **Source repository:** https://github.com/a2ui-project/a2ui
- **Source paths:** eight of the eleven files in `specification/v0_9/json/` (all but
  `client_to_server_list_wrapper.json`, `server_to_client_list_wrapper.json` and `sample.json`), plus
  `specification/v0_9/catalogs/basic/catalog.json` and `rules.txt`
- **Pinned commit:** `6e79b2e20fe2e1543aff3b9928613daa2b37189c` (upstream `main`, 2026-11-16). It includes
  `List.listStyle` in the basic catalog (a2ui-project/a2ui#3956)
- **Spec status:** v0.9 is a **previous** upstream version; v0.9.1 is the patch
  over it and v1.0 is current.

## Local adaptations

Only the catalog files are renamed on disk (`catalogs/basic/catalog.json` →
`basic_catalog.json`, `catalogs/basic/rules.txt` → `basic_catalog_rules.txt`) to match
this module's per-version layout. All ten vendored files (the eight message and type
schemas, the catalog and its rules) are **byte-faithful** to the pinned commit —
`$id`, `catalogId`, and all `$ref`s are unchanged.

ag2 loads only `server_to_client.json`, `common_types.json`, `basic_catalog.json` and
`basic_catalog_rules.txt`. The other schemas (`client_to_server*.json`,
`client_data_model.json`, `client_capabilities.json`, `server_capabilities.json`,
`server_to_client_list.json`) are kept for reference; incoming envelopes, capabilities
and data models are decoded structurally, not validated against them.

## Relationship to v0.9.1 / v1.0

v0.9.1 is served from these files too. At the pinned commit, upstream's v0.9.1 files
(`specification/v0_9_1/`, last changed in `ae97628de674407fd55eb76966d0b8eddb3a8497`) differ from these in
two places only:

- the messages' `version` field accepts the enum `["v0.9", "v0.9.1"]` instead of `const: "v0.9"`
  (in `server_to_client.json`, `client_to_server.json` and `client_data_model.json`).
  `ag2/a2ui/schema_manager/_spec.py` (`_accept_wire_versions`) applies that widening at load time to
  `server_to_client.json`, the only one of the three ag2 validates against;
- upstream's v0.9.1 basic catalog lacks `List.listStyle` (a2ui-project/a2ui#3956 changed only the v0.9 copy).
  ag2 accepts and describes `listStyle` for v0.9.1 too. Both copies carry the same catalog `$id` (`v0_9`), so
  the newer content under that id wins, as in the official SDK (`a2ui-core`), which also serves v0.9.1 from its
  v0.9 models.

v1.0 is a breaking revision with its own wire version (`"1.0"`), schemas and
catalog; see `../../v1_0/spec/README.md`.
