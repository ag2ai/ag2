---
status: accepted
date: 2026-11-10
---

# 0037. MCP app runtime injection reads the document's running scripts

## Context

`MCPApp` injects the `ag2ui` runtime into every document by default, and the
runtime performs the MCP Apps `ui/initialize` handshake. A document written
against the upstream `App` class from `@modelcontextprotocol/ext-apps` performs
the same handshake itself, so injecting into it makes the host receive two.
Until now, only the explicit `inject_runtime` flag decided injection. Most
authors would not know to switch it off.

Two review rounds of a regex scan over the whole body showed that text-matching
the HTML has no end point. Prose, attribute values, JavaScript comments, nested
`<template>`s and raw-text elements each suppressed the runtime by mistake.
Suppressing it by mistake is the worse failure: every `ag2ui` call throws and
the view never initializes.

## Decision

Injection is skipped when a script the browser would run already performs the
handshake: it loads the SDK (`src`, `import`, `from` or `import()` with a
specifier naming the package), or its code has `"ui/initialize"` as a string.
`ag2/mcp/_document_scripts.py` decides this:

- HTML is read with the standard library's `html.parser`. A stack of inert
  ancestors (`template`, `noscript`, `style`, `title`, `textarea`, …) keeps
  nested and raw-text content out, the same way on every supported Python.
- Script code is read by a small JavaScript tokenizer. It drops comments, tells
  strings and regular-expression literals from code, and matches tokens rather
  than text.

When a document that performs its own handshake also calls `ag2ui` without
defining it (`window.ag2ui = …`), `ag2ui` is injected anyway and a warning is
logged. That document is broken either way. With injection, the `ag2ui` calls
work and the host sees two handshakes. Without it, every call throws. The
warning names the document in both cases. Making `ag2ui` usage the stronger
signal also turns any remaining misdetection in a document that uses `ag2ui`
into a working view instead of a dead one.

`inject_runtime=False` keeps its meaning and is the escape hatch for what the
reader does not see:

- a method name built at run time;
- an SDK imported under a specifier that does not name the package, such as an
  import-map alias;
- legacy `<!--` line comments in classic scripts.

## Consequences

- A document mixing `App` and `ag2ui` is served with two handshakes, not none.
- The reader is not a full HTML or JavaScript parser. A new false positive is
  fixed in `_document_scripts.py` with a regression test, not with another
  pattern over the raw body.
- No third-party dependency is added. A browser-accurate parser such as
  `html5lib` was not worth an extra dependency for one decision.
