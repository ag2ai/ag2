---
status: accepted
date: 2026-10-11
---

# 0024. Toolkit composition preserves member tools

## Context

A toolkit can attach middleware for its own domain, such as argument changes or
approval hooks. Constructing `left | right` with the left toolkit's middleware
reapplied those hooks to existing members and extended them to unrelated tools
from the right. Combining collections should preserve the tools' behavior.

## Decision

`Toolkit.__or__` creates a new member dictionary containing the already configured
tool objects. It keeps their identities and middleware without copying or
rewrapping them. Plain functions are converted to tools, and an incoming tool
replaces an existing member with the same name.

The combined toolkit has no middleware of its own. Its later decorator additions
receive only explicitly supplied per-tool middleware. Middleware attachment
belongs to toolkit construction and `.tool()`, rather than composition.

## Consequences

Source collections remain independent while the configured member objects are
shared. Code that relied on `|` applying the left toolkit's hooks to incoming or
later-added tools must attach those hooks explicitly. A common policy can be
applied by constructing a toolkit with the desired middleware.
