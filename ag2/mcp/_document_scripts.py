# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""What the scripts of an MCP app document do, as far as serving it needs to know.

Serving decides whether to inject the ``ag2ui`` runtime from what the browser
would run. A ``<script>`` element counts only when its ``type`` is executable and
no ancestor holds it inert, and inside it only code counts, never a comment.
HTML is read with the standard library's parser, which keeps comments, attribute
values and raw-text elements out of script code. Script code is read by a small
tokenizer that skips comments and tells a string from the code around it.
Neither reads the whole language; ADR 0037 records what they miss, which
``inject_runtime=False`` covers.
"""

import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import NamedTuple

__all__ = ("SDK_PACKAGE", "DocumentScripts", "read_scripts")

# The upstream SDK whose ``App`` class performs the handshake itself.
SDK_PACKAGE = "@modelcontextprotocol/ext-apps"

# The handshake's method name as a string in code: the SDK inlined by a bundler,
# or ``ag2ui`` placed by hand.
_HANDSHAKE_METHOD = "ui/initialize"

# Elements whose content the browser never runs as a script: a ``<template>``
# holds an inert fragment, ``<noscript>`` is raw text while scripting is on, and
# the rest are raw or escapable raw text, so a ``<script>`` inside is only text.
# Tracked as a stack because the standard parser reports a ``<script>`` inside
# ``<title>`` or ``<textarea>`` as an element on some Python versions.
_INERT_ELEMENTS = frozenset({
    "template",
    "noscript",
    "textarea",
    "title",
    "style",
    "xmp",
    "iframe",
    "noembed",
    "noframes",
    "plaintext",
})

# The ``type`` values a browser runs a ``<script>`` for: none (or empty),
# ``module``, and the JavaScript MIME types. Anything else is a data block.
_EXECUTABLE_TYPE = re.compile(
    r"(?:module|(?:text|application)/(?:x-)?(?:java|ecma)script|text/jscript|text/livescript)?", re.IGNORECASE
)

# One JavaScript token at a time. Whitespace and comments are matched only to be
# skipped. A ``/`` starts a regular expression literal where an operand is
# expected, which the tokenizer decides from the token before it.
_JS_TOKEN = re.compile(
    r"""(?P<space>\s+)
    |(?P<comment>//[^\n]*|/\*.*?(?:\*/|\Z))
    |(?P<string>"(?:[^"\\\n]|\\.)*"?|'(?:[^'\\\n]|\\.)*'?|`(?:[^`\\]|\\.)*`?)
    |(?P<name>[\w$]+)
    |(?P<punct>.)""",
    re.VERBOSE | re.DOTALL,
)
_JS_REGEX_LITERAL = re.compile(r"/(?:[^/\\\[\n]|\\.|\[(?:[^\]\\\n]|\\.)*\])+/[A-Za-z]*")
_OPERAND_AFTER_PUNCT = frozenset("(,=:[!&|?{};+-*%<>~^")
_OPERAND_AFTER_NAME = frozenset({
    "return",
    "typeof",
    "case",
    "do",
    "else",
    "in",
    "of",
    "new",
    "delete",
    "void",
    "throw",
    "yield",
    "await",
    "instanceof",
})


@dataclass(frozen=True, slots=True)
class DocumentScripts:
    """What the document's running scripts do.

    Attributes:
        performs_handshake: A script loads the upstream SDK or carries the
            ``"ui/initialize"`` method name, so the document introduces itself
            to the host without ``ag2ui``.
        uses_ag2ui: Code calls into ``ag2ui`` (``ag2ui.`` or ``window.ag2ui.``).
        defines_ag2ui: Code assigns ``window.ag2ui``, as the placed runtime does.
    """

    performs_handshake: bool = False
    uses_ag2ui: bool = False
    defines_ag2ui: bool = False

    @property
    def needs_ag2ui(self) -> bool:
        """The document calls ``ag2ui`` and nothing in it defines it."""
        return self.uses_ag2ui and not self.defines_ag2ui


def read_scripts(html: str) -> DocumentScripts:
    """Read what the scripts the browser would run in ``html`` do."""
    collector = _ScriptCollector()
    collector.feed(html)
    collector.close()
    handshake = uses = defines = False
    for script in collector.scripts:
        tokens = list(_js_tokens(script.code))
        handshake = handshake or SDK_PACKAGE in script.src or _performs_handshake(tokens)
        uses = uses or _uses_ag2ui(tokens)
        defines = defines or _defines_ag2ui(tokens)
    return DocumentScripts(performs_handshake=handshake, uses_ag2ui=uses, defines_ag2ui=defines)


@dataclass(slots=True)
class _Script:
    src: str
    runs: bool
    code: list[str] = field(default_factory=list)


class _RunningScript(NamedTuple):
    src: str
    code: str


class _ScriptCollector(HTMLParser):
    """Collects the ``src`` and code of every ``<script>`` the browser would run."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.scripts: list[_RunningScript] = []
        self._inert: list[str] = []
        self._open: _Script | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _INERT_ELEMENTS:
            self._inert.append(tag)
        elif tag == "script":
            values = dict(attrs)
            declared = (values.get("type") or "").strip()
            runs = not self._inert and _EXECUTABLE_TYPE.fullmatch(declared) is not None
            # An inert or data script is still opened, so its code is taken as
            # its own and never as markup around it.
            self._open = _Script(src=values.get("src") or "", runs=runs)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        # HTML ignores the slash of ``<template/>`` or ``<script/>``: neither is
        # self-closing, so the element stays open until its end tag.
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        if tag == "script":
            self._finish_script()
        elif tag in self._inert:
            # Close the innermost open element of this name and whatever is left
            # open inside it, so a nested template ends at its own end tag.
            innermost = len(self._inert) - 1 - self._inert[::-1].index(tag)
            del self._inert[innermost:]

    def handle_data(self, data: str) -> None:
        if self._open is not None:
            self._open.code.append(data)

    def close(self) -> None:
        super().close()
        # A script left open runs to the end of the document. Older parsers keep
        # its code buffered instead of reporting it.
        if self._open is not None:
            self._open.code.append(self.rawdata)
            self.rawdata = ""
            self._finish_script()

    def _finish_script(self) -> None:
        if self._open is not None and self._open.runs:
            self.scripts.append(_RunningScript(self._open.src, "".join(self._open.code)))
        self._open = None


class _Token(NamedTuple):
    kind: str
    text: str

    @property
    def value(self) -> str:
        """A string token's content, without its quotes."""
        closed = len(self.text) > 1 and self.text[-1] == self.text[0]
        return self.text[1:-1] if closed else self.text[1:]


def _js_tokens(code: str) -> Iterator[_Token]:
    """The names, strings and punctuation of ``code``; whitespace and comments dropped."""
    position = 0
    previous: _Token | None = None
    while position < len(code):
        if code[position] == "/" and _operand_expected(previous):
            literal = _JS_REGEX_LITERAL.match(code, position)
            if literal is not None:
                position = literal.end()
                previous = _Token("regex", literal.group())
                continue
        # ``(?P<punct>.)`` matches any character, so a token is always found.
        match = _JS_TOKEN.match(code, position)
        if match is None or match.lastgroup is None:
            return
        position = match.end()
        if match.lastgroup in ("space", "comment"):
            continue
        previous = _Token(match.lastgroup, match.group())
        yield previous


def _operand_expected(previous: _Token | None) -> bool:
    if previous is None:
        return True
    if previous.kind == "punct":
        return previous.text in _OPERAND_AFTER_PUNCT
    return previous.kind == "name" and previous.text in _OPERAND_AFTER_NAME


def _performs_handshake(tokens: Sequence[_Token]) -> bool:
    """A string naming the method, or an ``import`` / ``from`` of a specifier naming the SDK."""
    for i, token in enumerate(tokens):
        if token.kind != "string":
            continue
        if token.value == _HANDSHAKE_METHOD:
            return True
        if SDK_PACKAGE in token.value and _is_import_specifier(tokens, i):
            return True
    return False


def _is_import_specifier(tokens: Sequence[_Token], i: int) -> bool:
    """``from "…"``, ``import "…"`` or ``import("…")``."""
    before = tokens[max(0, i - 2) : i]
    if before and before[-1].kind == "name" and before[-1].text in ("from", "import"):
        return True
    return len(before) == 2 and before[0].text == "import" and before[1].text == "("


def _uses_ag2ui(tokens: Sequence[_Token]) -> bool:
    """``ag2ui.`` on its own or as ``window.ag2ui.``, but not ``other.ag2ui.``."""
    for i, token in enumerate(tokens):
        if token.kind != "name" or token.text != "ag2ui" or _text_at(tokens, i + 1) != ".":
            continue
        if _text_at(tokens, i - 1) != "." or _text_at(tokens, i - 2) == "window":
            return True
    return False


def _defines_ag2ui(tokens: Sequence[_Token]) -> bool:
    """``window.ag2ui = …``, not a comparison."""
    return any(
        token.kind == "name"
        and token.text == "ag2ui"
        and _text_at(tokens, i - 1) == "."
        and _text_at(tokens, i - 2) == "window"
        and _text_at(tokens, i + 1) == "="
        and _text_at(tokens, i + 2) != "="
        for i, token in enumerate(tokens)
    )


def _text_at(tokens: Sequence[_Token], i: int) -> str | None:
    return tokens[i].text if 0 <= i < len(tokens) else None
