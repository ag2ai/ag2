# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""The ``format`` checks the vendored A2UI schemas use. Not exported.

Upstream's conformance runner validates with ``ajv-formats`` (full mode), so ``date``, ``time``, ``date-time``
and ``uri`` are asserted, not just annotated: ``function_catalog_validation.json`` expects
``openUrl({"url": "not a uri"})`` to be invalid. These are ports of ``ajv-formats`` 3.0.1's full-mode checks.
``jsonschema``'s own checkers for these formats need optional packages, and without them pass everything.
Every pattern is ASCII-only, as JavaScript's are: a digit is an ASCII digit, not any Unicode one.
"""

import re
from collections.abc import Callable, Mapping

_DATE = re.compile(r"(\d\d\d\d)-(\d\d)-(\d\d)", re.ASCII)
_DAYS = (0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
_TIME = re.compile(r"(\d\d):(\d\d):(\d\d(?:\.\d+)?)(z|([+-])(\d\d)(?::?(\d\d))?)?", re.IGNORECASE | re.ASCII)
_DATE_TIME_SEPARATOR = re.compile(r"t|\s", re.IGNORECASE | re.ASCII)
_NOT_URI_FRAGMENT = re.compile(r"/|:")
_H16 = r"[0-9a-f]{1,4}"
_IPV4 = r"(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)"
_IPV6 = (
    rf"(?:(?:(?:{_H16}:){{6}}|::(?:{_H16}:){{5}}|(?:{_H16})?::(?:{_H16}:){{4}}"
    rf"|(?:(?:{_H16}:){{0,1}}{_H16})?::(?:{_H16}:){{3}}|(?:(?:{_H16}:){{0,2}}{_H16})?::(?:{_H16}:){{2}}"
    rf"|(?:(?:{_H16}:){{0,3}}{_H16})?::{_H16}:|(?:(?:{_H16}:){{0,4}}{_H16})?::)(?:{_H16}:{_H16}|{_IPV4})"
    rf"|(?:(?:{_H16}:){{0,5}}{_H16})?::{_H16}|(?:(?:{_H16}:){{0,6}}{_H16})?::)"
)
_PCHAR = r"(?:[a-z0-9\-._~!$&'()*+,;=:@]|%[0-9a-f]{2})"
_URI = re.compile(
    r"(?:[a-z][a-z0-9+\-.]*:)"
    r"(?:\/?\/(?:(?:[a-z0-9\-._~!$&'()*+,;=:]|%[0-9a-f]{2})*@)?"
    rf"(?:\[(?:{_IPV6}|[Vv][0-9a-f]+\.[a-z0-9\-._~!$&'()*+,;=:]+)\]|{_IPV4}"
    r"|(?:[a-z0-9\-._~!$&'()*+,;=]|%[0-9a-f]{2})*)"
    rf"(?::\d*)?(?:\/{_PCHAR}*)*"
    rf"|\/(?:{_PCHAR}+(?:\/{_PCHAR}*)*)?"
    rf"|{_PCHAR}+(?:\/{_PCHAR}*)*)"
    r"(?:\?(?:[a-z0-9\-._~!$&'()*+,;=:@/?]|%[0-9a-f]{2})*)?"
    r"(?:#(?:[a-z0-9\-._~!$&'()*+,;=:@/?]|%[0-9a-f]{2})*)?",
    re.IGNORECASE | re.ASCII,
)


def _is_leap_year(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def is_date(value: str) -> bool:
    """RFC 3339 ``full-date``."""
    match = _DATE.fullmatch(value)
    if match is None:
        return False
    year, month, day = (int(group) for group in match.groups())
    return 1 <= month <= 12 and 1 <= day <= (29 if month == 2 and _is_leap_year(year) else _DAYS[month])


def is_time(value: str) -> bool:
    """RFC 3339 ``full-time``: the time zone is required, and a leap second only at 23:59 UTC."""
    match = _TIME.fullmatch(value)
    if match is None or match.group(4) is None:
        return False
    hour, minute, second = int(match.group(1)), int(match.group(2)), float(match.group(3))
    sign = -1 if match.group(5) == "-" else 1
    zone_hour, zone_minute = int(match.group(6) or 0), int(match.group(7) or 0)
    if zone_hour > 23 or zone_minute > 59:
        return False
    if hour <= 23 and minute <= 59 and second < 60:
        return True
    utc_minute = minute - zone_minute * sign
    utc_hour = hour - zone_hour * sign - (1 if utc_minute < 0 else 0)
    return utc_hour in (23, -1) and utc_minute in (59, -1) and second < 61


def is_date_time(value: str) -> bool:
    """RFC 3339 ``date-time``: a date and a time separated by ``T`` or whitespace."""
    parts = _DATE_TIME_SEPARATOR.split(value)
    return len(parts) == 2 and is_date(parts[0]) and is_time(parts[1])


def is_uri(value: str) -> bool:
    """An absolute RFC 3986 URI."""
    return _NOT_URI_FRAGMENT.search(value) is not None and _URI.fullmatch(value) is not None


SCHEMA_FORMATS: Mapping[str, Callable[[str], bool]] = {
    "date": is_date,
    "time": is_time,
    "date-time": is_date_time,
    "uri": is_uri,
}
"""Each standard ``format`` the vendored schemas use, with its check."""


__all__ = ("SCHEMA_FORMATS", "is_date", "is_date_time", "is_time", "is_uri")
