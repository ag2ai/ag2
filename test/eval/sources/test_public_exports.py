# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""The span → ``Trace`` bridge is public API.

Integrators who drive their own agents (their own sessions, auth, orchestration)
cannot use ``run_agent``, but still want their traces graded by the *same* code
AG2 grades its own runs with. That means reaching the reconstruction directly, so
``readable_spans_to_trace`` / ``spans_to_trace`` are exported rather than reached
through a private module.

Nothing here needs the OpenTelemetry SDK — which is the other half of the
contract: exporting the bridge must not put the SDK on ``ag2.eval``'s import path.
"""

import subprocess
import sys
import textwrap

import ag2.eval
import ag2.eval.sources
from ag2.eval.sources import _otel, _spans


def test_sources_package_exports_the_span_bridge() -> None:
    for name in ("readable_span_to_data", "readable_spans_to_trace", "spans_to_trace"):
        assert name in ag2.eval.sources.__all__

    assert ag2.eval.sources.readable_span_to_data is _otel.readable_span_to_data
    assert ag2.eval.sources.readable_spans_to_trace is _otel.readable_spans_to_trace
    assert ag2.eval.sources.spans_to_trace is _spans.spans_to_trace


def test_eval_package_reexports_the_span_bridge() -> None:
    """``readable_span_to_data`` stays at the sources level — it is a per-span primitive."""
    for name in ("readable_spans_to_trace", "spans_to_trace"):
        assert name in ag2.eval.__all__

    assert ag2.eval.readable_spans_to_trace is _otel.readable_spans_to_trace
    assert ag2.eval.spans_to_trace is _spans.spans_to_trace
    assert "readable_span_to_data" not in ag2.eval.__all__


def test_spans_to_trace_needs_no_sdk() -> None:
    """The SDK-free primitive is importable and callable in this process regardless."""
    assert ag2.eval.spans_to_trace([]).events == ()


def test_importing_ag2_eval_without_the_otel_sdk_still_works() -> None:
    """Exporting the bridge must not drag ``opentelemetry`` onto the eval import path.

    Run in a subprocess with ``opentelemetry`` blocked at the meta path, so the
    check is real rather than a stand-in — and so a poisoned ``sys.modules``
    cannot leak into the rest of the suite.
    """
    script = textwrap.dedent("""
        import sys

        class Blocker:
            def find_spec(self, name, path=None, target=None):
                if name == "opentelemetry" or name.startswith("opentelemetry."):
                    raise ImportError(f"blocked: {name}")
                return None

        for mod in [m for m in sys.modules if m == "opentelemetry" or m.startswith("opentelemetry.")]:
            del sys.modules[mod]
        sys.meta_path.insert(0, Blocker())

        import ag2.eval

        assert not any(m.startswith("opentelemetry") for m in sys.modules), "SDK leaked onto the import path"
        assert ag2.eval.spans_to_trace([]).events == ()

        try:
            ag2.eval.readable_spans_to_trace([])
        except ImportError as e:
            print(str(e).replace("\\n", " "))
        else:
            print("NO ERROR RAISED")
    """)
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)

    # Calling it without the SDK fails with the install hint, not an obscure NameError.
    assert "opentelemetry.sdk' is not installed" in result.stdout
    assert "pip install ag2[tracing]" in result.stdout
