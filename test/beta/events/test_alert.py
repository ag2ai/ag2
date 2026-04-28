# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0


from autogen.beta.events import HaltEvent, ObserverAlert, Severity


class TestObserverAlertCreation:
    def test_create_alert(self) -> None:
        s = ObserverAlert(source="monitor", severity=Severity.WARNING, message="High load")
        assert s.source == "monitor"
        assert s.severity == Severity.WARNING
        assert s.message == "High load"
        assert s.data == {}

    def test_create_alert_with_data(self) -> None:
        s = ObserverAlert(source="mon", severity="custom", message="msg", data={"key": "val"})
        assert s.data == {"key": "val"}
        assert s.severity == "custom"  # Accepts any string

    def test_severity_values(self) -> None:
        assert Severity.INFO == "info"
        assert Severity.WARNING == "warning"
        assert Severity.CRITICAL == "critical"
        assert Severity.FATAL == "fatal"


class TestHaltEvent:
    def test_halt_event_creation(self) -> None:
        h = HaltEvent(source="guard", reason="FATAL: budget exceeded")
        assert h.source == "guard"
        assert "FATAL" in h.reason


# NOTE: InjectToPrompt, EmitToStream, CallHandler, HaltOnFatal are REMOVED.
# Their functionality is now covered by AlertPolicy (from autogen.beta.policies).
