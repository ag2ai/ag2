# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
"""Cost Tracker Provider

This module provides a telemetry provider that tracks cost events.

Uses the following spans:
N/A

Uses the following events:
EventKind.COST

Will store a cost history as a list of dictionaries, where each dictionary contains the attributes of the cost event.

The total cost can be retrieved via the total_cost attribute.

Costs captured:
- LLM Costs

Code example:
```python
from autogen.telemetry.telemetry_core import InstrumentationManager, telemetry_context
from autogen.telemetry.providers.cost_tracker import CostTrackerProvider

cost_provider = CostTrackerProvider()

with telemetry_context(instrumentation_manager, "My Program Name", {"my_program_id": "456"}) as telemetry:
    ... AG2 workflow code here ...

    # Print out the cost and how many cost events there were
    print(f"Cost summary: {cost_provider.total_cost:.6f} ({len(cost_provider.cost_history)} cost events)")

```
"""
import json
from typing import Any, Dict, Optional

from ..base_telemetry import EventKind, SpanContext, SpanKind, TelemetryProvider, _is_list_of_string_dicts


class CostTrackerProvider(TelemetryProvider):
    def __init__(self):
        self.cost_history = []  # History of cost events, contains dictionary of attributes
        self.total_cost = 0.0

    def start_trace(self, name: str, core_span_id: str, attributes: Dict[str, Any] = None) -> SpanContext:
        pass

    def start_span(
        self,
        kind: SpanKind,
        core_span_id: str,
        parent_context: Optional[SpanContext] = None,
        attributes: Dict[str, Any] = None,
    ) -> SpanContext:
        pass

    def set_span_attribute(self, context: SpanContext, key: str, value: Any) -> None:
        pass

    def end_span(self, context: SpanContext) -> None:
        pass

    def record_event(self, span_context: SpanContext, event_kind: EventKind, attributes: Dict[str, Any] = None) -> None:
        """Record the cost event, storing the attributes as a cost history and tallying up the total cost.

        See telemetry_core.py for details on this event.
        """
        if event_kind == EventKind.COST:
            cost = attributes.get("ag2.cost")

            # Convert attributes to OpenTelemetry compatible formats
            formatted_attributes = {key: self.convert_attribute_value(value) for key, value in attributes.items()}

            if cost is not None:
                self.cost_history.append(formatted_attributes)
                self.total_cost += cost

    def convert_attribute_value(self, value: Any) -> Any:
        """Convert attributes for storage in a dictionary

        Args:
            value (Any): The value to convert

        Returns:
            Any: The converted value, stored in line with valid OpenTelemetry attributes
        """
        if isinstance(value, (str, int, float, bool)):
            return value
        if _is_list_of_string_dicts(value) or isinstance(value, dict):
            return json.dumps(value)  # Typical messages
        return str(value)
