# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import copy

import pytest

from autogen.agentchat.group.context_variables import ContextVariables


class TestContextVariables:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.context = ContextVariables()
        self.test_data = {"key1": "value1", "key2": 2, "key3": [1, 2, 3], "key4": {"nested": "value"}}
        self.context.update(self.test_data)

    def test_init(self) -> None:
        # Test empty initialisation
        empty_context = ContextVariables()
        assert len(empty_context) == 0
        assert empty_context.data == {}

        # Test initialisation with data
        context_with_data = ContextVariables(data=self.test_data)
        assert len(context_with_data) == 4
        assert context_with_data.data == self.test_data

    def test_get(self) -> None:
        # Test getting existing keys
        assert self.context.get("key1") == "value1"
        assert self.context.get("key2") == 2
        assert self.context.get("key3") == [1, 2, 3]
        assert self.context.get("key4") == {"nested": "value"}

        # Test getting non-existent key and with a default
        assert self.context.get("nonexistent") is None
        assert self.context.get("nonexistent", "default_value") == "default_value"

    def test_set(self) -> None:
        # Test setting new key
        self.context.set("new_key", "new_value")
        assert self.context.get("new_key") == "new_value"

        # Test overwriting existing key
        self.context.set("key1", "updated_value")
        assert self.context.get("key1") == "updated_value"

        # Test setting complex value
        complex_value = {"nested": {"deeper": [1, 2, 3]}}
        self.context.set("complex", complex_value)
        assert self.context.get("complex") == complex_value

    def test_remove(self) -> None:
        # Test removing existing key
        assert self.context.remove("key1") is True
        assert "key1" not in self.context

        # Test removing non-existent key
        assert self.context.remove("nonexistent") is False

    def test_keys_values_items(self) -> None:
        # Test keys
        assert set(self.context.keys()) == set(self.test_data.keys())

        # Test values
        context_values = list(self.context.values())
        test_data_values = list(self.test_data.values())
        # Need to verify all values exist without relying on order
        for value in context_values:
            if isinstance(value, list):
                # For lists, check if there's an equal list
                assert any(value == test_value for test_value in test_data_values if isinstance(test_value, list))
            elif isinstance(value, dict):
                # For dicts, check if there's an equal dict
                assert any(value == test_value for test_value in test_data_values if isinstance(test_value, dict))
            else:
                # For hashable values
                assert value in test_data_values

        # Test items - can't use set due to unhashable values in items
        context_items = list(self.context.items())
        test_data_items = list(self.test_data.items())
        assert len(context_items) == len(test_data_items)
        for key, value in context_items:
            assert key in self.test_data
            assert self.test_data[key] == value

    def test_clear(self) -> None:
        self.context.clear()
        assert len(self.context) == 0
        assert self.context.data == {}

    def test_contains(self) -> None:
        assert self.context.contains("key1") is True
        assert self.context.contains("nonexistent") is False

    def test_update(self) -> None:
        update_data = {"key1": "updated", "new_key": "new_value"}
        self.context.update(update_data)

        assert self.context.get("key1") == "updated"
        assert self.context.get("new_key") == "new_value"
        assert self.context.get("key2") == 2  # Original data should remain

    def test_to_dict(self) -> None:
        result = self.context.to_dict()

        # Ensure it's a copy, not the original
        assert result is not self.context.data

        # Test content is the same
        assert result == self.test_data

    def test_dunder_getitem(self) -> None:
        assert self.context["key1"] == "value1"

        with pytest.raises(KeyError) as e:
            _ = self.context["nonexistent"]
        assert "Context variable 'nonexistent' not found" in str(e.value)

    def test_dunder_setitem(self) -> None:
        self.context["key1"] = "updated_via_dunder"
        assert self.context.get("key1") == "updated_via_dunder"

        self.context["new_key"] = "new_value"
        assert self.context.get("new_key") == "new_value"

    def test_dunder_delitem(self) -> None:
        del self.context["key1"]
        assert "key1" not in self.context

        with pytest.raises(KeyError) as e:
            del self.context["nonexistent"]
        assert "Cannot delete non-existent context variable 'nonexistent'" in str(e.value)

    def test_dunder_contains(self) -> None:
        assert "key1" in self.context
        assert "nonexistent" not in self.context

    def test_dunder_len(self) -> None:
        assert len(self.context) == 4

        self.context.set("new_key", "value")
        assert len(self.context) == 5

        self.context.remove("key1")
        assert len(self.context) == 4

    def test_dunder_iter(self) -> None:
        keys = set()
        for key in self.context:
            keys.add(key)

        assert keys == set(self.test_data.keys())

    def test_dunder_str_repr(self) -> None:
        str_result = str(self.context)
        assert str_result.startswith("ContextVariables(")

        repr_result = repr(self.context)
        assert repr_result.startswith("ContextVariables(data=")

    def test_from_dict(self) -> None:
        custom_data = {"a": 1, "b": "test", "c": [1, 2, 3]}
        context = ContextVariables.from_dict(custom_data)

        assert isinstance(context, ContextVariables)
        assert context.data == custom_data

    def test_nested_data_operations(self) -> None:
        # Test operations with nested data structures
        self.context.set("nested", {"level1": {"level2": "value"}})

        # Get and modify the nested structure
        nested = self.context.get("nested")
        nested["level1"]["level2"] = "modified"
        self.context.set("nested", nested)

        assert self.context.get("nested")["level1"]["level2"] == "modified"

    def test_copy_semantics(self) -> None:
        # Test that to_dict returns a copy
        data_copy = self.context.to_dict()
        data_copy["key1"] = "modified_in_copy"
        assert self.context.get("key1") == "value1"  # Original unchanged

        # Test shallow copy behavior with nested structures
        self.context.set("nested", {"level1": {"level2": "value"}})
        data_copy = self.context.to_dict()
        data_copy["nested"]["level1"]["level2"] = "modified_in_copy"

        # The modification affects the original (shallow copy)
        assert self.context.get("nested")["level1"]["level2"] == "modified_in_copy"

        # Test with deep copy
        deep_copy = copy.deepcopy(self.context.to_dict())
        deep_copy["nested"]["level1"]["level2"] = "modified_in_deep_copy"

        # Original should be unchanged
        assert self.context.get("nested")["level1"]["level2"] == "modified_in_copy"
