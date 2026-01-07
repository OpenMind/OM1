"""Tests for the diff module."""

import pytest
from runtime.hot_reload.diff import (
    ConfigDiff,
    FieldChange,
    diff_configs,
    get_nested_value,
    set_nested_value,
)


class TestFieldChange:
    """Tests for FieldChange dataclass."""

    def test_field_change_creation(self):
        """Test creating a FieldChange instance."""
        change = FieldChange(
            field_path="system_prompt_base",
            change_type="modified",
            old_value="old prompt",
            new_value="new prompt",
        )
        assert change.field_path == "system_prompt_base"
        assert change.change_type == "modified"
        assert change.old_value == "old prompt"
        assert change.new_value == "new prompt"

    def test_field_change_added(self):
        """Test FieldChange for added field."""
        change = FieldChange(
            field_path="new_field",
            change_type="added",
            old_value=None,
            new_value="value",
        )
        assert change.change_type == "added"
        assert change.old_value is None

    def test_field_change_removed(self):
        """Test FieldChange for removed field."""
        change = FieldChange(
            field_path="old_field",
            change_type="removed",
            old_value="value",
            new_value=None,
        )
        assert change.change_type == "removed"
        assert change.new_value is None


class TestConfigDiff:
    """Tests for ConfigDiff dataclass."""

    def test_empty_diff(self):
        """Test empty ConfigDiff."""
        diff = ConfigDiff(changes=[])
        assert diff.has_changes is False
        assert len(diff.changes) == 0
        assert diff.changed_fields == {}

    def test_diff_with_changes(self):
        """Test ConfigDiff with changes."""
        changes = [
            FieldChange("field1", "modified", "old", "new"),
            FieldChange("field2", "added", None, "value"),
        ]
        diff = ConfigDiff(changes=changes)
        assert diff.has_changes is True
        assert len(diff.changes) == 2

    def test_changed_fields_property(self):
        """Test changed_fields property returns dict mapping."""
        changes = [
            FieldChange("field1", "modified", "old1", "new1"),
            FieldChange("field2", "modified", "old2", "new2"),
        ]
        diff = ConfigDiff(changes=changes)
        changed = diff.changed_fields

        assert "field1" in changed
        assert changed["field1"] == ("old1", "new1")
        assert "field2" in changed
        assert changed["field2"] == ("old2", "new2")


class TestGetNestedValue:
    """Tests for get_nested_value function."""

    def test_simple_key(self):
        """Test getting a simple top-level key."""
        data = {"name": "test", "value": 42}
        assert get_nested_value(data, "name") == "test"
        assert get_nested_value(data, "value") == 42

    def test_nested_key(self):
        """Test getting a nested key with dot notation."""
        data = {"config": {"llm": {"model": "gpt-4"}}}
        assert get_nested_value(data, "config.llm.model") == "gpt-4"

    def test_missing_key_returns_default(self):
        """Test that missing keys return the default value."""
        data = {"name": "test"}
        assert get_nested_value(data, "missing") is None
        assert get_nested_value(data, "missing", "default") == "default"

    def test_nested_missing_key(self):
        """Test missing nested key returns default."""
        data = {"config": {"llm": {}}}
        assert get_nested_value(data, "config.llm.model") is None

    def test_list_index_access(self):
        """Test accessing list elements by index."""
        data = {"items": ["a", "b", "c"]}
        assert get_nested_value(data, "items.0") == "a"
        assert get_nested_value(data, "items.1") == "b"
        assert get_nested_value(data, "items.2") == "c"

    def test_nested_list_access(self):
        """Test accessing nested structures in lists."""
        data = {"agents": [{"name": "agent1"}, {"name": "agent2"}]}
        assert get_nested_value(data, "agents.0.name") == "agent1"
        assert get_nested_value(data, "agents.1.name") == "agent2"


class TestSetNestedValue:
    """Tests for set_nested_value function."""

    def test_simple_key(self):
        """Test setting a simple top-level key."""
        data = {"name": "old"}
        set_nested_value(data, "name", "new")
        assert data["name"] == "new"

    def test_nested_key(self):
        """Test setting a nested key."""
        data = {"config": {"llm": {"model": "old"}}}
        set_nested_value(data, "config.llm.model", "new")
        assert data["config"]["llm"]["model"] == "new"

    def test_create_nested_structure(self):
        """Test that nested structures are created if missing."""
        data = {}
        set_nested_value(data, "config.llm.model", "gpt-4")
        assert data["config"]["llm"]["model"] == "gpt-4"

    def test_list_index_set(self):
        """Test setting list elements by index."""
        data = {"items": ["a", "b", "c"]}
        set_nested_value(data, "items.1", "B")
        assert data["items"][1] == "B"


class TestDiffConfigs:
    """Tests for diff_configs function."""

    def test_no_changes(self):
        """Test diffing identical configs."""
        old = {"name": "test", "hertz": 1}
        new = {"name": "test", "hertz": 1}
        diff = diff_configs(old, new)
        assert diff.has_changes is False

    def test_simple_field_change(self):
        """Test detecting simple field change."""
        old = {"name": "old_name"}
        new = {"name": "new_name"}
        diff = diff_configs(old, new)

        assert diff.has_changes is True
        assert len(diff.changes) == 1
        assert diff.changes[0].field_path == "name"
        assert diff.changes[0].old_value == "old_name"
        assert diff.changes[0].new_value == "new_name"

    def test_field_added(self):
        """Test detecting added field."""
        old = {"name": "test"}
        new = {"name": "test", "hertz": 2}
        diff = diff_configs(old, new)

        assert diff.has_changes is True
        added = [c for c in diff.changes if c.change_type == "added"]
        assert len(added) == 1
        assert added[0].field_path == "hertz"
        assert added[0].new_value == 2

    def test_field_removed(self):
        """Test detecting removed field."""
        old = {"name": "test", "hertz": 2}
        new = {"name": "test"}
        diff = diff_configs(old, new)

        assert diff.has_changes is True
        removed = [c for c in diff.changes if c.change_type == "removed"]
        assert len(removed) == 1
        assert removed[0].field_path == "hertz"
        assert removed[0].old_value == 2

    def test_nested_field_change(self):
        """Test detecting nested field change."""
        old = {"cortex_llm": {"config": {"model": "gpt-3.5"}}}
        new = {"cortex_llm": {"config": {"model": "gpt-4"}}}
        diff = diff_configs(old, new)

        assert diff.has_changes is True
        assert any(c.field_path == "cortex_llm.config.model" for c in diff.changes)

    def test_list_element_change(self):
        """Test detecting list element changes."""
        old = {"agent_inputs": [{"type": "A"}, {"type": "B"}]}
        new = {"agent_inputs": [{"type": "A"}, {"type": "C"}]}
        diff = diff_configs(old, new)

        assert diff.has_changes is True
        assert any("agent_inputs" in c.field_path for c in diff.changes)

    def test_list_length_change(self):
        """Test detecting list length changes."""
        old = {"items": [1, 2]}
        new = {"items": [1, 2, 3]}
        diff = diff_configs(old, new)

        assert diff.has_changes is True

    def test_multiple_changes(self):
        """Test detecting multiple simultaneous changes."""
        old = {
            "name": "old_name",
            "hertz": 1,
            "system_prompt_base": "old prompt",
        }
        new = {
            "name": "new_name",
            "hertz": 2,
            "system_prompt_base": "new prompt",
        }
        diff = diff_configs(old, new)

        assert diff.has_changes is True
        assert len(diff.changes) == 3
        changed_fields = {c.field_path for c in diff.changes}
        assert "name" in changed_fields
        assert "hertz" in changed_fields
        assert "system_prompt_base" in changed_fields

    def test_none_handling(self):
        """Test handling None values."""
        old = {"field": None}
        new = {"field": "value"}
        diff = diff_configs(old, new)

        assert diff.has_changes is True
        assert diff.changes[0].old_value is None
        assert diff.changes[0].new_value == "value"

    def test_type_change(self):
        """Test detecting type changes."""
        old = {"field": "string"}
        new = {"field": 123}
        diff = diff_configs(old, new)

        assert diff.has_changes is True
        assert diff.changes[0].old_value == "string"
        assert diff.changes[0].new_value == 123

    def test_empty_configs(self):
        """Test diffing empty configs."""
        diff = diff_configs({}, {})
        assert diff.has_changes is False

    def test_old_none(self):
        """Test when old config is None."""
        diff = diff_configs(None, {"field": "value"})
        assert diff.has_changes is True

    def test_new_none(self):
        """Test when new config is None."""
        diff = diff_configs({"field": "value"}, None)
        assert diff.has_changes is True

    def test_deep_nested_change(self):
        """Test deeply nested changes are detected."""
        old = {"a": {"b": {"c": {"d": {"e": "old"}}}}}
        new = {"a": {"b": {"c": {"d": {"e": "new"}}}}}
        diff = diff_configs(old, new)

        assert diff.has_changes is True
        assert diff.changes[0].field_path == "a.b.c.d.e"

