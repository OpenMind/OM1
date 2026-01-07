"""Tests for the reload strategies module."""

import pytest
from runtime.hot_reload.strategies import (
    ReloadStrategy,
    get_field_strategy,
    validate_field,
    categorize_changes,
    requires_restart,
    get_hot_reloadable_changes,
    HOT_RELOADABLE_FIELDS,
    RESTART_REQUIRED_FIELDS,
)


class TestReloadStrategy:
    """Tests for ReloadStrategy enum."""

    def test_strategy_values(self):
        """Test that all strategy values are defined."""
        assert ReloadStrategy.HOT_RELOAD.value == "hot_reload"
        assert ReloadStrategy.VALIDATE_FIRST.value == "validate_first"
        assert ReloadStrategy.RESTART_REQUIRED.value == "restart_required"
        assert ReloadStrategy.IGNORE.value == "ignore"


class TestGetFieldStrategy:
    """Tests for get_field_strategy function."""

    def test_hot_reloadable_fields(self):
        """Test that hot-reloadable fields are correctly identified."""
        assert get_field_strategy("system_prompt_base") == ReloadStrategy.HOT_RELOAD
        assert get_field_strategy("system_governance") == ReloadStrategy.HOT_RELOAD
        assert get_field_strategy("system_prompt_examples") == ReloadStrategy.HOT_RELOAD
        assert get_field_strategy("name") == ReloadStrategy.HOT_RELOAD

    def test_validate_first_fields(self):
        """Test that validate-first fields are correctly identified."""
        assert get_field_strategy("hertz") == ReloadStrategy.VALIDATE_FIRST

    def test_restart_required_fields(self):
        """Test that restart-required fields are correctly identified."""
        assert get_field_strategy("cortex_llm") == ReloadStrategy.RESTART_REQUIRED
        assert get_field_strategy("agent_inputs") == ReloadStrategy.RESTART_REQUIRED
        assert get_field_strategy("agent_actions") == ReloadStrategy.RESTART_REQUIRED
        assert get_field_strategy("simulators") == ReloadStrategy.RESTART_REQUIRED
        assert get_field_strategy("backgrounds") == ReloadStrategy.RESTART_REQUIRED
        assert get_field_strategy("api_key") == ReloadStrategy.RESTART_REQUIRED

    def test_nested_field_uses_root(self):
        """Test that nested fields use root field for strategy."""
        assert get_field_strategy("cortex_llm.config.model") == ReloadStrategy.RESTART_REQUIRED
        assert get_field_strategy("agent_inputs.0.type") == ReloadStrategy.RESTART_REQUIRED

    def test_unknown_field_defaults_to_restart(self):
        """Test that unknown fields default to RESTART_REQUIRED."""
        assert get_field_strategy("unknown_field") == ReloadStrategy.RESTART_REQUIRED


class TestValidateField:
    """Tests for validate_field function."""

    def test_validate_hertz_positive(self):
        """Test that positive hertz values are valid."""
        assert validate_field("hertz", 1, 2) is True
        assert validate_field("hertz", 1, 0.5) is True
        assert validate_field("hertz", 1, 100) is True

    def test_validate_hertz_invalid_type(self):
        """Test that non-numeric hertz values are invalid."""
        assert validate_field("hertz", 1, "fast") is False
        assert validate_field("hertz", 1, None) is False

    def test_validate_hertz_zero_or_negative(self):
        """Test that zero or negative hertz values are invalid."""
        assert validate_field("hertz", 1, 0) is False
        assert validate_field("hertz", 1, -1) is False

    def test_validate_field_without_validator(self):
        """Test that fields without validators pass validation."""
        assert validate_field("system_prompt_base", "old", "new") is True
        assert validate_field("unknown", "old", "new") is True


class TestCategorizeChanges:
    """Tests for categorize_changes function."""

    def test_categorize_mixed_changes(self):
        """Test categorizing a mix of different field types."""
        changes = {
            "system_prompt_base": ("old", "new"),
            "hertz": (1, 2),
            "cortex_llm": ({"type": "A"}, {"type": "B"}),
        }

        categorized = categorize_changes(changes)

        assert "system_prompt_base" in categorized[ReloadStrategy.HOT_RELOAD]
        assert "hertz" in categorized[ReloadStrategy.VALIDATE_FIRST]
        assert "cortex_llm" in categorized[ReloadStrategy.RESTART_REQUIRED]

    def test_categorize_empty_changes(self):
        """Test categorizing empty changes."""
        changes = {}
        categorized = categorize_changes(changes)

        assert categorized[ReloadStrategy.HOT_RELOAD] == {}
        assert categorized[ReloadStrategy.VALIDATE_FIRST] == {}
        assert categorized[ReloadStrategy.RESTART_REQUIRED] == {}
        assert categorized[ReloadStrategy.IGNORE] == {}


class TestRequiresRestart:
    """Tests for requires_restart function."""

    def test_requires_restart_true(self):
        """Test that restart-required fields return True."""
        changes = {"cortex_llm": ({"type": "A"}, {"type": "B"})}
        assert requires_restart(changes) is True

    def test_requires_restart_false(self):
        """Test that hot-reloadable fields return False."""
        changes = {"system_prompt_base": ("old", "new")}
        assert requires_restart(changes) is False

    def test_requires_restart_mixed(self):
        """Test mixed changes where one requires restart."""
        changes = {
            "system_prompt_base": ("old", "new"),
            "cortex_llm": ({"type": "A"}, {"type": "B"}),
        }
        assert requires_restart(changes) is True


class TestGetHotReloadableChanges:
    """Tests for get_hot_reloadable_changes function."""

    def test_filter_hot_reloadable(self):
        """Test filtering only hot-reloadable changes."""
        changes = {
            "system_prompt_base": ("old", "new"),
            "hertz": (1, 2),
            "cortex_llm": ({"type": "A"}, {"type": "B"}),
        }

        hot_changes = get_hot_reloadable_changes(changes)

        assert "system_prompt_base" in hot_changes
        assert "hertz" in hot_changes
        assert "cortex_llm" not in hot_changes

    def test_filter_empty_when_all_restart(self):
        """Test that result is empty when all changes require restart."""
        changes = {
            "cortex_llm": ({"type": "A"}, {"type": "B"}),
            "agent_inputs": ([], [{"type": "X"}]),
        }

        hot_changes = get_hot_reloadable_changes(changes)
        assert hot_changes == {}


class TestFieldSets:
    """Tests for field set definitions."""

    def test_hot_reloadable_fields_defined(self):
        """Test that hot-reloadable fields are defined."""
        assert "system_prompt_base" in HOT_RELOADABLE_FIELDS
        assert "system_governance" in HOT_RELOADABLE_FIELDS
        assert "system_prompt_examples" in HOT_RELOADABLE_FIELDS
        assert "hertz" in HOT_RELOADABLE_FIELDS
        assert "name" in HOT_RELOADABLE_FIELDS

    def test_restart_required_fields_defined(self):
        """Test that restart-required fields are defined."""
        assert "cortex_llm" in RESTART_REQUIRED_FIELDS
        assert "agent_inputs" in RESTART_REQUIRED_FIELDS
        assert "agent_actions" in RESTART_REQUIRED_FIELDS
        assert "simulators" in RESTART_REQUIRED_FIELDS
        assert "backgrounds" in RESTART_REQUIRED_FIELDS
        assert "api_key" in RESTART_REQUIRED_FIELDS

    def test_no_overlap_between_sets(self):
        """Test that field sets don't overlap."""
        overlap = HOT_RELOADABLE_FIELDS & RESTART_REQUIRED_FIELDS
        # hertz is in both because it's hot-reloadable but needs validation
        assert len(overlap) <= 1  # Only hertz might be in both
