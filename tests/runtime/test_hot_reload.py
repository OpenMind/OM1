"""
Unit tests for HotReloadManager.

Tests selective hot-reload functionality including detection, validation,
categorization, and application of configuration changes.
"""

import pytest

from runtime.single_mode.hot_reload import (
    ConfigChange,
    HotReloadManager,
    ReloadStrategy,
)


@pytest.fixture
def manager():
    """Create a fresh HotReloadManager for each test."""
    return HotReloadManager()


@pytest.fixture
def sample_old_config():
    """Sample old configuration."""
    return {
        "version": "1.0.0",
        "name": "TestBot",
        "hertz": 10,
        "system_prompt_base": "You are a helpful assistant.",
        "cortex_llm": {
            "type": "openai",
            "config": {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 1000,
            },
        },
    }


@pytest.fixture
def sample_new_config():
    """Sample new configuration with changes."""
    return {
        "version": "1.0.0",
        "name": "TestBot",
        "hertz": 10,
        "system_prompt_base": "You are an expert assistant.",  # Changed
        "cortex_llm": {
            "type": "openai",
            "config": {
                "temperature": 0.8,  # Changed
                "top_p": 0.9,
                "max_tokens": 1000,
            },
        },
    }


class TestInit:
    """Tests for initialization."""

    def test_creates_empty_structures(self, manager):
        """Test that initialization creates empty data structures."""
        assert isinstance(manager._field_configs, dict)
        assert isinstance(manager._change_history, list)
        assert len(manager._change_history) == 0

    def test_registers_default_fields(self, manager):
        """Test that default fields are registered."""
        assert "system_prompt_base" in manager._field_configs
        assert "cortex_llm.config.temperature" in manager._field_configs
        assert "hertz" in manager._field_configs


class TestDetectChanges:
    """Tests for detect_changes method."""

    def test_no_changes(self, manager, sample_old_config):
        """Test detection when configs are identical."""
        changes = manager.detect_changes(sample_old_config, sample_old_config)
        assert len(changes) == 0

    def test_detects_changes(self, manager, sample_old_config, sample_new_config):
        """Test detection of changes."""
        changes = manager.detect_changes(sample_old_config, sample_new_config)
        assert len(changes) == 2  # system_prompt_base and temperature

    def test_detects_nested_changes(
        self, manager, sample_old_config, sample_new_config
    ):
        """Test detection of nested field changes."""
        changes = manager.detect_changes(sample_old_config, sample_new_config)
        temp_changes = [
            c for c in changes if c.field_path == "cortex_llm.config.temperature"
        ]
        assert len(temp_changes) == 1
        assert temp_changes[0].old_value == 0.7
        assert temp_changes[0].new_value == 0.8


class TestValidateChanges:
    """Tests for validate_changes method."""

    def test_validates_valid_temperature(self, manager):
        """Test validation of valid temperature."""
        change = ConfigChange(
            "cortex_llm.config.temperature", 0.7, 0.8, ReloadStrategy.VALIDATE_FIRST
        )
        results = manager.validate_changes([change])
        assert results["cortex_llm.config.temperature"] is True

    def test_rejects_invalid_temperature(self, manager):
        """Test validation rejects invalid temperature."""
        change = ConfigChange(
            "cortex_llm.config.temperature", 0.7, 3.0, ReloadStrategy.VALIDATE_FIRST
        )
        results = manager.validate_changes([change])
        assert results["cortex_llm.config.temperature"] is False

    def test_accepts_field_without_validator(self, manager):
        """Test validation passes for fields without validator."""
        change = ConfigChange(
            "system_prompt_base", "old", "new", ReloadStrategy.HOT_RELOAD
        )
        results = manager.validate_changes([change])
        assert results["system_prompt_base"] is True


class TestCategorizeChanges:
    """Tests for categorize_changes method."""

    def test_categorizes_by_strategy(self, manager):
        """Test categorization by strategy."""
        changes = [
            ConfigChange("system_prompt_base", "old", "new", ReloadStrategy.HOT_RELOAD),
            ConfigChange(
                "cortex_llm.config.temperature", 0.7, 0.8, ReloadStrategy.VALIDATE_FIRST
            ),
            ConfigChange("hertz", 10, 20, ReloadStrategy.RESTART_REQUIRED),
        ]
        categorized = manager.categorize_changes(changes)
        assert len(categorized[ReloadStrategy.HOT_RELOAD]) == 1
        assert len(categorized[ReloadStrategy.VALIDATE_FIRST]) == 1
        assert len(categorized[ReloadStrategy.RESTART_REQUIRED]) == 1


class TestApplyChanges:
    """Tests for apply_changes method."""

    def test_applies_hot_reload_changes(self, manager):
        """Test applying hot-reloadable changes."""
        config = {"system_prompt_base": "old"}
        change = ConfigChange(
            "system_prompt_base", "old", "new", ReloadStrategy.HOT_RELOAD
        )
        results = manager.apply_changes(config, [change])
        assert results["system_prompt_base"] is True
        assert config["system_prompt_base"] == "new"

    def test_applies_nested_changes(self, manager):
        """Test applying nested field changes."""
        config = {"cortex_llm": {"config": {"temperature": 0.7}}}
        change = ConfigChange(
            "cortex_llm.config.temperature", 0.7, 0.9, ReloadStrategy.VALIDATE_FIRST
        )
        results = manager.apply_changes(config, [change])
        assert results["cortex_llm.config.temperature"] is True
        assert config["cortex_llm"]["config"]["temperature"] == 0.9

    def test_skips_restart_required(self, manager):
        """Test that restart-required changes are not applied."""
        config = {"hertz": 10}
        change = ConfigChange("hertz", 10, 20, ReloadStrategy.RESTART_REQUIRED)
        results = manager.apply_changes(config, [change])
        assert results["hertz"] is False
        assert config["hertz"] == 10  # Should not change


class TestNestedOperations:
    """Tests for nested value operations."""

    def test_get_nested_value(self, manager):
        """Test getting nested value."""
        config = {"a": {"b": {"c": "value"}}}
        result = manager._get_nested_value(config, "a.b.c")
        assert result == "value"

    def test_get_nested_value_missing(self, manager):
        """Test getting non-existent nested value."""
        config = {"a": {"b": "value"}}
        result = manager._get_nested_value(config, "a.b.c")
        assert result is None

    def test_set_nested_value(self, manager):
        """Test setting nested value."""
        config = {"a": {"b": "old"}}
        manager._set_nested_value(config, "a.b", "new")
        assert config["a"]["b"] == "new"

    def test_set_nested_value_creates_path(self, manager):
        """Test that set creates missing nested dictionaries."""
        config = {}
        manager._set_nested_value(config, "a.b.c", "value")
        assert config["a"]["b"]["c"] == "value"


class TestChangeTracking:
    """Tests for change tracking."""

    def test_tracks_changes(self, manager):
        """Test that changes are tracked."""
        change = ConfigChange("field", "old", "new", ReloadStrategy.HOT_RELOAD)
        manager.track_change(change)
        history = manager.get_change_history(limit=10)
        assert len(history) == 1
        assert history[0] == change

    def test_history_limit(self, manager):
        """Test that history respects max limit."""
        for i in range(60):
            change = ConfigChange(
                f"field{i}", f"old{i}", f"new{i}", ReloadStrategy.HOT_RELOAD
            )
            manager.track_change(change)
        assert len(manager._change_history) == 50


class TestGetFields:
    """Tests for getting field lists."""

    def test_get_hot_reloadable_fields(self, manager):
        """Test getting hot-reloadable fields."""
        fields = manager.get_hot_reloadable_fields()
        assert "system_prompt_base" in fields
        assert "cortex_llm.config.temperature" in fields
        assert "hertz" not in fields

    def test_get_restart_required_fields(self, manager):
        """Test getting restart-required fields."""
        fields = manager.get_restart_required_fields()
        assert "hertz" in fields
        assert "agent_inputs" in fields
        assert "system_prompt_base" not in fields


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_full_workflow(self, manager, sample_old_config, sample_new_config):
        """Test complete workflow: detect → validate → categorize → apply."""
        # Detect
        changes = manager.detect_changes(sample_old_config, sample_new_config)
        assert len(changes) > 0

        # Validate
        validation = manager.validate_changes(changes)
        assert all(validation.values())

        # Categorize
        categorized = manager.categorize_changes(changes)
        hot_reload = categorized[ReloadStrategy.HOT_RELOAD]
        validate_first = categorized[ReloadStrategy.VALIDATE_FIRST]
        assert len(hot_reload) > 0
        assert len(validate_first) > 0

        # Apply
        config = sample_old_config.copy()
        all_changes = hot_reload + validate_first
        results = manager.apply_changes(config, all_changes)
        assert all(results.values())

        # Verify
        assert config["system_prompt_base"] == sample_new_config["system_prompt_base"]
        assert (
            config["cortex_llm"]["config"]["temperature"]
            == sample_new_config["cortex_llm"]["config"]["temperature"]
        )


class TestEdgeCases:
    """Edge case tests for coverage."""

    def test_validate_unknown_field(self, manager):
        """Test validation of field not in registry (lines 252-254)."""
        change = ConfigChange(
            field_path="unknown.field.path",
            old_value="old",
            new_value="new",
            strategy=ReloadStrategy.HOT_RELOAD,
        )
        results = manager.validate_changes([change])
        assert results["unknown.field.path"] is False

    def test_validate_with_validator_exception(self, manager):
        """Test validator that throws exception (lines 266-268)."""

        def broken_validator(value):
            raise ValueError("Validator intentionally broken")

        # Correct way to register field
        manager.register_field(
            field_path="test.broken",
            strategy=ReloadStrategy.VALIDATE_FIRST,
            validator=broken_validator,
            description="Field with broken validator",
        )

        change = ConfigChange(
            field_path="test.broken",
            old_value="old",
            new_value="new",
            strategy=ReloadStrategy.VALIDATE_FIRST,
        )
        results = manager.validate_changes([change])
        assert results["test.broken"] is False

    def test_apply_changes_with_exception(self, manager):
        """Test apply_changes when setting value fails (lines 392-396)."""
        # Correct way to register field
        manager.register_field(
            field_path="test.field",
            strategy=ReloadStrategy.HOT_RELOAD,
            description="Field that will fail to apply",
        )

        # Create config where nested field doesn't exist as dict
        config = {"test": None}  # None will cause error on nested set

        change = ConfigChange(
            field_path="test.field",
            old_value="old",
            new_value="new",
            strategy=ReloadStrategy.HOT_RELOAD,
        )

        results = manager.apply_changes(config, [change])
        assert results["test.field"] is False

    def test_validate_field_explicitly_without_validator(self, manager):
        """Test field validation without validator (lines 269-271)."""
        # Correct way to register field
        manager.register_field(
            field_path="custom.no_validator",
            strategy=ReloadStrategy.HOT_RELOAD,
            validator=None,
            description="Field without validator",
        )

        change = ConfigChange(
            field_path="custom.no_validator",
            old_value="old",
            new_value="new",
            strategy=ReloadStrategy.HOT_RELOAD,
        )

        results = manager.validate_changes([change])
        assert results["custom.no_validator"] is True
