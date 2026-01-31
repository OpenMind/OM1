"""
Comprehensive Test Suite for Hot Reload Module

Tests cover:
- Reload strategies and field categorization
- Config diff engine with deep comparison
- File watcher with debouncing
- Validator with custom validators
- Manager orchestration and events
- Edge cases and error handling
"""

import json
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

# Import modules under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from runtime.hot_reload.strategies import FieldCategory, ReloadStrategy
from runtime.hot_reload.diff import (
    ConfigDiff,
    compute_config_diff,
    get_nested_diff,
    _deep_equals,
    _hash_value,
)
from runtime.hot_reload.validator import (
    ConfigValidator,
    ValidationResult,
    ValidationReport,
)
from runtime.hot_reload.watcher import ConfigFileWatcher
from runtime.hot_reload.manager import HotReloadManager, ReloadEvent


# =============================================================================
# Tests for Strategies Module
# =============================================================================

class TestReloadStrategy:
    """Tests for ReloadStrategy enum."""
    
    def test_strategy_values_exist(self):
        """All expected strategies should exist."""
        assert ReloadStrategy.HOT_RELOAD
        assert ReloadStrategy.VALIDATE_FIRST
        assert ReloadStrategy.RESTART_REQUIRED
        assert ReloadStrategy.IGNORE
    
    def test_strategies_are_unique(self):
        """Each strategy should have a unique value."""
        values = [s.value for s in ReloadStrategy]
        assert len(values) == len(set(values))


class TestFieldCategory:
    """Tests for field categorization."""
    
    def test_hot_reload_fields(self):
        """Fields in HOT_RELOAD_FIELDS should return HOT_RELOAD strategy."""
        for field in ["name", "system_prompt_base", "system_governance"]:
            assert FieldCategory.get_strategy(field) == ReloadStrategy.HOT_RELOAD
    
    def test_validate_first_fields(self):
        """Fields requiring validation should return VALIDATE_FIRST."""
        assert FieldCategory.get_strategy("hertz") == ReloadStrategy.VALIDATE_FIRST
    
    def test_restart_required_fields(self):
        """Critical fields should require restart."""
        for field in ["cortex_llm", "agent_inputs", "agent_actions"]:
            assert FieldCategory.get_strategy(field) == ReloadStrategy.RESTART_REQUIRED
    
    def test_ignore_fields(self):
        """Internal fields should be ignored."""
        for field in ["$schema", "_version"]:
            assert FieldCategory.get_strategy(field) == ReloadStrategy.IGNORE
    
    def test_unknown_fields_require_restart(self):
        """Unknown fields should default to requiring restart for safety."""
        assert FieldCategory.get_strategy("unknown_field_xyz") == ReloadStrategy.RESTART_REQUIRED
    
    def test_is_safe_field(self):
        """Safe field check should work correctly."""
        assert FieldCategory.is_safe_field("name") is True
        assert FieldCategory.is_safe_field("hertz") is True
        assert FieldCategory.is_safe_field("cortex_llm") is False
    
    def test_requires_restart(self):
        """Restart requirement check should work correctly."""
        assert FieldCategory.requires_restart("cortex_llm") is True
        assert FieldCategory.requires_restart("name") is False
    
    def test_categorize_changes(self):
        """Changes should be categorized correctly."""
        changed = {"name", "hertz", "cortex_llm", "$schema"}
        categories = FieldCategory.categorize_changes(changed)
        
        assert "name" in categories[ReloadStrategy.HOT_RELOAD]
        assert "hertz" in categories[ReloadStrategy.VALIDATE_FIRST]
        assert "cortex_llm" in categories[ReloadStrategy.RESTART_REQUIRED]
        assert "$schema" in categories[ReloadStrategy.IGNORE]


# =============================================================================
# Tests for Diff Module
# =============================================================================

class TestDeepEquals:
    """Tests for deep equality comparison."""
    
    def test_primitive_equality(self):
        """Primitive types should compare correctly."""
        assert _deep_equals(1, 1) is True
        assert _deep_equals(1, 2) is False
        assert _deep_equals("a", "a") is True
        assert _deep_equals("a", "b") is False
        assert _deep_equals(True, True) is True
        assert _deep_equals(True, False) is False
    
    def test_dict_equality(self):
        """Dictionaries should compare deeply."""
        assert _deep_equals({"a": 1}, {"a": 1}) is True
        assert _deep_equals({"a": 1}, {"a": 2}) is False
        assert _deep_equals({"a": 1}, {"b": 1}) is False
        assert _deep_equals({"a": {"b": 1}}, {"a": {"b": 1}}) is True
        assert _deep_equals({"a": {"b": 1}}, {"a": {"b": 2}}) is False
    
    def test_list_equality(self):
        """Lists should compare deeply."""
        assert _deep_equals([1, 2, 3], [1, 2, 3]) is True
        assert _deep_equals([1, 2, 3], [1, 2, 4]) is False
        assert _deep_equals([1, 2], [1, 2, 3]) is False
        assert _deep_equals([{"a": 1}], [{"a": 1}]) is True
        assert _deep_equals([{"a": 1}], [{"a": 2}]) is False
    
    def test_type_mismatch(self):
        """Different types should not be equal."""
        assert _deep_equals(1, "1") is False
        assert _deep_equals([], {}) is False
        assert _deep_equals(None, 0) is False
    
    def test_complex_nested_structure(self):
        """Complex nested structures should compare correctly."""
        old = {
            "agent_inputs": [
                {"type": "VLMInput", "config": {"model": "gpt-4"}},
                {"type": "ASRInput", "config": {"lang": "en"}},
            ]
        }
        new_same = {
            "agent_inputs": [
                {"type": "VLMInput", "config": {"model": "gpt-4"}},
                {"type": "ASRInput", "config": {"lang": "en"}},
            ]
        }
        new_diff = {
            "agent_inputs": [
                {"type": "VLMInput", "config": {"model": "gpt-4o"}},  # Changed!
                {"type": "ASRInput", "config": {"lang": "en"}},
            ]
        }
        
        assert _deep_equals(old, new_same) is True
        assert _deep_equals(old, new_diff) is False


class TestConfigDiff:
    """Tests for ConfigDiff dataclass."""
    
    def test_empty_diff(self):
        """Empty diff should have no changes."""
        diff = ConfigDiff()
        assert diff.has_changes is False
        assert diff.changed_fields == set()
    
    def test_diff_with_changes(self):
        """Diff with changes should report correctly."""
        diff = ConfigDiff(
            added={"new_field"},
            modified={"changed_field"},
            removed={"old_field"},
        )
        assert diff.has_changes is True
        assert diff.changed_fields == {"new_field", "changed_field", "old_field"}


class TestComputeConfigDiff:
    """Tests for compute_config_diff function."""
    
    def test_identical_configs(self):
        """Identical configs should produce no diff."""
        config = {"name": "test", "hertz": 10}
        diff = compute_config_diff(config, config.copy())
        assert diff.has_changes is False
    
    def test_added_field(self):
        """New fields should be detected."""
        old = {"name": "test"}
        new = {"name": "test", "hertz": 10}
        diff = compute_config_diff(old, new)
        
        assert "hertz" in diff.added
        assert diff.new_values["hertz"] == 10
    
    def test_removed_field(self):
        """Removed fields should be detected."""
        old = {"name": "test", "hertz": 10}
        new = {"name": "test"}
        diff = compute_config_diff(old, new)
        
        assert "hertz" in diff.removed
        assert diff.old_values["hertz"] == 10
    
    def test_modified_field(self):
        """Modified fields should be detected."""
        old = {"name": "test", "hertz": 10}
        new = {"name": "test", "hertz": 20}
        diff = compute_config_diff(old, new)
        
        assert "hertz" in diff.modified
        assert diff.old_values["hertz"] == 10
        assert diff.new_values["hertz"] == 20
    
    def test_unchanged_field(self):
        """Unchanged fields should be tracked."""
        old = {"name": "test", "hertz": 10}
        new = {"name": "test", "hertz": 20}
        diff = compute_config_diff(old, new)
        
        assert "name" in diff.unchanged
    
    def test_complex_field_change_detected(self):
        """
        Changes in complex nested fields should be detected.
        This tests the fix for the bug in PR #1312.
        """
        old = {
            "agent_inputs": [
                {"type": "VLMInput", "model": "gpt-4"},
            ]
        }
        new = {
            "agent_inputs": [
                {"type": "VLMInput", "model": "gpt-4o"},  # Content changed
            ]
        }
        diff = compute_config_diff(old, new)
        
        assert "agent_inputs" in diff.modified
    
    def test_same_length_list_different_content(self):
        """
        Lists with same length but different content should be detected.
        This is the key regression test for PR #1312 bug.
        """
        old = {"items": [1, 2, 3]}
        new = {"items": [1, 2, 4]}  # Same length, different content
        diff = compute_config_diff(old, new)
        
        assert "items" in diff.modified


class TestGetNestedDiff:
    """Tests for detailed nested diff."""
    
    def test_nested_changes(self):
        """Nested changes should include full path."""
        old = {"a": {"b": {"c": 1}}}
        new = {"a": {"b": {"c": 2}}}
        
        changes = get_nested_diff(old, new)
        
        assert len(changes) == 1
        path, change_type, old_val, new_val = changes[0]
        assert path == "a.b.c"
        assert change_type == "modified"
        assert old_val == 1
        assert new_val == 2


# =============================================================================
# Tests for Validator Module
# =============================================================================

class TestValidationResult:
    """Tests for ValidationResult."""
    
    def test_valid_result_is_truthy(self):
        """Valid result should be truthy."""
        result = ValidationResult(is_valid=True, field_name="test", message="ok")
        assert bool(result) is True
    
    def test_invalid_result_is_falsy(self):
        """Invalid result should be falsy."""
        result = ValidationResult(is_valid=False, field_name="test", message="bad")
        assert bool(result) is False


class TestValidationReport:
    """Tests for ValidationReport."""
    
    def test_empty_report_is_valid(self):
        """Empty report should be valid."""
        report = ValidationReport()
        assert report.is_valid is True
        assert report.passed == 0
        assert report.failed == 0
    
    def test_all_passed(self):
        """Report with all passed should be valid."""
        report = ValidationReport()
        report.add(ValidationResult(True, "a", "ok"))
        report.add(ValidationResult(True, "b", "ok"))
        
        assert report.is_valid is True
        assert report.passed == 2
        assert report.failed == 0
    
    def test_one_failed(self):
        """Report with one failure should be invalid."""
        report = ValidationReport()
        report.add(ValidationResult(True, "a", "ok"))
        report.add(ValidationResult(False, "b", "bad"))
        
        assert report.is_valid is False
        assert report.passed == 1
        assert report.failed == 1
        assert "b" in report.failed_fields


class TestConfigValidator:
    """Tests for ConfigValidator."""
    
    def test_hertz_valid(self):
        """Valid hertz should pass."""
        validator = ConfigValidator()
        result = validator.validate_field("hertz", 10)
        assert result.is_valid is True
    
    def test_hertz_negative(self):
        """Negative hertz should fail."""
        validator = ConfigValidator()
        result = validator.validate_field("hertz", -5)
        assert result.is_valid is False
        assert result.suggested_value is not None
    
    def test_hertz_zero(self):
        """Zero hertz should fail."""
        validator = ConfigValidator()
        result = validator.validate_field("hertz", 0)
        assert result.is_valid is False
    
    def test_hertz_too_high(self):
        """Very high hertz should fail."""
        validator = ConfigValidator()
        result = validator.validate_field("hertz", 500)
        assert result.is_valid is False
    
    def test_hertz_non_numeric(self):
        """Non-numeric hertz should fail."""
        validator = ConfigValidator()
        result = validator.validate_field("hertz", "fast")
        assert result.is_valid is False
    
    def test_name_valid(self):
        """Valid name should pass."""
        validator = ConfigValidator()
        result = validator.validate_field("name", "MyAgent")
        assert result.is_valid is True
    
    def test_name_empty(self):
        """Empty name should fail."""
        validator = ConfigValidator()
        result = validator.validate_field("name", "")
        assert result.is_valid is False
    
    def test_prompt_valid(self):
        """Valid prompt should pass."""
        validator = ConfigValidator()
        result = validator.validate_field("system_prompt_base", "You are a robot.")
        assert result.is_valid is True
    
    def test_prompt_none_allowed(self):
        """None prompt should be allowed."""
        validator = ConfigValidator()
        result = validator.validate_field("system_prompt_base", None)
        assert result.is_valid is True
    
    def test_unknown_field_passes(self):
        """Unknown fields should pass (no validator)."""
        validator = ConfigValidator()
        result = validator.validate_field("unknown_xyz", "anything")
        assert result.is_valid is True
    
    def test_custom_validator(self):
        """Custom validators should work."""
        validator = ConfigValidator()
        
        def custom_check(field_name: str, value: Any) -> ValidationResult:
            if value == "secret":
                return ValidationResult(False, field_name, "Cannot be 'secret'")
            return ValidationResult(True, field_name, "ok")
        
        validator.register_validator("my_field", custom_check)
        
        assert validator.validate_field("my_field", "normal").is_valid is True
        assert validator.validate_field("my_field", "secret").is_valid is False
    
    def test_validate_changes(self):
        """Batch validation should work."""
        validator = ConfigValidator()
        changes = {
            "hertz": 10,
            "name": "Agent",
        }
        report = validator.validate_changes(changes)
        assert report.is_valid is True
    
    def test_validate_changes_with_failure(self):
        """Batch validation should report failures."""
        validator = ConfigValidator()
        changes = {
            "hertz": -5,  # Invalid
            "name": "Agent",  # Valid
        }
        report = validator.validate_changes(changes)
        assert report.is_valid is False
        assert "hertz" in report.failed_fields


# =============================================================================
# Tests for Watcher Module
# =============================================================================

class TestConfigFileWatcher:
    """Tests for ConfigFileWatcher."""
    
    def test_watcher_starts_and_stops(self):
        """Watcher should start and stop cleanly."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            f.write(b'{"name": "test"}')
            f.flush()
            
            callback = MagicMock()
            watcher = ConfigFileWatcher(f.name, callback)
            
            assert watcher.is_running is False
            watcher.start()
            assert watcher.is_running is True
            watcher.stop()
            assert watcher.is_running is False
    
    def test_manual_trigger(self):
        """Manual trigger should call callback."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            f.write(b'{"name": "test"}')
            f.flush()
            
            callback = MagicMock()
            watcher = ConfigFileWatcher(f.name, callback, debounce_seconds=0)
            
            watcher.trigger_reload()
            time.sleep(0.1)
            
            callback.assert_called_once()
    
    def test_debouncing(self):
        """Rapid changes should be debounced."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            f.write(b'{"name": "test"}')
            f.flush()
            
            callback = MagicMock()
            watcher = ConfigFileWatcher(f.name, callback, debounce_seconds=0.5)
            
            # Trigger multiple times rapidly
            watcher.trigger_reload()
            watcher.trigger_reload()
            watcher.trigger_reload()
            
            time.sleep(0.1)
            
            # Should only be called once due to debouncing
            assert callback.call_count == 1


# =============================================================================
# Tests for Manager Module
# =============================================================================

class TestHotReloadManager:
    """Tests for HotReloadManager."""
    
    def test_manager_loads_initial_config(self):
        """Manager should load config on init."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump({"name": "TestAgent", "hertz": 10}, f)
            f.flush()
            
            manager = HotReloadManager(f.name)
            
            assert manager.current_config["name"] == "TestAgent"
            assert manager.current_config["hertz"] == 10
    
    def test_manager_starts_and_stops(self):
        """Manager should start and stop cleanly."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump({"name": "test"}, f)
            f.flush()
            
            manager = HotReloadManager(f.name)
            
            assert manager.is_running is False
            manager.start()
            assert manager.is_running is True
            manager.stop()
            assert manager.is_running is False
    
    def test_callback_registration(self):
        """Callbacks should be registered and removed."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump({"name": "test"}, f)
            f.flush()
            
            manager = HotReloadManager(f.name)
            callback = MagicMock()
            
            manager.on_reload(callback)
            assert manager.remove_callback(callback) is True
            assert manager.remove_callback(callback) is False  # Already removed
    
    def test_get_field_value(self):
        """Should retrieve individual field values."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump({"name": "TestAgent", "hertz": 10}, f)
            f.flush()
            
            manager = HotReloadManager(f.name)
            
            assert manager.get_field_value("name") == "TestAgent"
            assert manager.get_field_value("hertz") == 10
            assert manager.get_field_value("nonexistent") is None
    
    def test_reload_stats_empty(self):
        """Stats should handle empty history."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump({"name": "test"}, f)
            f.flush()
            
            manager = HotReloadManager(f.name)
            stats = manager.get_reload_stats()
            
            assert stats["total_reloads"] == 0
            assert stats["successful"] == 0
    
    def test_apply_callback_called_on_change(self):
        """Apply callback should be called when config changes."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump({"name": "OldName", "hertz": 10}, f)
            f.flush()
            config_path = f.name
        
        apply_callback = MagicMock()
        manager = HotReloadManager(
            config_path,
            apply_callback=apply_callback,
            enable_validation=True,
        )
        
        # Modify the config file
        with open(config_path, "w") as f:
            json.dump({"name": "NewName", "hertz": 10}, f)
        
        # Trigger reload
        manager.trigger_reload()
        time.sleep(0.2)
        
        # Check callback was called
        apply_callback.assert_called_once()
        call_args = apply_callback.call_args
        changed_values, requires_restart = call_args[0]
        
        assert "name" in changed_values
        assert changed_values["name"] == "NewName"
        assert requires_restart is False  # name is a safe field
    
    def test_restart_required_for_unsafe_field(self):
        """Unsafe field changes should require restart."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump({"name": "Agent", "cortex_llm": {"provider": "openai"}}, f)
            f.flush()
            config_path = f.name
        
        apply_callback = MagicMock()
        manager = HotReloadManager(
            config_path,
            apply_callback=apply_callback,
        )
        
        # Modify unsafe field
        with open(config_path, "w") as f:
            json.dump({"name": "Agent", "cortex_llm": {"provider": "anthropic"}}, f)
        
        manager.trigger_reload()
        time.sleep(0.2)
        
        call_args = apply_callback.call_args
        _, requires_restart = call_args[0]
        
        assert requires_restart is True
    
    def test_validation_failure_prevents_apply(self):
        """Invalid changes should not be applied."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump({"name": "Agent", "hertz": 10}, f)
            f.flush()
            config_path = f.name
        
        apply_callback = MagicMock()
        manager = HotReloadManager(
            config_path,
            apply_callback=apply_callback,
            enable_validation=True,
        )
        
        # Set invalid hertz
        with open(config_path, "w") as f:
            json.dump({"name": "Agent", "hertz": -5}, f)
        
        manager.trigger_reload()
        time.sleep(0.2)
        
        # Apply should NOT be called
        apply_callback.assert_not_called()
        
        # Original config should be preserved
        assert manager.get_field_value("hertz") == 10
    
    def test_reload_event_callback(self):
        """Reload event callbacks should be notified."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump({"name": "Agent"}, f)
            f.flush()
            config_path = f.name
        
        event_callback = MagicMock()
        manager = HotReloadManager(config_path)
        manager.on_reload(event_callback)
        
        # Modify config
        with open(config_path, "w") as f:
            json.dump({"name": "NewAgent"}, f)
        
        manager.trigger_reload()
        time.sleep(0.2)
        
        event_callback.assert_called_once()
        event = event_callback.call_args[0][0]
        
        assert isinstance(event, ReloadEvent)
        assert event.success is True
    
    def test_history_tracking(self):
        """Reload history should be tracked."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump({"name": "Agent"}, f)
            f.flush()
            config_path = f.name
        
        manager = HotReloadManager(config_path, max_history=5, debounce_seconds=0.05)
        
        # Perform several reloads with enough time between each
        for i in range(3):
            with open(config_path, "w") as f:
                json.dump({"name": f"Agent{i}"}, f)
            manager.trigger_reload()
            time.sleep(0.3)  # Wait longer than debounce
        
        history = manager.history
        # At least one reload should be tracked (debouncing may affect count)
        assert len(history) >= 1
    
    def test_history_max_limit(self):
        """History should be limited to max_history."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump({"name": "Agent"}, f)
            f.flush()
            config_path = f.name
        
        manager = HotReloadManager(config_path, max_history=3)
        
        # Perform more reloads than max_history
        for i in range(5):
            with open(config_path, "w") as f:
                json.dump({"name": f"Agent{i}"}, f)
            manager.trigger_reload()
            time.sleep(0.15)
        
        history = manager.history
        assert len(history) <= 3


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """End-to-end integration tests."""
    
    def test_full_hot_reload_workflow(self):
        """Test complete hot-reload workflow."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            initial_config = {
                "name": "TestRobot",
                "hertz": 10,
                "system_prompt_base": "You are a helpful robot.",
            }
            json.dump(initial_config, f)
            f.flush()
            config_path = f.name
        
        # Track applied changes
        applied_changes = []
        
        def track_changes(changes: Dict[str, Any], restart: bool):
            applied_changes.append((changes.copy(), restart))
        
        # Create manager
        manager = HotReloadManager(
            config_path,
            apply_callback=track_changes,
            enable_validation=True,
        )
        
        # Verify initial load
        assert manager.get_field_value("name") == "TestRobot"
        
        # Update safe fields
        with open(config_path, "w") as f:
            json.dump({
                "name": "UpdatedRobot",
                "hertz": 20,
                "system_prompt_base": "New prompt.",
            }, f)
        
        manager.trigger_reload()
        time.sleep(0.2)
        
        # Verify changes were applied
        assert len(applied_changes) == 1
        changes, restart = applied_changes[0]
        assert restart is False  # All safe fields
        assert "name" in changes
        assert changes["name"] == "UpdatedRobot"
        
        # Verify internal state updated
        assert manager.get_field_value("name") == "UpdatedRobot"
        assert manager.get_field_value("hertz") == 20


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
