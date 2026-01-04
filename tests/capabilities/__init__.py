"""
Unit tests for the capability introspection and negotiation layer.

Tests cover:
- CapabilityDescriptor creation and usage
- CapabilityRegistry registration and querying
- CapabilitySummary generation and formatting
"""

import pytest

from capabilities import (
    CapabilityDescriptor,
    CapabilityRegistry,
    CapabilitySummary,
    ComponentType,
    Constraint,
    get_capability_registry,
    reset_capability_registry,
)


class TestConstraint:
    """Tests for the Constraint dataclass."""

    def test_constraint_with_unit(self):
        """Test constraint with unit."""
        constraint = Constraint(name="max_speed", value=1.2, unit="m/s")
        assert constraint.name == "max_speed"
        assert constraint.value == 1.2
        assert constraint.unit == "m/s"
        assert str(constraint) == "max_speed=1.2 m/s"

    def test_constraint_without_unit(self):
        """Test constraint without unit."""
        constraint = Constraint(name="max_count", value=10)
        assert str(constraint) == "max_count=10"

    def test_constraint_with_description(self):
        """Test constraint with description."""
        constraint = Constraint(
            name="update_rate",
            value=30,
            unit="Hz",
            description="Sensor update frequency",
        )
        assert constraint.description == "Sensor update frequency"


class TestCapabilityDescriptor:
    """Tests for the CapabilityDescriptor dataclass."""

    def test_basic_descriptor(self):
        """Test creating a basic capability descriptor."""
        descriptor = CapabilityDescriptor(
            component_name="test_action",
            component_type=ComponentType.ACTION,
            supported_features=["move", "rotate"],
            is_available=True,
        )
        assert descriptor.component_name == "test_action"
        assert descriptor.component_type == ComponentType.ACTION
        assert descriptor.supported_features == ["move", "rotate"]
        assert descriptor.is_available is True
        assert descriptor.constraints == []
        assert descriptor.metadata == {}

    def test_descriptor_with_constraints(self):
        """Test descriptor with constraints."""
        constraints = [
            Constraint(name="max_speed", value=1.5, unit="m/s"),
            Constraint(name="max_payload", value=5, unit="kg"),
        ]
        descriptor = CapabilityDescriptor(
            component_name="robot_arm",
            component_type=ComponentType.ACTION,
            supported_features=["grab", "release"],
            constraints=constraints,
        )
        assert len(descriptor.constraints) == 2
        assert descriptor.get_constraint("max_speed").value == 1.5
        assert descriptor.get_constraint("nonexistent") is None

    def test_has_feature(self):
        """Test feature checking."""
        descriptor = CapabilityDescriptor(
            component_name="camera",
            component_type=ComponentType.SENSOR,
            supported_features=["RGB", "depth", "infrared"],
        )
        assert descriptor.has_feature("RGB") is True
        assert descriptor.has_feature("rgb") is True  # Case insensitive
        assert descriptor.has_feature("thermal") is False

    def test_to_summary_dict(self):
        """Test converting to summary dictionary."""
        descriptor = CapabilityDescriptor(
            component_name="lidar",
            component_type=ComponentType.SENSOR,
            supported_features=["scan"],
            constraints=[Constraint(name="range", value=10, unit="m")],
            is_available=True,
        )
        summary = descriptor.to_summary_dict()
        assert summary["name"] == "lidar"
        assert summary["type"] == "sensor"
        assert summary["features"] == ["scan"]
        assert summary["available"] is True
        assert "range" in summary["constraints"]


class TestCapabilityRegistry:
    """Tests for the CapabilityRegistry."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset global registry before each test."""
        reset_capability_registry()
        yield
        reset_capability_registry()

    def test_register_and_get(self):
        """Test registering and retrieving a descriptor."""
        registry = CapabilityRegistry()
        descriptor = CapabilityDescriptor(
            component_name="test",
            component_type=ComponentType.ACTION,
        )
        registry.register(descriptor)
        assert registry.get("test") == descriptor
        assert registry.get("nonexistent") is None

    def test_unregister(self):
        """Test unregistering a component."""
        registry = CapabilityRegistry()
        descriptor = CapabilityDescriptor(
            component_name="removable",
            component_type=ComponentType.ACTION,
        )
        registry.register(descriptor)
        assert registry.get("removable") is not None
        registry.unregister("removable")
        assert registry.get("removable") is None

    def test_get_available_actions(self):
        """Test getting available actions."""
        registry = CapabilityRegistry()
        registry.register(
            CapabilityDescriptor(
                component_name="action1",
                component_type=ComponentType.ACTION,
                is_available=True,
            )
        )
        registry.register(
            CapabilityDescriptor(
                component_name="action2",
                component_type=ComponentType.ACTION,
                is_available=False,
            )
        )
        registry.register(
            CapabilityDescriptor(
                component_name="sensor1",
                component_type=ComponentType.SENSOR,
                is_available=True,
            )
        )
        available = registry.get_available_actions()
        assert "action1" in available
        assert "action2" not in available
        assert "sensor1" not in available

    def test_get_available_sensors(self):
        """Test getting available sensors."""
        registry = CapabilityRegistry()
        registry.register(
            CapabilityDescriptor(
                component_name="camera",
                component_type=ComponentType.SENSOR,
                is_available=True,
            )
        )
        registry.register(
            CapabilityDescriptor(
                component_name="broken_lidar",
                component_type=ComponentType.SENSOR,
                is_available=False,
            )
        )
        available = registry.get_available_sensors()
        assert "camera" in available
        assert "broken_lidar" not in available

    def test_update_availability(self):
        """Test updating component availability."""
        registry = CapabilityRegistry()
        registry.register(
            CapabilityDescriptor(
                component_name="sensor",
                component_type=ComponentType.SENSOR,
                is_available=True,
            )
        )
        assert registry.get("sensor").is_available is True
        registry.update_availability("sensor", False)
        assert registry.get("sensor").is_available is False

    def test_get_all_constraints(self):
        """Test getting all constraints."""
        registry = CapabilityRegistry()
        registry.register(
            CapabilityDescriptor(
                component_name="motor",
                component_type=ComponentType.ACTION,
                constraints=[
                    Constraint(name="max_rpm", value=3000, unit="rpm"),
                ],
            )
        )
        registry.register(
            CapabilityDescriptor(
                component_name="sensor",
                component_type=ComponentType.SENSOR,
                # No constraints
            )
        )
        all_constraints = registry.get_all_constraints()
        assert "motor" in all_constraints
        assert "sensor" not in all_constraints

    def test_global_registry(self):
        """Test global registry singleton."""
        reg1 = get_capability_registry()
        reg2 = get_capability_registry()
        assert reg1 is reg2


class TestCapabilitySummary:
    """Tests for CapabilitySummary."""

    @pytest.fixture
    def populated_registry(self):
        """Create a registry with test data."""
        registry = CapabilityRegistry()
        registry.register(
            CapabilityDescriptor(
                component_name="move",
                component_type=ComponentType.ACTION,
                supported_features=["walk", "run", "jump"],
                constraints=[
                    Constraint(name="max_speed", value=1.2, unit="m/s"),
                ],
                is_available=True,
            )
        )
        registry.register(
            CapabilityDescriptor(
                component_name="grab",
                component_type=ComponentType.ACTION,
                supported_features=["grasp", "release"],
                is_available=False,  # Unavailable
            )
        )
        registry.register(
            CapabilityDescriptor(
                component_name="camera_rgb",
                component_type=ComponentType.SENSOR,
                supported_features=["capture", "stream"],
                constraints=[
                    Constraint(name="resolution", value="1080p"),
                ],
                is_available=True,
            )
        )
        return registry

    def test_available_actions(self, populated_registry):
        """Test getting available actions from summary."""
        summary = CapabilitySummary(populated_registry)
        assert "move" in summary.available_actions
        assert "grab" not in summary.available_actions

    def test_available_sensors(self, populated_registry):
        """Test getting available sensors from summary."""
        summary = CapabilitySummary(populated_registry)
        assert "camera_rgb" in summary.available_sensors

    def test_constraints(self, populated_registry):
        """Test getting constraints from summary."""
        summary = CapabilitySummary(populated_registry)
        constraints = summary.constraints
        assert "move.max_speed" in constraints
        assert "camera_rgb.resolution" in constraints

    def test_to_prompt_string(self, populated_registry):
        """Test generating prompt-friendly string."""
        summary = CapabilitySummary(populated_registry)
        prompt = summary.to_prompt_string()

        assert "RUNTIME CAPABILITIES:" in prompt
        assert "Available actions:" in prompt
        assert "move" in prompt
        assert "Available sensors:" in prompt
        assert "camera_rgb" in prompt
        assert "Constraints:" in prompt
        assert "Unavailable:" in prompt
        assert "grab" in prompt

    def test_to_dict(self, populated_registry):
        """Test converting to dictionary."""
        summary = CapabilitySummary(populated_registry)
        data = summary.to_dict()

        assert "available_actions" in data
        assert "available_sensors" in data
        assert "constraints" in data
        assert "components" in data
        assert len(data["components"]) == 3
