"""
Tests for capability introspection in base classes.
"""

import pytest
from typing import Optional
from unittest.mock import MagicMock

from actions.base import ActionConfig, ActionConnector
from capabilities import CapabilityDescriptor, ComponentType, Constraint
from inputs.base import Sensor, SensorConfig


class MockActionConnector(ActionConnector[ActionConfig, str]):
    """Mock action connector without capability implementation."""

    async def connect(self, output_interface: str) -> None:
        pass


class CapableActionConnector(ActionConnector[ActionConfig, str]):
    """Mock action connector with capability implementation."""

    async def connect(self, output_interface: str) -> None:
        pass

    def get_capabilities(self) -> Optional[CapabilityDescriptor]:
        return CapabilityDescriptor(
            component_name="capable_action",
            component_type=ComponentType.ACTION,
            supported_features=["feature1", "feature2"],
            constraints=[
                Constraint(name="limit", value=100),
            ],
            is_available=True,
            description="A capable action",
        )


class MockSensor(Sensor[SensorConfig, str]):
    """Mock sensor without capability implementation."""

    async def raw_to_text(self, raw_input: str):
        pass

    def formatted_latest_buffer(self) -> str | None:
        return None


class CapableSensor(Sensor[SensorConfig, str]):
    """Mock sensor with capability implementation."""

    async def raw_to_text(self, raw_input: str):
        pass

    def formatted_latest_buffer(self) -> str | None:
        return None

    def get_capabilities(self) -> Optional[CapabilityDescriptor]:
        return CapabilityDescriptor(
            component_name="capable_sensor",
            component_type=ComponentType.SENSOR,
            supported_features=["sense1", "sense2"],
            constraints=[
                Constraint(name="update_rate", value=30, unit="Hz"),
            ],
            is_available=True,
            description="A capable sensor",
        )


class TestActionConnectorCapabilities:
    """Tests for ActionConnector.get_capabilities()."""

    def test_default_returns_none(self):
        """Test that default implementation returns None."""
        config = ActionConfig()
        connector = MockActionConnector(config)
        assert connector.get_capabilities() is None

    def test_implemented_returns_descriptor(self):
        """Test that implemented get_capabilities returns descriptor."""
        config = ActionConfig()
        connector = CapableActionConnector(config)
        caps = connector.get_capabilities()

        assert caps is not None
        assert caps.component_name == "capable_action"
        assert caps.component_type == ComponentType.ACTION
        assert "feature1" in caps.supported_features
        assert caps.get_constraint("limit").value == 100


class TestSensorCapabilities:
    """Tests for Sensor.get_capabilities()."""

    def test_default_returns_none(self):
        """Test that default implementation returns None."""
        config = SensorConfig()
        sensor = MockSensor(config)
        assert sensor.get_capabilities() is None

    def test_implemented_returns_descriptor(self):
        """Test that implemented get_capabilities returns descriptor."""
        config = SensorConfig()
        sensor = CapableSensor(config)
        caps = sensor.get_capabilities()

        assert caps is not None
        assert caps.component_name == "capable_sensor"
        assert caps.component_type == ComponentType.SENSOR
        assert "sense1" in caps.supported_features
        assert caps.get_constraint("update_rate").value == 30
        assert caps.get_constraint("update_rate").unit == "Hz"
