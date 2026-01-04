"""
Capability Introspection and Negotiation Layer.

This module provides a runtime capability awareness layer for OM1. It allows
HAL components (actions, sensors) to expose their capabilities, constraints,
and supported features at runtime.

Key components:
- CapabilityDescriptor: Describes what a component can do
- CapabilityRegistry: Collects and manages all capability descriptors
- CapabilitySummary: Generates a summarized view for the agent context
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class ComponentType(str, Enum):
    """Type of component exposing capabilities."""

    ACTION = "action"
    SENSOR = "sensor"
    BACKGROUND = "background"
    SIMULATOR = "simulator"


@dataclass
class Constraint:
    """
    Represents a constraint or limit on a capability.

    Parameters
    ----------
    name : str
        Name of the constraint (e.g., "max_speed", "update_rate")
    value : Any
        The constraint value
    unit : Optional[str]
        Unit of measurement (e.g., "m/s", "Hz")
    description : Optional[str]
        Human-readable description of the constraint
    """

    name: str
    value: Any
    unit: Optional[str] = None
    description: Optional[str] = None

    def __str__(self) -> str:
        if self.unit:
            return f"{self.name}={self.value} {self.unit}"
        return f"{self.name}={self.value}"


@dataclass
class CapabilityDescriptor:
    """
    Describes the capabilities of a HAL component.

    This descriptor is returned by the optional get_capabilities() method
    on actions, sensors, and other plugins.

    Parameters
    ----------
    component_name : str
        Unique identifier for this component
    component_type : ComponentType
        Type of component (action, sensor, etc.)
    supported_features : List[str]
        List of supported features/operations
    constraints : List[Constraint]
        List of constraints/limits
    is_available : bool
        Whether the component is currently operational
    metadata : Dict[str, Any]
        Additional metadata about the component
    description : Optional[str]
        Human-readable description of the component
    """

    component_name: str
    component_type: ComponentType
    supported_features: List[str] = field(default_factory=list)
    constraints: List[Constraint] = field(default_factory=list)
    is_available: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    description: Optional[str] = None

    def get_constraint(self, name: str) -> Optional[Constraint]:
        """Get a constraint by name."""
        for constraint in self.constraints:
            if constraint.name == name:
                return constraint
        return None

    def has_feature(self, feature: str) -> bool:
        """Check if a feature is supported."""
        return feature.lower() in [f.lower() for f in self.supported_features]

    def to_summary_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary for summarization."""
        return {
            "name": self.component_name,
            "type": self.component_type.value,
            "features": self.supported_features,
            "constraints": {c.name: str(c) for c in self.constraints},
            "available": self.is_available,
        }


class CapabilityRegistry:
    """
    Central registry for collecting and managing capability descriptors.

    The registry collects capabilities from all HAL components at startup
    and provides methods for querying and summarizing available capabilities.
    """

    def __init__(self):
        """Initialize an empty capability registry."""
        self._descriptors: Dict[str, CapabilityDescriptor] = {}
        self._actions: Set[str] = set()
        self._sensors: Set[str] = set()

    def register(self, descriptor: CapabilityDescriptor) -> None:
        """
        Register a capability descriptor.

        Parameters
        ----------
        descriptor : CapabilityDescriptor
            The capability descriptor to register
        """
        self._descriptors[descriptor.component_name] = descriptor

        if descriptor.component_type == ComponentType.ACTION:
            self._actions.add(descriptor.component_name)
        elif descriptor.component_type == ComponentType.SENSOR:
            self._sensors.add(descriptor.component_name)

    def unregister(self, component_name: str) -> None:
        """
        Unregister a component by name.

        Parameters
        ----------
        component_name : str
            Name of the component to unregister
        """
        if component_name in self._descriptors:
            descriptor = self._descriptors.pop(component_name)
            if descriptor.component_type == ComponentType.ACTION:
                self._actions.discard(component_name)
            elif descriptor.component_type == ComponentType.SENSOR:
                self._sensors.discard(component_name)

    def get(self, component_name: str) -> Optional[CapabilityDescriptor]:
        """
        Get a capability descriptor by component name.

        Parameters
        ----------
        component_name : str
            Name of the component

        Returns
        -------
        Optional[CapabilityDescriptor]
            The descriptor if found, None otherwise
        """
        return self._descriptors.get(component_name)

    def get_all(self) -> List[CapabilityDescriptor]:
        """Get all registered capability descriptors."""
        return list(self._descriptors.values())

    def get_available_actions(self) -> List[str]:
        """
        Get list of available action names.

        Returns
        -------
        List[str]
            Names of all actions that are currently available
        """
        return [
            name
            for name in self._actions
            if self._descriptors[name].is_available
        ]

    def get_available_sensors(self) -> List[str]:
        """
        Get list of available sensor names.

        Returns
        -------
        List[str]
            Names of all sensors that are currently available
        """
        return [
            name
            for name in self._sensors
            if self._descriptors[name].is_available
        ]

    def get_all_constraints(self) -> Dict[str, List[Constraint]]:
        """
        Get all constraints grouped by component.

        Returns
        -------
        Dict[str, List[Constraint]]
            Mapping of component names to their constraints
        """
        return {
            name: desc.constraints
            for name, desc in self._descriptors.items()
            if desc.constraints
        }

    def update_availability(self, component_name: str, is_available: bool) -> None:
        """
        Update the availability status of a component.

        Parameters
        ----------
        component_name : str
            Name of the component
        is_available : bool
            New availability status
        """
        if component_name in self._descriptors:
            self._descriptors[component_name].is_available = is_available

    def generate_summary(self) -> "CapabilitySummary":
        """
        Generate a summarized view of all capabilities for agent context.

        Returns
        -------
        CapabilitySummary
            A summary object suitable for injection into agent context
        """
        return CapabilitySummary(self)


@dataclass
class CapabilitySummary:
    """
    A summarized view of system capabilities for agent context.

    This summary is injected into the agent context before inference,
    allowing the agent to reason about available hardware and constraints.
    """

    _registry: "CapabilityRegistry"

    @property
    def available_actions(self) -> List[str]:
        """Get list of available action names."""
        return self._registry.get_available_actions()

    @property
    def available_sensors(self) -> List[str]:
        """Get list of available sensor names."""
        return self._registry.get_available_sensors()

    @property
    def constraints(self) -> Dict[str, str]:
        """Get flattened constraints as name=value strings."""
        result = {}
        for component_name, constraints in self._registry.get_all_constraints().items():
            for constraint in constraints:
                key = f"{component_name}.{constraint.name}"
                result[key] = str(constraint)
        return result

    def to_prompt_string(self) -> str:
        """
        Generate a prompt-friendly string representation.

        Returns
        -------
        str
            Formatted string for injection into agent context
        """
        lines = ["RUNTIME CAPABILITIES:"]

        # Available actions
        actions = self.available_actions
        if actions:
            lines.append(f"Available actions: {', '.join(actions)}")
        else:
            lines.append("Available actions: none")

        # Available sensors
        sensors = self.available_sensors
        if sensors:
            lines.append(f"Available sensors: {', '.join(sensors)}")
        else:
            lines.append("Available sensors: none")

        # Key constraints
        constraints = self.constraints
        if constraints:
            lines.append("Constraints:")
            for key, value in constraints.items():
                lines.append(f"  {value}")

        # Unavailable components
        unavailable = [
            desc.component_name
            for desc in self._registry.get_all()
            if not desc.is_available
        ]
        if unavailable:
            lines.append(f"Unavailable: {', '.join(unavailable)}")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to a dictionary representation.

        Returns
        -------
        Dict[str, Any]
            Dictionary with all capability information
        """
        return {
            "available_actions": self.available_actions,
            "available_sensors": self.available_sensors,
            "constraints": self.constraints,
            "components": [
                desc.to_summary_dict() for desc in self._registry.get_all()
            ],
        }


# Global registry instance (singleton pattern)
_global_registry: Optional[CapabilityRegistry] = None


def get_capability_registry() -> CapabilityRegistry:
    """
    Get or create the global capability registry.

    Returns
    -------
    CapabilityRegistry
        The global capability registry instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = CapabilityRegistry()
    return _global_registry


def reset_capability_registry() -> None:
    """Reset the global capability registry (mainly for testing)."""
    global _global_registry
    _global_registry = None
