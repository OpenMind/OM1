from dataclasses import dataclass
from enum import Enum

from actions.base import Interface


class GPSAction(str, Enum):
    """
    Enumeration of possible GPS-related actions for location sharing.

    This enum defines the set of actions an agent can perform related to
    GPS location data. GPS (Global Positioning System) provides geographic
    coordinates that enable location-aware behaviors and peer-to-peer
    communication on the FABRIC network.

    Geographic coordinates use the WGS84 (World Geodetic System 1984)
    standard, which is the reference coordinate system used by GPS:

    - **Latitude**: Angular distance north or south of the equator,
      measured in degrees from -90° (South Pole) to +90° (North Pole).
    - **Longitude**: Angular distance east or west of the Prime Meridian
      (Greenwich, UK), measured in degrees from -180° to +180°.
    - **Yaw**: Compass heading/orientation in degrees (0-360°), where
      0° = North, 90° = East, 180° = South, 270° = West.

    Attributes
    ----------
    SHARE_LOCATION : str
        Broadcasts the agent's current GPS coordinates (latitude, longitude,
        and yaw/heading) to the FABRIC peer-to-peer network. This enables
        other agents and systems to discover and locate this agent. Useful
        for multi-agent coordination, rendezvous operations, and fleet
        management scenarios.
    IDLE : str
        No GPS action is performed. The agent maintains its current state
        without broadcasting location data. Use this when location sharing
        is not needed or to conserve network bandwidth.

    Examples
    --------
    >>> from actions.gps.interface import GPSAction
    >>> action = GPSAction.SHARE_LOCATION
    >>> print(action.value)
    'share location'

    >>> # Check if an action is valid
    >>> GPSAction("idle") == GPSAction.IDLE
    True

    >>> # Iterate over all available GPS actions
    >>> for action in GPSAction:
    ...     print(f"{action.name}: {action.value}")
    SHARE_LOCATION: share location
    IDLE: idle

    Notes
    -----
    Location sharing requires that the agent has access to GPS data through
    a GPS provider (e.g., hardware GPS module, RTK system, or simulated
    coordinates). If no GPS data is available, the SHARE_LOCATION action
    will log an error but not raise an exception.
    """

    SHARE_LOCATION = "share location"
    IDLE = "idle"


@dataclass
class GPSInput:
    """
    Input interface for the GPS action.

    This dataclass defines the input structure required to trigger a
    GPS-related action on the agent. The action determines whether the
    agent broadcasts its location to the FABRIC network or remains idle.

    Parameters
    ----------
    action : GPSAction
        The GPS action to perform. Must be one of the valid GPSAction
        enum values: SHARE_LOCATION or IDLE.

    Examples
    --------
    >>> from actions.gps.interface import GPSInput, GPSAction
    >>> # Create an input to share location
    >>> share_input = GPSInput(action=GPSAction.SHARE_LOCATION)

    >>> # Create an input to remain idle (no location broadcast)
    >>> idle_input = GPSInput(action=GPSAction.IDLE)

    >>> # Access the action value
    >>> share_input.action.value
    'share location'

    Notes
    -----
    The actual GPS coordinates (latitude, longitude, yaw) are retrieved
    from the IOProvider at execution time, not specified in this input.
    This allows the system to always use the most current location data.
    """

    action: GPSAction


@dataclass
class GPS(Interface[GPSInput, GPSInput]):
    """
    Action interface for GPS location sharing on the FABRIC network.

    This action enables an agent to broadcast its geographic coordinates
    to the FABRIC peer-to-peer network, allowing other agents and systems
    to discover its location. Location sharing is fundamental for:

    - **Multi-agent coordination**: Agents can find and navigate toward
      each other for collaborative tasks.
    - **Fleet management**: Central systems can track and monitor
      distributed robot fleets.
    - **Rendezvous operations**: Agents can agree on meeting points
      and confirm arrivals.
    - **Geofencing**: Systems can verify agents are within designated
      operational areas.

    Effect: When SHARE_LOCATION is triggered, the agent retrieves its
    current GPS coordinates from the IOProvider and broadcasts them via
    JSON-RPC to the configured FABRIC endpoint using the `omp2p_shareStatus`
    method. The broadcast includes:

    - ``latitude``: Current latitude in decimal degrees (WGS84)
    - ``longitude``: Current longitude in decimal degrees (WGS84)
    - ``yaw``: Current heading/orientation in degrees (0-360°)

    Parameters
    ----------
    input : GPSInput
        The input containing the desired GPS action to perform.
    output : GPSInput
        The output echoing the action that was performed. Matches the input
        to confirm the action was processed.

    Examples
    --------
    >>> from actions.gps.interface import GPS, GPSInput, GPSAction
    >>> # Create a GPS action to share location
    >>> gps_input = GPSInput(action=GPSAction.SHARE_LOCATION)
    >>> gps_action = GPS(input=gps_input, output=gps_input)

    >>> # Create a GPS action to remain idle
    >>> idle_input = GPSInput(action=GPSAction.IDLE)
    >>> idle_action = GPS(input=idle_input, output=idle_input)

    See Also
    --------
    actions.navigate_location.interface.NavigateLocation :
        Action for navigating to specific GPS coordinates.
    actions.remember_location.interface.RememberLocation :
        Action for storing named locations for future reference.
    inputs.plugins.gps.GPSInput :
        Input plugin that provides GPS coordinate data.

    Notes
    -----
    **Coordinate Precision**: GPS coordinates are typically accurate to
    within a few meters for consumer-grade receivers, or centimeter-level
    for RTK (Real-Time Kinematic) systems.

    **Privacy Considerations**: Location data is broadcast on the FABRIC
    network. Ensure appropriate access controls are in place for
    privacy-sensitive deployments.

    **Network Requirements**: Location sharing requires connectivity to
    the FABRIC network endpoint (default: http://localhost:8545).
    """

    input: GPSInput
    output: GPSInput
