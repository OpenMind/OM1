"""Zenoh message types and session management module.

This module implements a comprehensive message type system for distributed robot
communication architectures utilizing the Zenoh middleware protocol. The module
provides a structured taxonomy of message types that facilitate type-safe data
exchange across heterogeneous robotic systems, encompassing temporal primitives,
spatial representations, sensor data structures, and system status indicators.

The module's architectural design follows a hierarchical namespace organization
pattern, wherein message types are categorized into semantic domains that reflect
their functional roles within robotic communication protocols. This organization
methodology enables developers to leverage type annotations and static analysis
tools to ensure protocol compliance and reduce runtime errors in distributed
systems.

The exported message type categories include:
- std_msgs: Fundamental temporal and structural primitives (Time, Duration, Header)
- status_msgs: System state and inter-component communication protocols
- geographic_msgs: Geospatial coordinate representations and transformations
- geometry_msgs: Geometric primitives and transformations (Point, Pose, Twist)
- nav_msgs: Navigation and path planning data structures
- sensor_msgs: Sensor fusion and perception data representations

Additionally, the module provides session management utilities that abstract
the complexity of Zenoh configuration and connection establishment, implementing
a fallback mechanism that prioritizes local connections while maintaining
backward compatibility with network discovery protocols.
"""

from __future__ import annotations

from . import session
from .idl import (
    IMU,
    Accel,
    AccelWithCovariance,
    AccelWithCovarianceStamped,
    AIStatusRequest,
    AIStatusResponse,
    AMCLPose,
    ASRText,
    AudioStatus,
    AvatarFaceRequest,
    AvatarFaceResponse,
    BatteryState,
    CameraInfo,
    CameraStatus,
    ColorRGBA,
    ConfigRequest,
    ConfigResponse,
    Detection,
    DockStatus,
    Duration,
    GeoPoint,
    GeoPointStamped,
    GoalID,
    GoalInfo,
    GoalStatus,
    HazardDetection,
    HazardDetectionVector,
    Header,
    Image,
    LaserScan,
    ModeStatusRequest,
    ModeStatusResponse,
    Nav2Status,
    NavSatFix,
    NavSatStatus,
    Odometry,
    Paths,
    Point,
    Point32,
    PointCloud,
    PointCloud2,
    PointField,
    Pose,
    PoseStamped,
    PoseWithCovariance,
    PoseWithCovarianceStamped,
    Quaternion,
    RegionOfInterest,
    String,
    Time,
    TTSStatusRequest,
    TTSStatusResponse,
    Twist,
    TwistWithCovariance,
    TwistWithCovarianceStamped,
    Vector3,
    geographic_msgs,
    geometry_msgs,
    nav_msgs,
    prepare_header,
    sensor_msgs,
    status_msgs,
    std_msgs,
)
from .session import create_zenoh_config, open_zenoh_session

__all__: list[str] = [
    # std_msgs
    "Time",
    "Duration",
    "Header",
    "ColorRGBA",
    "String",
    "prepare_header",
    # status_msgs
    "AudioStatus",
    "CameraStatus",
    "AIStatusRequest",
    "AIStatusResponse",
    "ASRText",
    "AvatarFaceRequest",
    "AvatarFaceResponse",
    "ConfigRequest",
    "ConfigResponse",
    "ModeStatusRequest",
    "ModeStatusResponse",
    "TTSStatusRequest",
    "TTSStatusResponse",
    # geographic_msgs
    "GeoPoint",
    "GeoPointStamped",
    # geometry_msgs
    "Point",
    "Point32",
    "Quaternion",
    "Pose",
    "PoseStamped",
    "PoseWithCovariance",
    "PoseWithCovarianceStamped",
    "Vector3",
    "Twist",
    "TwistWithCovariance",
    "TwistWithCovarianceStamped",
    "Accel",
    "AccelWithCovariance",
    "AccelWithCovarianceStamped",
    # nav_msgs
    "Odometry",
    "AMCLPose",
    "GoalID",
    "GoalInfo",
    "GoalStatus",
    "Nav2Status",
    # sensor_msgs
    "RegionOfInterest",
    "CameraInfo",
    "Image",
    "IMU",
    "Detection",
    "HazardDetection",
    "HazardDetectionVector",
    "NavSatStatus",
    "NavSatFix",
    "PointField",
    "PointCloud",
    "PointCloud2",
    "BatteryState",
    "LaserScan",
    "DockStatus",
    "Paths",
    # session
    "create_zenoh_config",
    "open_zenoh_session",
    # modules
    "session",
    # idl submodules
    "std_msgs",
    "status_msgs",
    "geographic_msgs",
    "geometry_msgs",
    "nav_msgs",
    "sensor_msgs",
]
