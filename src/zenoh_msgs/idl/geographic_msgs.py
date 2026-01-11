from dataclasses import dataclass

from pycdr2 import IdlStruct
from pycdr2.types import float64

from .std_msgs import Header


@dataclass
class GeoPoint(IdlStruct, typename="GeoPoint"):
    """GeoPoint message."""

    latitude: float64
    longitude: float64
    altitude: float64


@dataclass
class GeoPointStamped(IdlStruct, typename="GeoPointStamped"):
    """GeoPointStamped message."""

    header: Header
    position: GeoPoint
