"""Standard message definitions for Zenoh communication."""

import math
import time
from dataclasses import dataclass

from pycdr2 import IdlStruct
from pycdr2.types import float32, int32, uint32


@dataclass
class Time(IdlStruct, typename="Time"):
    """Time message."""

    sec: int32
    nanosec: uint32


@dataclass
class Duration(IdlStruct, typename="Duration"):
    """Duration message."""

    sec: int32
    nanosec: uint32


@dataclass
class Header(IdlStruct, typename="Header"):
    """Standard metadata for higher-level stamped data types."""

    stamp: Time
    frame_id: str


@dataclass
class ColorRGBA(IdlStruct, typename="ColorRGBA"):
    """RGBA color message."""

    r: float32
    g: float32
    b: float32
    a: float32


@dataclass
class String(IdlStruct, typename="String"):
    """String message."""

    data: str


def prepare_header(frame_id: str = "") -> Header:
    """
    Prepare a Header with the current timestamp and a given frame ID.

    Parameters
    ----------
    frame_id : str
        The frame ID to be set in the header.
    """
    ts = time.time()
    remainder, seconds = math.modf(ts)
    timestamp = Time(sec=int32(seconds), nanosec=uint32(remainder * 1000000000))  # type: ignore
    header = Header(stamp=timestamp, frame_id=frame_id)
    return header
