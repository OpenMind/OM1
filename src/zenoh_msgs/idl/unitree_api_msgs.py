"""Unitree API message definitions for Zenoh communication."""

from dataclasses import dataclass, field
from typing import List

from pycdr2 import IdlStruct
from pycdr2.types import int32, int64, uint8


@dataclass
class RequestIdentity(IdlStruct, typename="unitree_api::msg::dds_::RequestIdentity_"):
    """Request identity containing API routing information."""

    id: int64 = 0
    api_id: int64 = 0


@dataclass
class RequestLease(IdlStruct, typename="unitree_api::msg::dds_::RequestLease_"):
    """Request lease information."""

    id: int64 = 0


@dataclass
class RequestPolicy(IdlStruct, typename="unitree_api::msg::dds_::RequestPolicy_"):
    """Request policy settings."""

    priority: int32 = 0
    noreply: bool = False


@dataclass
class RequestHeader(IdlStruct, typename="unitree_api::msg::dds_::RequestHeader_"):
    """Request header containing identity, lease, and policy."""

    identity: RequestIdentity = field(default_factory=RequestIdentity)
    lease: RequestLease = field(default_factory=RequestLease)
    policy: RequestPolicy = field(default_factory=RequestPolicy)


@dataclass
class Request(IdlStruct, typename="unitree_api::msg::dds_::Request_"):
    """Unitree API request message for sport/action commands."""

    header: RequestHeader = field(default_factory=RequestHeader)
    parameter: str = ""
    binary: List[uint8] = field(default_factory=list)
