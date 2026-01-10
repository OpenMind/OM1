"""Custom exceptions for Zenoh session management."""


class ZenohSessionError(Exception):
    """Base exception for Zenoh session errors."""

    pass


class ZenohConnectionError(ZenohSessionError):
    """Raised when unable to establish a Zenoh connection."""

    pass


class ZenohConfigurationError(ZenohSessionError):
    """Raised when Zenoh configuration is invalid."""

    pass
