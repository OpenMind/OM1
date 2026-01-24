"""
Hooks module for OM1 runtime.

This module provides hooks for starting and stopping various services
such as Nav2, SLAM, and person following.
"""


class HookError(Exception):
    """
    Base exception for hook-related errors.

    This exception provides a safe way to propagate errors from hooks
    without exposing internal implementation details.
    """

    pass


class Nav2Error(HookError):
    """
    Exception raised when Nav2 operations fail.

    This exception is used for errors related to starting, stopping,
    or communicating with the Nav2 navigation system.
    """

    pass


class SLAMError(HookError):
    """
    Exception raised when SLAM operations fail.

    This exception is used for errors related to starting, stopping,
    or communicating with the SLAM mapping system.
    """

    pass
