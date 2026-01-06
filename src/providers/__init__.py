# ruff: noqa: I001
# isort: skip_file
from typing import TYPE_CHECKING

from .context_provider import ContextProvider
from .conversation_history_provider import (
    ConversationEntry,
    ConversationHistoryProvider as _ConversationHistoryProvider,
)
from .io_provider import IOProvider
from .teleops_status_provider import (
    BatteryStatus,
    CommandStatus,
    TeleopsStatus,
    TeleopsStatusProvider,
)

if TYPE_CHECKING:
    # Type stub for singleton-wrapped class

    class ConversationHistoryProvider(_ConversationHistoryProvider):  # type: ignore
        """Type stub for singleton-wrapped ConversationHistoryProvider."""

        def reset(self) -> None:  # type: ignore[override]
            """Reset the singleton instance (provided by @singleton decorator)."""
            ...

else:
    ConversationHistoryProvider = _ConversationHistoryProvider

__all__ = [
    "ContextProvider",
    "ConversationEntry",
    "ConversationHistoryProvider",
    "IOProvider",
    "TeleopsStatusProvider",
    "CommandStatus",
    "BatteryStatus",
    "TeleopsStatus",
]
