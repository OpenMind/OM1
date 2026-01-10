"""Tests for lifecycle hook timeout handling improvements."""

import asyncio

import pytest
from unittest.mock import AsyncMock, patch

from runtime.multi_mode.hook import (
    LifecycleHook,
    LifecycleHookType,
    execute_lifecycle_hooks,
)


class TestHookTimeoutHandling:
    """Test cases for hook timeout handling improvements."""

    @pytest.mark.asyncio
    async def test_hook_with_default_timeout(self):
        """Test that hooks without explicit timeout use default timeout."""
        hooks = [
            LifecycleHook(
                hook_type=LifecycleHookType.ON_ENTRY,
                handler_type="message",
                handler_config={"message": "test"},
                timeout_seconds=None,  # No explicit timeout
            )
        ]

        async def slow_execution(context):
            await asyncio.sleep(35)  # Longer than default timeout (30s)
            return True

        mock_handler = AsyncMock()
        mock_handler.async_execution = True
        mock_handler.execute.side_effect = slow_execution

        with patch(
            "runtime.multi_mode.hook.create_hook_handler", return_value=mock_handler
        ):
            with patch("runtime.multi_mode.hook.logging") as mock_logging:
                result = await execute_lifecycle_hooks(
                    hooks, LifecycleHookType.ON_ENTRY
                )
                assert result is False
                # Verify timeout error was logged
                mock_logging.error.assert_called()
                error_call = mock_logging.error.call_args[0][0]
                assert "timed out" in error_call.lower()
                assert "30" in error_call  # Default timeout

    @pytest.mark.asyncio
    async def test_hook_with_explicit_timeout(self):
        """Test that hooks with explicit timeout use that timeout."""
        hooks = [
            LifecycleHook(
                hook_type=LifecycleHookType.ON_ENTRY,
                handler_type="message",
                handler_config={"message": "test"},
                timeout_seconds=0.1,  # Explicit timeout
            )
        ]

        async def slow_execution(context):
            await asyncio.sleep(1)  # Longer than explicit timeout
            return True

        mock_handler = AsyncMock()
        mock_handler.async_execution = True
        mock_handler.execute.side_effect = slow_execution

        with patch(
            "runtime.multi_mode.hook.create_hook_handler", return_value=mock_handler
        ):
            with patch("runtime.multi_mode.hook.logging") as mock_logging:
                result = await execute_lifecycle_hooks(
                    hooks, LifecycleHookType.ON_ENTRY
                )
                assert result is False
                error_call = mock_logging.error.call_args[0][0]
                assert "0.1" in error_call  # Explicit timeout

    @pytest.mark.asyncio
    async def test_sync_hook_with_timeout(self):
        """Test that sync hooks also get timeout protection."""
        hooks = [
            LifecycleHook(
                hook_type=LifecycleHookType.ON_ENTRY,
                handler_type="message",
                handler_config={"message": "test"},
                timeout_seconds=0.1,
                async_execution=False,  # Sync hook
            )
        ]

        def slow_sync_execution(context):
            import time
            time.sleep(1)  # Longer than timeout
            return True

        mock_handler = AsyncMock()
        mock_handler.async_execution = False
        mock_handler.execute.side_effect = slow_sync_execution

        with patch(
            "runtime.multi_mode.hook.create_hook_handler", return_value=mock_handler
        ):
            with patch("runtime.multi_mode.hook.logging") as mock_logging:
                result = await execute_lifecycle_hooks(
                    hooks, LifecycleHookType.ON_ENTRY
                )
                assert result is False
                # Verify timeout was applied even to sync hook
                mock_logging.error.assert_called()

    @pytest.mark.asyncio
    async def test_hook_completes_within_timeout(self):
        """Test that hooks completing within timeout succeed."""
        hooks = [
            LifecycleHook(
                hook_type=LifecycleHookType.ON_ENTRY,
                handler_type="message",
                handler_config={"message": "test"},
                timeout_seconds=5.0,
            )
        ]

        async def fast_execution(context):
            await asyncio.sleep(0.1)  # Much faster than timeout
            return True

        mock_handler = AsyncMock()
        mock_handler.async_execution = True
        mock_handler.execute.side_effect = fast_execution

        with patch(
            "runtime.multi_mode.hook.create_hook_handler", return_value=mock_handler
        ):
            result = await execute_lifecycle_hooks(hooks, LifecycleHookType.ON_ENTRY)
            assert result is True
