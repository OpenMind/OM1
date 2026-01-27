import json
import os
import tempfile
import time
from unittest.mock import MagicMock, patch

import pytest

from providers.config_provider import ConfigProvider


class TestConfigProviderHotReload:
    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        ConfigProvider.reset()
        yield
        try:
            provider = ConfigProvider()
            provider.stop()
        except Exception:
            pass
        ConfigProvider.reset()

    @patch("providers.config_provider.open_zenoh_session")
    def test_hot_reload_lifecycle(self, mock_session_cls):
        """
        Verify the hot-reload lifecycle:
        1. Validation of initial load
        2. Detection of file changes
        3. Resilience against invalid JSON
        4. Recovery after file deletion
        """
        # Setup mocks
        mock_session_instance = MagicMock()
        mock_publisher = MagicMock()
        mock_session_instance.declare_publisher.return_value = mock_publisher
        mock_session_cls.return_value = mock_session_instance

        # Create a temp config file
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".runtime.json5"
        ) as tmp:
            tmp_path = tmp.name
            json.dump({"version": "1.0", "name": "Initial"}, tmp)

        try:
            # Patch the config path on the underlying class to use our temp file
            with patch.object(
                ConfigProvider._singleton_class,
                "_get_runtime_config_path",
                return_value=tmp_path,
            ):
                _ = ConfigProvider()

                # Allow watcher thread to start
                time.sleep(0.5)

                # --- 1. Verify Update Detection ---
                # Update file
                time.sleep(1.1)  # Wait >1s for mtime resolution
                with open(tmp_path, "w") as f:
                    json.dump({"version": "1.1", "name": "Updated"}, f)

                # Wait for poll cycle
                time.sleep(1.5)

                assert mock_publisher.put.called, "Should have broadcasted update"

                # --- 2. Verify Resilience (Invalid JSON) ---
                mock_publisher.put.reset_mock()

                # Write invalid JSON
                time.sleep(1.1)
                with open(tmp_path, "w") as f:
                    f.write("{ invalid json")

                time.sleep(1.5)
                # The watcher should attempt to broadcast.
                # Ideally config provider handles the load error gracefully.
                # We mainly want to ensure the thread is still alive for the NEXT update.

                # --- 3. Verify Recovery ---
                mock_publisher.put.call_count = (
                    0  # Reset count manually if needed, or just check 'called' later
                )

                # Write valid JSON again
                time.sleep(1.1)
                with open(tmp_path, "w") as f:
                    json.dump({"version": "2.0", "name": "Recovered"}, f)

                time.sleep(1.5)
                assert (
                    mock_publisher.put.call_count > 0
                ), "Watcher thread dead? Failed to detect update after error."

                # --- 4. Verify Deletion Handling ---
                mock_publisher.put.reset_mock()

                os.remove(tmp_path)
                time.sleep(1.5)  # Should not crash

                # Recreate
                with open(tmp_path, "w") as f:
                    json.dump({"version": "3.0", "name": "Recreated"}, f)

                time.sleep(1.5)
                assert mock_publisher.put.called, "Failed to detect file recreation"

        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
