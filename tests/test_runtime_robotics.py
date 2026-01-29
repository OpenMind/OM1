"""
Unit tests for the runtime robotics module (src/runtime/robotics.py).
Tests the load_unitree function.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# --- Setup path *before* importing from src ---
current_file_dir = Path(__file__).resolve().parent
project_root = current_file_dir.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))
# ------------------------------------------------

from src.runtime.robotics import load_unitree  # noqa: E402


class TestLoadUnitree:

    def test_load_unitree_with_valid_ethernet(self):
        unitree_ethernet = "eth0"
        mock_init = MagicMock()

        # Mock the entire chain of imports
        with patch.dict(
            "sys.modules",
            {
                "unitree": MagicMock(),
                "unitree.unitree_sdk2py": MagicMock(),
                "unitree.unitree_sdk2py.core": MagicMock(),
                "unitree.unitree_sdk2py.core.channel": MagicMock(),
            },
        ) as mock_modules:
            # Now patch the specific function inside the channel module
            with patch.object(
                mock_modules["unitree.unitree_sdk2py.core.channel"],
                "ChannelFactoryInitialize",
                mock_init,
            ):
                with patch("src.runtime.robotics.logging") as mock_log:
                    load_unitree(unitree_ethernet)

                    mock_log.info.assert_any_call(
                        f"Using {unitree_ethernet} as the Unitree Network Ethernet Adapter"
                    )
                    mock_init.assert_called_once_with(0, unitree_ethernet)
                    mock_log.info.assert_any_call("Booting Unitree and CycloneDDS")

    def test_load_unitree_with_none(self):
        # When unitree_ethernet is None, function should do nothing
        with patch("src.runtime.robotics.logging") as mock_log:
            load_unitree(None)

            # Ensure no call to ChannelFactoryInitialize since it's not executed
            mock_log.info.assert_not_called()
            mock_log.error.assert_not_called()

    def test_load_unitree_initialization_fails(self):
        unitree_ethernet = "eth0"
        mock_exception = Exception("Network error")
        mock_init = MagicMock(side_effect=mock_exception)

        # Mock the entire chain of imports
        with patch.dict(
            "sys.modules",
            {
                "unitree": MagicMock(),
                "unitree.unitree_sdk2py": MagicMock(),
                "unitree.unitree_sdk2py.core": MagicMock(),
                "unitree.unitree_sdk2py.core.channel": MagicMock(),
            },
        ) as mock_modules:
            # Now patch the specific function inside the channel module
            with patch.object(
                mock_modules["unitree.unitree_sdk2py.core.channel"],
                "ChannelFactoryInitialize",
                mock_init,
            ):
                with patch("src.runtime.robotics.logging") as mock_log:
                    load_unitree(unitree_ethernet)

                    mock_log.error.assert_called_once_with(
                        f"Failed to initialize Unitree Ethernet channel: {mock_exception}"
                    )
