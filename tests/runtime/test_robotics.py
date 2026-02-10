"""Test suite for src.runtime.robotics module."""

import sys
from unittest.mock import MagicMock, patch

from src.runtime.robotics import load_unitree


class TestLoadUnitree:
    @patch("src.runtime.robotics.logging")
    def test_load_unitree_with_valid_adapter_success(self, mock_logging):
        adapter = "eth0"
        mock_channel_module = MagicMock()
        mock_channel_init_func = MagicMock()
        mock_channel_module.ChannelFactoryInitialize = mock_channel_init_func

        with patch.dict(
            sys.modules,
            {
                "unitree": MagicMock(),
                "unitree.unitree_sdk2py": MagicMock(),
                "unitree.unitree_sdk2py.core": MagicMock(),
                "unitree.unitree_sdk2py.core.channel": mock_channel_module,
            },
        ):
            load_unitree(adapter)

        mock_logging.info.assert_any_call(
            f"Using {adapter} as the Unitree Network Ethernet Adapter"
        )
        mock_logging.info.assert_any_call("Booting Unitree and CycloneDDS")
        mock_channel_init_func.assert_called_once_with(0, adapter)

    @patch("src.runtime.robotics.logging")
    def test_load_unitree_with_valid_adapter_failure(self, mock_logging):
        adapter = "eth0"
        error_msg = "Initialization failed"
        mock_channel_module = MagicMock()
        mock_channel_init_func = MagicMock(side_effect=RuntimeError(error_msg))
        mock_channel_module.ChannelFactoryInitialize = mock_channel_init_func

        with patch.dict(
            sys.modules,
            {
                "unitree": MagicMock(),
                "unitree.unitree_sdk2py": MagicMock(),
                "unitree.unitree_sdk2py.core": MagicMock(),
                "unitree.unitree_sdk2py.core.channel": mock_channel_module,
            },
        ):
            load_unitree(adapter)

        mock_logging.info.assert_any_call(
            f"Using {adapter} as the Unitree Network Ethernet Adapter"
        )
        mock_logging.info.assert_any_call("Booting Unitree and CycloneDDS")
        mock_logging.error.assert_called_once_with(
            f"Failed to initialize Unitree Ethernet channel: {error_msg}"
        )

    @patch("src.runtime.robotics.logging")
    def test_load_unitree_with_none(self, mock_logging):
        load_unitree(None)  # type: ignore
        mock_logging.info.assert_not_called()
        mock_logging.error.assert_not_called()

    @patch("src.runtime.robotics.logging")
    def test_load_unitree_logs_boot_message_on_success(self, mock_logging):
        adapter = "wlan0"
        mock_channel_module = MagicMock()
        mock_channel_init_func = MagicMock()
        mock_channel_module.ChannelFactoryInitialize = mock_channel_init_func

        with patch.dict(
            sys.modules,
            {
                "unitree": MagicMock(),
                "unitree.unitree_sdk2py": MagicMock(),
                "unitree.unitree_sdk2py.core": MagicMock(),
                "unitree.unitree_sdk2py.core.channel": mock_channel_module,
            },
        ):
            load_unitree(adapter)

        mock_logging.info.assert_any_call(
            f"Using {adapter} as the Unitree Network Ethernet Adapter"
        )
        mock_logging.info.assert_any_call("Booting Unitree and CycloneDDS")

    @patch("src.runtime.robotics.logging")
    def test_load_unitree_logs_boot_message_on_failure(self, mock_logging):
        adapter = "eth0"
        error_msg = "Simulated failure"
        mock_channel_module = MagicMock()
        mock_channel_init_func = MagicMock(side_effect=RuntimeError(error_msg))
        mock_channel_module.ChannelFactoryInitialize = mock_channel_init_func

        with patch.dict(
            sys.modules,
            {
                "unitree": MagicMock(),
                "unitree.unitree_sdk2py": MagicMock(),
                "unitree.unitree_sdk2py.core": MagicMock(),
                "unitree.unitree_sdk2py.core.channel": mock_channel_module,
            },
        ):
            load_unitree(adapter)

        mock_logging.info.assert_any_call(
            f"Using {adapter} as the Unitree Network Ethernet Adapter"
        )
        mock_logging.info.assert_any_call("Booting Unitree and CycloneDDS")
        mock_logging.error.assert_called_once_with(
            f"Failed to initialize Unitree Ethernet channel: {error_msg}"
        )
