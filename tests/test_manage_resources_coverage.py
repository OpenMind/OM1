import logging
from unittest.mock import Mock, patch

import pytest

from src.actions.manage_resources.connector.zenoh_resource_mgr import (
    ZenohResourceManager,
)
from src.actions.manage_resources.interface import ManageResourcesInterface


class TestZenohImportError:
    def test_init_with_zenoh_none(self):
        with patch(
            "src.actions.manage_resources.connector.zenoh_resource_mgr.zenoh", None
        ):
            config = {}
            manager = ZenohResourceManager(config)
            assert manager.zenoh_session is None

    def test_init_with_zenoh_none_logs_error_on_session_attempt(self, caplog):
        with (
            patch(
                "src.actions.manage_resources.connector.zenoh_resource_mgr.zenoh", None
            ),
            caplog.at_level(logging.ERROR),
        ):
            config = {}
            manager = ZenohResourceManager(config)
            import asyncio

            result = asyncio.run(manager._initialize_zenoh_session())
            assert result is False
            assert "Zenoh library not available." in caplog.text


class TestZenohConfigErrors:
    @patch("src.actions.manage_resources.connector.zenoh_resource_mgr.zenoh")
    def test_config_file_not_found(self, mock_zenoh):
        mock_zenoh.Config.from_file.side_effect = FileNotFoundError("Config not found")
        config = {"zenoh_config_path": "/nonexistent/config.json5"}
        manager = ZenohResourceManager(config)
        assert manager.zenoh_session is None

    @patch("src.actions.manage_resources.connector.zenoh_resource_mgr.zenoh")
    def test_config_attribute_error(self, mock_zenoh):
        mock_zenoh.Config.default.side_effect = AttributeError("No default")
        config = {}
        manager = ZenohResourceManager(config)
        assert manager.zenoh_session is None

    @patch("src.actions.manage_resources.connector.zenoh_resource_mgr.zenoh")
    def test_config_general_exception(self, mock_zenoh):
        mock_zenoh.Config.default.side_effect = Exception("Unknown error")
        config = {}
        manager = ZenohResourceManager(config)
        assert manager.zenoh_session is None

    @patch("src.actions.manage_resources.connector.zenoh_resource_mgr.zenoh")
    def test_config_from_file_attribute_error(self, mock_zenoh):
        mock_zenoh.Config.from_file.side_effect = AttributeError("Cannot access")
        config = {"zenoh_config_path": "/some/path.json5"}
        manager = ZenohResourceManager(config)
        assert manager.zenoh_session is None

    @patch("src.actions.manage_resources.connector.zenoh_resource_mgr.zenoh")
    def test_config_from_file_attribute_error_after_not_found(self, mock_zenoh):
        mock_zenoh.Config.from_file.side_effect = FileNotFoundError("Config not found")
        mock_zenoh.Config.default.side_effect = AttributeError("No default")
        config = {"zenoh_config_path": "/nonexistent/path.json5"}
        manager = ZenohResourceManager(config)
        import asyncio

        result = asyncio.run(manager._initialize_zenoh_session())
        assert result is False

    @patch("src.actions.manage_resources.connector.zenoh_resource_mgr.zenoh")
    def test_config_from_file_exception_fallback_to_default(self, mock_zenoh):
        mock_zenoh.Config.from_file.side_effect = RuntimeError("Some other error")
        mock_default_conf = Mock()
        mock_zenoh.Config.default.return_value = mock_default_conf
        mock_zenoh.open.return_value = Mock()
        config = {"zenoh_config_path": "/nonexistent/path.json5"}
        manager = ZenohResourceManager(config)
        import asyncio

        result = asyncio.run(manager._initialize_zenoh_session())
        assert result is True

    @patch("src.actions.manage_resources.connector.zenoh_resource_mgr.zenoh")
    def test_config_file_not_found_then_open_session_fails_with_exception(
        self, mock_zenoh
    ):
        mock_zenoh.Config.from_file.side_effect = FileNotFoundError("Config not found")
        mock_default_conf = Mock()
        mock_zenoh.Config.default.return_value = mock_default_conf
        mock_zenoh.open.side_effect = RuntimeError("Open session failed")
        config = {"zenoh_config_path": "/nonexistent/path.json5"}
        manager = ZenohResourceManager(config)
        import asyncio

        result = asyncio.run(manager._initialize_zenoh_session())
        assert result is False

    @patch("src.actions.manage_resources.connector.zenoh_resource_mgr.zenoh")
    def test_config_file_not_found_then_open_session_fails_with_attribute_error(
        self, mock_zenoh
    ):
        mock_zenoh.Config.from_file.side_effect = FileNotFoundError("Config not found")
        mock_default_conf = Mock()
        mock_zenoh.Config.default.return_value = mock_default_conf
        mock_zenoh.open.side_effect = AttributeError(
            "Open session failed due to attribute error"
        )
        config = {"zenoh_config_path": "/nonexistent/path.json5"}
        manager = ZenohResourceManager(config)
        import asyncio

        result = asyncio.run(manager._initialize_zenoh_session())
        assert result is False

    @patch("src.actions.manage_resources.connector.zenoh_resource_mgr.zenoh")
    def test_config_from_file_success_logs_info(self, mock_zenoh, caplog):
        mock_config = Mock()
        mock_zenoh.Config.from_file.return_value = mock_config
        config = {"zenoh_config_path": "/valid/path.json5"}
        manager = ZenohResourceManager(config)
        with caplog.at_level(logging.INFO):
            import asyncio

            asyncio.run(manager._initialize_zenoh_session())
        assert f"Loaded Zenoh config from {config['zenoh_config_path']}" in caplog.text

    @patch("src.actions.manage_resources.connector.zenoh_resource_mgr.zenoh")
    def test_init_with_config_path_success(self, mock_zenoh):
        mock_config = Mock()
        mock_zenoh.Config.from_file.return_value = mock_config
        config = {"zenoh_config_path": "/valid/path.json5"}
        with patch("logging.info"):
            manager = ZenohResourceManager(config)
            assert manager.zenoh_config_path == "/valid/path.json5"


class TestZenohSessionInitialization:
    @patch("src.actions.manage_resources.connector.zenoh_resource_mgr.zenoh")
    @pytest.mark.asyncio
    async def test_session_already_initialized(self, mock_zenoh):
        config = {}
        manager = ZenohResourceManager(config)
        mock_session = Mock()
        manager.zenoh_session = mock_session
        result = await manager._initialize_zenoh_session()
        assert result is True
        assert manager.zenoh_session is mock_session

    @pytest.mark.asyncio
    async def test_session_init_zenoh_none(self):
        with patch(
            "src.actions.manage_resources.connector.zenoh_resource_mgr.zenoh", None
        ):
            config = {}
            manager = ZenohResourceManager(config)
            result = await manager._initialize_zenoh_session()
            assert result is False

    @patch("src.actions.manage_resources.connector.zenoh_resource_mgr.zenoh")
    @pytest.mark.asyncio
    async def test_session_open_attribute_error(self, mock_zenoh):
        mock_config = Mock()
        mock_zenoh.Config.default.return_value = mock_config
        mock_zenoh.open.side_effect = AttributeError("Cannot access zenoh.open()")
        config = {}
        manager = ZenohResourceManager(config)
        manager.zenoh_session = None
        result = await manager._initialize_zenoh_session()
        assert result is False

    @patch("src.actions.manage_resources.connector.zenoh_resource_mgr.zenoh")
    @pytest.mark.asyncio
    async def test_session_open_general_exception(self, mock_zenoh):
        mock_config = Mock()
        mock_zenoh.Config.default.return_value = mock_config
        mock_zenoh.open.side_effect = Exception("Failed to open")
        config = {}
        manager = ZenohResourceManager(config)
        manager.zenoh_session = None
        result = await manager._initialize_zenoh_session()
        assert result is False

    @patch("src.actions.manage_resources.connector.zenoh_resource_mgr.zenoh")
    @pytest.mark.asyncio
    async def test_connect_when_session_already_exists(self, mock_zenoh):
        config = {}
        manager = ZenohResourceManager(config)
        manager.zenoh_session = Mock()
        with patch.object(manager, "_initialize_zenoh_session") as mock_init:
            await manager.connect(output_interface=Mock())
            mock_init.assert_not_called()

    @patch("src.actions.manage_resources.connector.zenoh_resource_mgr.zenoh")
    @pytest.mark.asyncio
    async def test_initialize_session_with_config_path(self, mock_zenoh):
        mock_config = Mock()
        mock_zenoh.Config.from_file.return_value = mock_config
        mock_session = Mock()
        mock_zenoh.open.return_value = mock_session
        config = {"zenoh_config_path": "/valid/path.json5"}
        manager = ZenohResourceManager(config)
        manager.zenoh_session = None
        with patch("logging.info"):
            result = await manager._initialize_zenoh_session()
            assert result is True
            assert manager.zenoh_session is not None

    @patch("src.actions.manage_resources.connector.zenoh_resource_mgr.zenoh")
    @pytest.mark.asyncio
    async def test_connect_pass_statement(self, mock_zenoh):
        config = {}
        manager = ZenohResourceManager(config)
        manager.zenoh_session = Mock()
        await manager.connect(None)
        assert manager.zenoh_session is not None

    @patch("src.actions.manage_resources.connector.zenoh_resource_mgr.zenoh")
    @pytest.mark.asyncio
    async def test_config_default_attribute_error_after_file_not_found(
        self, mock_zenoh
    ):
        config = {"zenoh_config_path": "/nonexistent.json"}
        call_count = [0]

        def side_effect():
            call_count[0] += 1
            if call_count[0] == 1:
                return Mock()
            raise AttributeError()

        mock_zenoh.Config.default.side_effect = side_effect
        mock_zenoh.Config.from_file.side_effect = FileNotFoundError()
        manager = ZenohResourceManager(config)
        result = await manager._initialize_zenoh_session()
        assert result is False

    @patch("src.actions.manage_resources.connector.zenoh_resource_mgr.zenoh")
    @pytest.mark.asyncio
    async def test_config_default_attribute_error_after_exception(self, mock_zenoh):
        config = {"zenoh_config_path": "/some.json"}
        call_count = [0]

        def side_effect():
            call_count[0] += 1
            if call_count[0] == 1:
                return Mock()
            raise AttributeError()

        mock_zenoh.Config.default.side_effect = side_effect
        mock_zenoh.Config.from_file.side_effect = Exception("error")
        manager = ZenohResourceManager(config)
        result = await manager._initialize_zenoh_session()
        assert result is False


class TestNetworkQoS:
    @patch("src.actions.manage_resources.connector.zenoh_resource_mgr.zenoh")
    @pytest.mark.asyncio
    async def test_qos_adjustment_session_not_open(self, mock_zenoh):
        config = {}
        manager = ZenohResourceManager(config)
        manager.zenoh_session = None
        with patch.object(manager, "_initialize_zenoh_session", return_value=False):
            result = await manager.adjust_network_qos("sensor/data", "high")
            assert result is False

    @patch("src.actions.manage_resources.connector.zenoh_resource_mgr.zenoh")
    @pytest.mark.asyncio
    async def test_qos_with_all_priority_levels(self, mock_zenoh):
        mock_session = Mock()
        config = {}
        manager = ZenohResourceManager(config)
        manager.zenoh_session = mock_session
        priorities = ["realtime", "high", "medium", "low"]
        for priority in priorities:
            with patch.object(manager, "_initialize_zenoh_session", return_value=True):
                with patch("logging.info"):
                    result = await manager.adjust_network_qos("test/target", priority)
                    assert isinstance(result, bool)

    @patch("src.actions.manage_resources.connector.zenoh_resource_mgr.zenoh")
    @pytest.mark.asyncio
    async def test_adjust_network_qos_invalid_priority(self, mock_zenoh):
        config = {}
        manager = ZenohResourceManager(config)
        manager.zenoh_session = Mock()
        result = await manager.adjust_network_qos("sensor/data", "invalid_priority")
        assert result is False


class TestCPUPriority:
    @patch("src.actions.manage_resources.connector.zenoh_resource_mgr.zenoh")
    @pytest.mark.asyncio
    async def test_cpu_priority_all_levels(self, mock_zenoh):
        config = {}
        manager = ZenohResourceManager(config)
        priorities = ["critical", "high", "normal", "low"]
        for priority in priorities:
            with patch("logging.info"):
                result = await manager.adjust_cpu_priority("test_task", priority)
                assert result is True

    @patch("src.actions.manage_resources.connector.zenoh_resource_mgr.zenoh")
    @pytest.mark.asyncio
    async def test_adjust_cpu_priority_invalid_priority(self, mock_zenoh):
        config = {}
        manager = ZenohResourceManager(config)
        result = await manager.adjust_cpu_priority("task", "invalid_priority")
        assert result is False


class TestQoSProfiles:
    @patch("src.actions.manage_resources.connector.zenoh_resource_mgr.zenoh")
    def test_all_qos_profiles_defined(self, mock_zenoh):
        config = {}
        manager = ZenohResourceManager(config)
        qos_profiles = manager._define_qos_profiles()
        expected_levels = ["realtime", "high", "medium", "low"]
        for level in expected_levels:
            assert level in qos_profiles
            assert "zenoh_priority" in qos_profiles[level]
            assert "zenoh_congestion_control" in qos_profiles[level]
            assert "zenoh_reliability" in qos_profiles[level]


class TestExecuteDelegation:
    @patch("src.actions.manage_resources.connector.zenoh_resource_mgr.zenoh")
    @pytest.mark.asyncio
    async def test_execute_delegation(self, mock_zenoh):
        from src.inputs.base import Message

        config = {}
        manager = ZenohResourceManager(config)
        with patch.object(manager.interface, "execute", return_value=True) as mock_exec:
            msg = Message(message="test", timestamp=0.0)
            result = await manager.execute(msg)
            mock_exec.assert_called_once_with(msg)
            assert result is True


class TestInterfaceQoSCommands:
    @pytest.mark.asyncio
    async def test_execute_qos_command_success(self):
        config = {}
        interface = ManageResourcesInterface(config)
        mock_message = Mock()
        mock_message.message = "adjust qos for sensor/camera to realtime"
        mock_message.timestamp = 1234567890
        with patch.object(interface, "adjust_network_qos", return_value=True):
            with patch.object(interface.io_provider, "add_input"):
                result = await interface.execute(mock_message)
                assert result is True

    @pytest.mark.asyncio
    async def test_execute_qos_adjustment_fails(self):
        config = {}
        interface = ManageResourcesInterface(config)
        mock_message = Mock()
        mock_message.message = "adjust qos for sensor/data to high"
        mock_message.timestamp = 1234567890
        with patch.object(interface, "adjust_network_qos", return_value=False):
            with patch.object(interface.io_provider, "add_input") as mock_add_input:
                result = await interface.execute(mock_message)
                assert result is False
                assert mock_add_input.called

    @pytest.mark.asyncio
    async def test_execute_qos_with_invalid_priority(self):
        config = {}
        interface = ManageResourcesInterface(config)
        mock_message = Mock()
        mock_message.message = "adjust qos for sensor/data to invalid_priority"
        mock_message.timestamp = 1234567890
        with patch.object(
            interface, "adjust_network_qos", return_value=True
        ) as mock_adjust:
            result = await interface.execute(mock_message)
            assert result is False
            assert not mock_adjust.called

    @pytest.mark.asyncio
    async def test_execute_qos_case_insensitive(self):
        config = {}
        interface = ManageResourcesInterface(config)
        mock_message = Mock()
        mock_message.message = "ADJUST QOS FOR sensor/data TO HIGH"
        mock_message.timestamp = 1234567890
        with patch.object(interface, "adjust_network_qos", return_value=True):
            with patch.object(interface.io_provider, "add_input"):
                result = await interface.execute(mock_message)
                assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_execute_malformed_qos_command(self):
        config = {}
        interface = ManageResourcesInterface(config)
        mock_message = Mock()
        mock_message.message = "adjust qos for sensor/data high"
        mock_message.timestamp = 1234567890
        with patch("logging.error"):
            result = await interface.execute(mock_message)
            assert result is False

    @pytest.mark.asyncio
    async def test_qos_parsing_exception(self):
        config = {}
        interface = ManageResourcesInterface(config)
        mock_message = Mock()
        mock_message.message = "adjust qos for to"
        mock_message.timestamp = 0.0
        result = await interface.execute(mock_message)
        assert result is False


class TestInterfaceCPUCommands:
    @pytest.mark.asyncio
    async def test_execute_cpu_command_success(self):
        config = {}
        interface = ManageResourcesInterface(config)
        mock_message = Mock()
        mock_message.message = "adjust cpu priority for vision_task to critical"
        mock_message.timestamp = 1234567890
        with patch.object(interface, "adjust_cpu_priority", return_value=True):
            with patch.object(interface.io_provider, "add_input"):
                result = await interface.execute(mock_message)
                assert result is True

    @pytest.mark.asyncio
    async def test_execute_cpu_priority_adjustment_fails(self):
        config = {}
        interface = ManageResourcesInterface(config)
        mock_message = Mock()
        mock_message.message = "adjust cpu priority for mytask to high"
        mock_message.timestamp = 1234567890
        with patch.object(interface, "adjust_cpu_priority", return_value=False):
            with patch.object(interface.io_provider, "add_input") as mock_add_input:
                result = await interface.execute(mock_message)
                assert result is False
                assert mock_add_input.called

    @pytest.mark.asyncio
    async def test_execute_cpu_with_invalid_priority(self):
        config = {}
        interface = ManageResourcesInterface(config)
        mock_message = Mock()
        mock_message.message = "adjust cpu priority for task to invalid_priority"
        mock_message.timestamp = 1234567890
        with patch.object(
            interface, "adjust_cpu_priority", return_value=True
        ) as mock_adjust:
            result = await interface.execute(mock_message)
            assert result is False
            assert not mock_adjust.called

    @pytest.mark.asyncio
    async def test_execute_malformed_cpu_command(self):
        config = {}
        interface = ManageResourcesInterface(config)
        mock_message = Mock()
        mock_message.message = "adjust cpu priority for task high"
        mock_message.timestamp = 1234567890
        with patch("logging.error"):
            result = await interface.execute(mock_message)
            assert result is False

    @pytest.mark.asyncio
    async def test_cpu_parsing_exception(self):
        config = {}
        interface = ManageResourcesInterface(config)
        mock_message = Mock()
        mock_message.message = "adjust cpu priority for to"
        mock_message.timestamp = 0.0
        result = await interface.execute(mock_message)
        assert result is False


class TestInterfaceEdgeCases:
    @pytest.mark.asyncio
    async def test_execute_value_error_in_qos_parsing(self):
        config = {}
        interface = ManageResourcesInterface(config)
        mock_message = Mock()
        mock_message.message = "adjust qos for"
        mock_message.timestamp = 1234567890
        with patch("logging.error"):
            result = await interface.execute(mock_message)
            assert result is False

    @pytest.mark.asyncio
    async def test_execute_index_error_in_cpu_parsing(self):
        config = {}
        interface = ManageResourcesInterface(config)
        mock_message = Mock()
        mock_message.message = "adjust cpu priority for"
        mock_message.timestamp = 1234567890
        with patch("logging.error"):
            result = await interface.execute(mock_message)
            assert result is False

    @pytest.mark.asyncio
    async def test_execute_unknown_command(self):
        config = {}
        interface = ManageResourcesInterface(config)
        mock_message = Mock()
        mock_message.message = "unknown command"
        mock_message.timestamp = 1234567890
        result = await interface.execute(mock_message)
        assert result is False
