import logging
from unittest.mock import patch

from runtime.logging import (
    LoggingConfig,
    get_logging_config,
    setup_logging,
)


class TestLoggingConfig:
    def test_defaults(self):
        config = LoggingConfig()
        assert config.log_level == "INFO"
        assert config.log_to_file is False

    def test_custom_values(self):
        config = LoggingConfig(log_level="DEBUG", log_to_file=True)
        assert config.log_level == "DEBUG"
        assert config.log_to_file is True


class TestSetupLogging:
    def teardown_method(self):
        logging.getLogger().handlers.clear()

    def test_defaults(self):
        setup_logging("test_config")
        logger = logging.getLogger()
        assert logger.level == logging.INFO
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0], logging.StreamHandler)

    def test_custom_level(self):
        setup_logging("test_config", log_level="DEBUG")
        assert logging.getLogger().level == logging.DEBUG

    def test_with_config_object(self):
        config_obj = LoggingConfig(log_level="WARNING", log_to_file=True)
        setup_logging("test_config", logging_config=config_obj)
        assert logging.getLogger().level == logging.WARNING

    def test_logs_to_file(self):
        with (
            patch("runtime.logging.os.makedirs"),
            patch("runtime.logging.time.strftime", return_value="2023-01-01_12-00-00"),
        ):
            setup_logging("test_config", log_to_file=True)
            logger = logging.getLogger()
            assert len(logger.handlers) == 2
            handler_types = [type(h).__name__ for h in logger.handlers]
            assert "StreamHandler" in handler_types
            assert "FileHandler" in handler_types

    def test_clears_handlers(self):
        dummy_handler = logging.StreamHandler()
        logging.getLogger().addHandler(dummy_handler)
        setup_logging("test_config")
        logger = logging.getLogger()
        assert len([h for h in logger.handlers if h is dummy_handler]) == 0
        assert len(logger.handlers) == 1


class TestGetLoggingConfig:
    def teardown_method(self):
        logging.getLogger().handlers.clear()

    def test_default(self):
        setup_logging("test_config")
        config = get_logging_config()
        assert config.log_level == "INFO"
        assert config.log_to_file is False

    def test_with_file_handler(self):
        with (
            patch("runtime.logging.os.makedirs"),
            patch("runtime.logging.time.strftime", return_value="2023-01-01_12-00-00"),
        ):
            setup_logging("test_config", log_to_file=True)
            config = get_logging_config()
            assert config.log_level == "INFO"
            assert config.log_to_file is True

    def test_custom_level(self):
        setup_logging("test_config", log_level="ERROR")
        config = get_logging_config()
        assert config.log_level == "ERROR"
