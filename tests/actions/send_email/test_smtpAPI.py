"""Tests for send_email action."""

from unittest.mock import patch

import pytest

from actions.send_email.connector.smtpAPI import SMTPConfig, SMTPConnector
from actions.send_email.interface import SendEmail, SendEmailInput


class TestSendEmailInput:
    """Tests for SendEmailInput dataclass."""

    def test_default_values(self):
        """Test SendEmailInput with default values."""
        input_obj = SendEmailInput()
        assert input_obj.action == ""

    def test_with_value(self):
        """Test SendEmailInput with custom value."""
        input_obj = SendEmailInput(action="Hello from robot!")
        assert input_obj.action == "Hello from robot!"

    def test_with_subject_and_body(self):
        """Test SendEmailInput with subject and body format."""
        input_obj = SendEmailInput(
            action="subject: Alert | body: Robot detected low battery."
        )
        assert "subject:" in input_obj.action
        assert "body:" in input_obj.action


class TestSendEmailInterface:
    """Tests for SendEmail interface."""

    def test_interface_creation(self):
        """Test SendEmail interface creation."""
        input_obj = SendEmailInput(action="Test message")
        output_obj = SendEmailInput(action="Test message")
        email = SendEmail(input=input_obj, output=output_obj)
        assert email.input.action == "Test message"
        assert email.output.action == "Test message"


class TestSMTPConfig:
    """Tests for SMTPConfig."""

    def test_default_values(self):
        """Test SMTPConfig default values."""
        config = SMTPConfig(
            sender_email="test@example.com",
            sender_password="password",
            recipient_email="user@example.com",
        )
        assert config.smtp_server == "smtp.gmail.com"
        assert config.smtp_port == 587
        assert config.use_tls is True
        assert config.default_subject == "Message from Robot"

    def test_custom_values(self):
        """Test SMTPConfig with custom values."""
        config = SMTPConfig(
            smtp_server="smtp.custom.com",
            smtp_port=465,
            sender_email="robot@custom.com",
            sender_password="secret",
            recipient_email="admin@custom.com",
            use_tls=False,
            default_subject="Robot Alert",
        )
        assert config.smtp_server == "smtp.custom.com"
        assert config.smtp_port == 465
        assert config.use_tls is False
        assert config.default_subject == "Robot Alert"


class TestSMTPConnector:
    """Tests for SMTPConnector."""

    def test_init_with_credentials(self):
        """Test initialization with credentials."""
        config = SMTPConfig(
            sender_email="robot@example.com",
            sender_password="password123",
            recipient_email="user@example.com",
        )
        connector = SMTPConnector(config)
        assert connector.config.sender_email == "robot@example.com"
        assert connector.config.recipient_email == "user@example.com"

    def test_init_without_sender_email(self):
        """Test initialization without sender email logs warning."""
        with patch(
            "actions.send_email.connector.smtpAPI.logging.warning"
        ) as mock_warning:
            config = SMTPConfig(
                sender_email="",
                sender_password="password",
                recipient_email="user@example.com",
            )
            SMTPConnector(config)
            mock_warning.assert_any_call("Sender email not provided in configuration")

    def test_init_without_sender_password(self):
        """Test initialization without sender password logs warning."""
        with patch(
            "actions.send_email.connector.smtpAPI.logging.warning"
        ) as mock_warning:
            config = SMTPConfig(
                sender_email="robot@example.com",
                sender_password="",
                recipient_email="user@example.com",
            )
            SMTPConnector(config)
            mock_warning.assert_any_call(
                "Sender password not provided in configuration"
            )

    def test_init_without_recipient_email(self):
        """Test initialization without recipient email logs warning."""
        with patch(
            "actions.send_email.connector.smtpAPI.logging.warning"
        ) as mock_warning:
            config = SMTPConfig(
                sender_email="robot@example.com",
                sender_password="password",
                recipient_email="",
            )
            SMTPConnector(config)
            mock_warning.assert_any_call(
                "Recipient email not provided in configuration"
            )


class TestSMTPConnectorParseEmail:
    """Tests for email content parsing."""

    @pytest.fixture
    def connector(self):
        """Create a connector for testing."""
        config = SMTPConfig(
            sender_email="robot@example.com",
            sender_password="password",
            recipient_email="user@example.com",
            default_subject="Default Subject",
        )
        return SMTPConnector(config)

    def test_parse_simple_content(self, connector):
        """Test parsing simple content uses default subject."""
        subject, body = connector._parse_email_content("Just a simple message")
        assert subject == "Default Subject"
        assert body == "Just a simple message"

    def test_parse_with_subject_and_body(self, connector):
        """Test parsing content with subject and body."""
        subject, body = connector._parse_email_content(
            "subject: Alert | body: Robot needs attention"
        )
        assert subject == "Alert"
        assert body == "Robot needs attention"

    def test_parse_with_uppercase_subject(self, connector):
        """Test parsing handles uppercase Subject."""
        subject, body = connector._parse_email_content(
            "Subject: Important | Body: Check this out"
        )
        assert subject == "Important"
        assert body == "Check this out"

    def test_parse_with_extra_spaces(self, connector):
        """Test parsing strips extra spaces."""
        subject, body = connector._parse_email_content(
            "subject:   Spaced   |  body:   Content here  "
        )
        assert subject == "Spaced"
        assert body == "Content here"


class TestSMTPConnectorConnect:
    """Tests for SMTPConnector.connect method."""

    @pytest.fixture
    def connector_with_credentials(self):
        """Create a connector with credentials."""
        config = SMTPConfig(
            sender_email="robot@example.com",
            sender_password="password123",
            recipient_email="user@example.com",
        )
        return SMTPConnector(config)

    @pytest.mark.asyncio
    async def test_connect_without_sender_credentials(self):
        """Test that connect returns early without sender credentials."""
        config = SMTPConfig(
            sender_email="",
            sender_password="",
            recipient_email="user@example.com",
        )
        connector = SMTPConnector(config)

        with patch("actions.send_email.connector.smtpAPI.logging.error") as mock_error:
            input_obj = SendEmailInput(action="Test")
            await connector.connect(input_obj)
            mock_error.assert_called_with("Email credentials not configured")

    @pytest.mark.asyncio
    async def test_connect_without_recipient(self):
        """Test that connect returns early without recipient."""
        config = SMTPConfig(
            sender_email="robot@example.com",
            sender_password="password",
            recipient_email="",
        )
        connector = SMTPConnector(config)

        with patch("actions.send_email.connector.smtpAPI.logging.error") as mock_error:
            input_obj = SendEmailInput(action="Test")
            await connector.connect(input_obj)
            mock_error.assert_called_with("Recipient email not configured")

    @pytest.mark.asyncio
    async def test_connect_with_empty_content(self, connector_with_credentials):
        """Test that connect skips empty content."""
        with patch(
            "actions.send_email.connector.smtpAPI.logging.warning"
        ) as mock_warning:
            input_obj = SendEmailInput(action="")
            await connector_with_credentials.connect(input_obj)
            mock_warning.assert_called_with("Empty email content, skipping send")

    @pytest.mark.asyncio
    async def test_connect_logs_message(self, connector_with_credentials):
        """Test that connect logs the message being sent."""
        with patch("actions.send_email.connector.smtpAPI.aiosmtplib.send") as mock_send:
            mock_send.return_value = None

            with patch(
                "actions.send_email.connector.smtpAPI.logging.info"
            ) as mock_info:
                input_obj = SendEmailInput(action="Test notification")
                await connector_with_credentials.connect(input_obj)
                mock_info.assert_any_call("SendThisToEmail: Test notification")

    @pytest.mark.asyncio
    async def test_connect_sends_email(self, connector_with_credentials):
        """Test that connect sends email via aiosmtplib."""
        with patch("actions.send_email.connector.smtpAPI.aiosmtplib.send") as mock_send:
            mock_send.return_value = None

            input_obj = SendEmailInput(action="Hello from robot!")
            await connector_with_credentials.connect(input_obj)

            mock_send.assert_called_once()
            call_kwargs = mock_send.call_args[1]
            assert call_kwargs["hostname"] == "smtp.gmail.com"
            assert call_kwargs["port"] == 587
            assert call_kwargs["username"] == "robot@example.com"
            assert call_kwargs["password"] == "password123"
            assert call_kwargs["start_tls"] is True

    @pytest.mark.asyncio
    async def test_connect_logs_success(self, connector_with_credentials):
        """Test that connect logs success after sending."""
        with patch("actions.send_email.connector.smtpAPI.aiosmtplib.send") as mock_send:
            mock_send.return_value = None

            with patch(
                "actions.send_email.connector.smtpAPI.logging.info"
            ) as mock_info:
                input_obj = SendEmailInput(action="Test message")
                await connector_with_credentials.connect(input_obj)

                success_logged = any(
                    "Email sent successfully" in str(call)
                    for call in mock_info.call_args_list
                )
                assert success_logged

    @pytest.mark.asyncio
    async def test_connect_handles_auth_error(self, connector_with_credentials):
        """Test that connect handles authentication errors."""
        import aiosmtplib

        with patch("actions.send_email.connector.smtpAPI.aiosmtplib.send") as mock_send:
            mock_send.side_effect = aiosmtplib.SMTPAuthenticationError(
                535, "Authentication failed"
            )

            with patch(
                "actions.send_email.connector.smtpAPI.logging.error"
            ) as mock_error:
                input_obj = SendEmailInput(action="Test")

                with pytest.raises(aiosmtplib.SMTPAuthenticationError):
                    await connector_with_credentials.connect(input_obj)

                assert any(
                    "authentication failed" in str(call).lower()
                    for call in mock_error.call_args_list
                )
