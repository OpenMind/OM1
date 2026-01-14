import logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import aiosmtplib
from pydantic import Field

from actions.base import ActionConfig, ActionConnector
from actions.send_email.interface import SendEmailInput


class SMTPConfig(ActionConfig):
    """
    Configuration class for SMTP Email Connector.

    Parameters
    ----------
    smtp_server : str
        SMTP server address (e.g., smtp.gmail.com).
    smtp_port : int
        SMTP server port (default: 587 for TLS).
    sender_email : str
        Email address to send from.
    sender_password : str
        Password or app-specific password for authentication.
    recipient_email : str
        Default recipient email address.
    use_tls : bool
        Whether to use TLS encryption (default: True).
    default_subject : str
        Default email subject if not specified in message.
    """

    smtp_server: str = Field(
        default="smtp.gmail.com", description="SMTP server address"
    )
    smtp_port: int = Field(default=587, description="SMTP server port")
    sender_email: str = Field(description="Sender email address")
    sender_password: str = Field(description="Sender email password or app password")
    recipient_email: str = Field(description="Default recipient email address")
    use_tls: bool = Field(default=True, description="Use TLS encryption")
    default_subject: str = Field(
        default="Message from Robot", description="Default email subject"
    )


class SMTPConnector(ActionConnector[SMTPConfig, SendEmailInput]):
    """
    Connector for SMTP-based email sending.

    This connector sends emails using SMTP protocol. It supports TLS encryption
    and can parse subject from the message content.
    """

    def __init__(self, config: SMTPConfig):
        """
        Initialize the SMTP connector.

        Parameters
        ----------
        config : SMTPConfig
            Configuration object for the connector.
        """
        super().__init__(config)

        if not self.config.sender_email:
            logging.warning("Sender email not provided in configuration")
        if not self.config.sender_password:
            logging.warning("Sender password not provided in configuration")
        if not self.config.recipient_email:
            logging.warning("Recipient email not provided in configuration")

    def _parse_email_content(self, content: str) -> tuple[str, str]:
        """
        Parse email content to extract subject and body.

        Parameters
        ----------
        content : str
            Raw email content. Can be formatted as:
            - "subject: Subject | body: Body text"
            - "Just the body text" (uses default subject)

        Returns
        -------
        tuple[str, str]
            Tuple of (subject, body).
        """
        if "|" in content and "subject:" in content.lower():
            parts = content.split("|", 1)
            subject_part = parts[0].strip()
            body_part = parts[1].strip() if len(parts) > 1 else ""

            if subject_part.lower().startswith("subject:"):
                subject = subject_part[8:].strip()
            else:
                subject = subject_part

            if body_part.lower().startswith("body:"):
                body = body_part[5:].strip()
            else:
                body = body_part

            return subject, body
        else:
            return self.config.default_subject, content

    async def connect(self, output_interface: SendEmailInput) -> None:
        """
        Send email via SMTP.

        Parameters
        ----------
        output_interface : SendEmailInput
            The SendEmailInput interface containing the email content.
        """
        if not self.config.sender_email or not self.config.sender_password:
            logging.error("Email credentials not configured")
            return

        if not self.config.recipient_email:
            logging.error("Recipient email not configured")
            return

        try:
            email_content = output_interface.action
            if not email_content:
                logging.warning("Empty email content, skipping send")
                return

            logging.info(f"SendThisToEmail: {email_content}")

            subject, body = self._parse_email_content(email_content)

            msg = MIMEMultipart()
            msg["From"] = self.config.sender_email
            msg["To"] = self.config.recipient_email
            msg["Subject"] = subject
            msg.attach(MIMEText(body, "plain"))

            await aiosmtplib.send(
                msg,
                hostname=self.config.smtp_server,
                port=self.config.smtp_port,
                username=self.config.sender_email,
                password=self.config.sender_password,
                start_tls=self.config.use_tls,
            )

            logging.info(
                f"Email sent successfully to {self.config.recipient_email} "
                f"with subject: {subject}"
            )

        except aiosmtplib.SMTPAuthenticationError as e:
            logging.error(f"SMTP authentication failed: {str(e)}")
            raise
        except aiosmtplib.SMTPException as e:
            logging.error(f"SMTP error while sending email: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"Failed to send email: {str(e)}")
            raise
