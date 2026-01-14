from dataclasses import dataclass

from actions.base import Interface


@dataclass
class SendEmailInput:
    """
    Input interface for the Send Email action.

    Parameters
    ----------
    action : str
        The email content to be sent. Can be formatted as:
        - Simple text: "Hello, this is a message from the robot."
        - With subject: "subject: Alert | body: The robot detected something."
        If no subject is provided, a default subject will be used.
    """

    action: str = ""


@dataclass
class SendEmail(Interface[SendEmailInput, SendEmailInput]):
    """
    This action allows the robot to send emails via SMTP.

    Effect: Sends an email with the specified content to the configured
    recipient using SMTP. The email is sent immediately and logged upon
    successful delivery. Useful for notifications, alerts, and reports.
    """

    input: SendEmailInput
    output: SendEmailInput
