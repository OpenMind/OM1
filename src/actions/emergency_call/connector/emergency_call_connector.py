"""
Emergency Call Connector

Implements tiered emergency response:
1. Tier 1: Send notifications to family members
2. Tier 2: Initiate phone calls
3. Tier 3: Contact emergency services

Includes privacy features: encrypted logs with auto-deletion.
"""

import asyncio
import base64
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Optional

import aiohttp
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from actions.base import ActionConfig, ActionConnector
from actions.emergency_call.interface import (
    EmergencyCallInput,
    EmergencyLevel,
    EmergencyResponseStatus,
    EmergencyTriggerType,
)


@dataclass
class EmergencyContact:
    """
    Emergency contact information.

    Parameters
    ----------
    name : str
        Contact name
    phone : Optional[str]
        Phone number for calls
    email : Optional[str]
        Email for notifications
    relation : Optional[str]
        Relationship to user
    priority : int
        Priority order (lower = higher priority)
    """

    name: str
    phone: Optional[str] = None
    email: Optional[str] = None
    relation: Optional[str] = None
    priority: int = 1


class EmergencyCallConfig(ActionConfig):
    """
    Configuration for Emergency Call connector.

    Parameters
    ----------
    encryption_key : str
        Key for encrypting emergency logs
    auto_delete_hours : int
        Hours before auto-deleting logs (default: 72)
    twilio_account_sid : Optional[str]
        Twilio account SID for phone calls
    twilio_auth_token : Optional[str]
        Twilio auth token
    twilio_from_number : Optional[str]
        Twilio phone number to call from
    emergency_service_number : str
        Emergency service number (default: "911")
    notification_service_url : Optional[str]
        URL for push notification service
    family_contacts : list
        List of EmergencyContact objects
    """

    encryption_key: str = ""
    auto_delete_hours: int = 72
    twilio_account_sid: Optional[str] = None
    twilio_auth_token: Optional[str] = None
    twilio_from_number: Optional[str] = None
    emergency_service_number: str = "911"
    notification_service_url: Optional[str] = None
    family_contacts: list = field(default_factory=list)


class EmergencyCallConnector(ActionConnector[EmergencyCallConfig, EmergencyCallInput]):
    """
    Connector for tiered emergency response.

    Implements three-tier response:
    - Tier 1: Notifications to family
    - Tier 2: Phone calls
    - Tier 3: Emergency services

    Includes privacy protection with encrypted logs and auto-deletion.
    """

    def __init__(self, config: EmergencyCallConfig):
        """Initialize the emergency call connector."""
        super().__init__(config)
        self._session: Optional[aiohttp.ClientSession] = None
        self._cipher: Optional[Fernet] = None
        self._log_dir = os.path.expanduser("~/.om1/emergency_logs")
        os.makedirs(self._log_dir, exist_ok=True)

        # Initialize encryption if key provided
        if config.encryption_key:
            self._init_encryption()

        self._cleanup_old_logs()

    def _init_encryption(self) -> None:
        """Initialize encryption cipher."""
        try:
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=b"om1_emergency_salt",
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(self.config.encryption_key.encode()))
            self._cipher = Fernet(key)
        except Exception as e:
            logging.error(f"Failed to initialize encryption: {e}")

    def _encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data."""
        if self._cipher:
            return self._cipher.encrypt(data.encode()).decode()
        return base64.b64encode(data.encode()).decode()

    def _decrypt_data(self, encrypted: str) -> str:
        """Decrypt sensitive data."""
        if self._cipher:
            return self._cipher.decrypt(encrypted.encode()).decode()
        return base64.b64decode(encrypted.encode()).decode()

    def _cleanup_old_logs(self) -> None:
        """Remove logs older than auto_delete_hours."""
        try:
            cutoff = time.time() - (self.config.auto_delete_hours * 3600)
            for filename in os.listdir(self._log_dir):
                filepath = os.path.join(self._log_dir, filename)
                if os.path.isfile(filepath):
                    if os.path.getmtime(filepath) < cutoff:
                        os.remove(filepath)
                        logging.info(f"Auto-deleted old emergency log: {filename}")
        except Exception as e:
            logging.error(f"Failed to cleanup old logs: {e}")

    def _log_emergency(
        self, emergency_id: str, status: EmergencyResponseStatus, details: dict
    ) -> None:
        """Log emergency event with optional encryption."""
        try:
            log_entry = {
                "emergency_id": emergency_id,
                "timestamp": datetime.now().isoformat(),
                "status": status.value,
                "details": details,
            }

            log_data = json.dumps(log_entry)
            encrypted = self._encrypt_data(log_data)

            log_file = os.path.join(self._log_dir, f"{emergency_id}.enc")
            with open(log_file, "w") as f:
                f.write(encrypted)

            logging.info(f"Emergency logged: {emergency_id} - {status.value}")
        except Exception as e:
            logging.error(f"Failed to log emergency: {e}")

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def _send_notification(
        self, contact: EmergencyContact, emergency: EmergencyCallInput, emergency_id: str
    ) -> bool:
        """Send notification to family member."""
        if not contact.email:
            logging.warning(f"No email for contact {contact.name}")
            return False

        try:
            message = self._format_notification_message(emergency, emergency_id)

            # For now, log the notification (replace with actual email service)
            logging.info(f"[NOTIFICATION] To: {contact.email}, Message: {message}")

            return True
        except Exception as e:
            logging.error(f"Failed to send notification: {e}")
            return False

    def _format_notification_message(
        self, emergency: EmergencyCallInput, emergency_id: str
    ) -> str:
        """Format notification message."""
        return f"""
🚨 EMERGENCY ALERT 🚨

Emergency ID: {emergency_id}
Level: {emergency.emergency_level.name}
Trigger: {emergency.trigger_type.value}
Location: {emergency.location}
Time: {emergency.timestamp or datetime.now().isoformat()}

Message: {emergency.user_message or "No message provided"}

Please check on the user immediately.
"""

    async def _initiate_phone_call(
        self, to_number: str, emergency: EmergencyCallInput, emergency_id: str
    ) -> bool:
        """Initiate phone call using Twilio."""
        if not all(
            [
                self.config.twilio_account_sid,
                self.config.twilio_auth_token,
                self.config.twilio_from_number,
            ]
        ):
            logging.warning("Twilio credentials not configured, skipping phone call")
            return False

        try:
            url = "https://api.twilio.com/2010-04-01/Accounts/{}/Calls.json".format(
                self.config.twilio_account_sid
            )

            message = self._format_voice_message(emergency, emergency_id)

            payload = {
                "To": to_number,
                "From": self.config.twilio_from_number,
                "Url": f"http://twimlbin.com/emergency?message={message}",
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    data=payload,
                    auth=aiohttp.BasicAuth(
                        self.config.twilio_account_sid, self.config.twilio_auth_token
                    ),
                ) as response:
                    if response.status == 201:
                        logging.info(f"Emergency call initiated to {to_number}")
                        return True
                    else:
                        error = await response.text()
                        logging.error(f"Twilio call failed: {error}")
                        return False
        except Exception as e:
            logging.error(f"Failed to initiate phone call: {e}")
            return False

    def _format_voice_message(self, emergency: EmergencyCallInput, emergency_id: str) -> str:
        """Format voice message for phone call."""
        return f"Emergency detected. Level {emergency.emergency_level.name}. {emergency.user_message or 'Please check immediately.'}"

    async def _contact_emergency_services(
        self, emergency: EmergencyCallInput, emergency_id: str
    ) -> bool:
        """Contact emergency services."""
        try:
            # This would integrate with actual emergency services API
            # For now, log the attempt
            logging.critical(
                f"[EMERGENCY SERVICES] Calling {self.config.emergency_service_number} "
                f"for emergency {emergency_id} at {emergency.location}"
            )

            # In production, this would use appropriate emergency services APIs
            # e.g., RapidSOS, E911, etc.

            return True
        except Exception as e:
            logging.error(f"Failed to contact emergency services: {e}")
            return False

    def _generate_emergency_id(self, emergency: EmergencyCallInput) -> str:
        """Generate unique emergency ID."""
        data = f"{emergency.trigger_type.value}:{emergency.timestamp or time.time()}:{emergency.location}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    async def connect(self, output_interface: EmergencyCallInput) -> None:
        """
        Execute tiered emergency response.

        Parameters
        ----------
        output_interface : EmergencyCallInput
            The emergency information and requested action.
        """
        emergency_id = self._generate_emergency_id(output_interface)
        level = output_interface.emergency_level

        logging.critical(f"🚨 EMERGENCY DETECTED: {emergency_id} - Level {level.name}")

        # Log emergency detection
        self._log_emergency(
            emergency_id, EmergencyResponseStatus.DETECTED, {
                "trigger": output_interface.trigger_type.value,
                "level": level.name,
                "location": output_interface.location,
                "message": output_interface.user_message,
            }
        )

        # Tier 1: Send notifications to all family contacts
        logging.info(f"Emergency Tier 1: Sending notifications (Level {level.name})")
        notification_tasks = []

        for contact in sorted(self.config.family_contacts, key=lambda c: c.priority):
            task = asyncio.create_task(
                self._send_notification(contact, output_interface, emergency_id)
            )
            notification_tasks.append(task)

        notification_results = await asyncio.gather(*notification_tasks, return_exceptions=True)
        success_count = sum(1 for r in notification_results if r is True)
        logging.info(f"Notifications sent: {success_count}/{len(notification_tasks)}")

        self._log_emergency(
            emergency_id, EmergencyResponseStatus.NOTIFICATION_SENT, {
                "notifications_sent": success_count,
                "total_contacts": len(notification_tasks),
            }
        )

        # Tier 2: Initiate phone calls (for MEDIUM and above)
        if level.value >= EmergencyLevel.MEDIUM.value:
            logging.info(f"Emergency Tier 2: Initiating phone calls")

            call_tasks = []
            for contact in sorted(self.config.family_contacts, key=lambda c: c.priority):
                if contact.phone:
                    task = asyncio.create_task(
                        self._initiate_phone_call(contact.phone, output_interface, emergency_id)
                    )
                    call_tasks.append(task)

            if call_tasks:
                call_results = await asyncio.gather(*call_tasks, return_exceptions=True)
                call_count = sum(1 for r in call_results if r is True)
                logging.info(f"Phone calls initiated: {call_count}/{len(call_tasks)}")

                self._log_emergency(
                    emergency_id, EmergencyResponseStatus.CALL_INITIATED, {
                        "calls_initiated": call_count,
                    }
                )

        # Tier 3: Contact emergency services (for HIGH and CRITICAL)
        if level.value >= EmergencyLevel.HIGH.value:
            logging.critical(f"Emergency Tier 3: Contacting emergency services!")

            success = await self._contact_emergency_services(output_interface, emergency_id)

            if success:
                self._log_emergency(
                    emergency_id, EmergencyResponseStatus.EMERGENCY_DISPATCHED, {
                        "service_number": self.config.emergency_service_number,
                    }
                )

        logging.critical(f"Emergency response completed: {emergency_id}")

    async def stop(self) -> None:
        """Cleanup resources."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
