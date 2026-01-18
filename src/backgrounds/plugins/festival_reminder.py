import asyncio
import logging
import time
from typing import Optional

from pydantic import Field

from backgrounds.base import Background, BackgroundConfig
from providers.context_provider import ContextProvider
from providers.festival_provider import FestivalProvider


class FestivalReminderConfig(BackgroundConfig):
    """
    Configuration for Festival Reminder Background.

    Parameters
    ----------
    check_interval_seconds : int
        Interval in seconds to check for upcoming festivals. Default is 3600 (1 hour).
    enable_reminders : bool
        Whether to enable festival reminders. Default is True.
    reminder_hour : int
        Hour of the day to send reminders (0-23). Default is 9 (9 AM).
    """

    check_interval_seconds: int = Field(
        default=3600, description="Interval in seconds to check for upcoming festivals"
    )
    enable_reminders: bool = Field(
        default=True, description="Whether to enable festival reminders"
    )
    reminder_hour: int = Field(
        default=9, description="Hour of the day to send reminders (0-23)"
    )


class FestivalReminder(Background[FestivalReminderConfig]):
    """
    Background task for checking and reminding about upcoming festivals.

    This background task periodically checks the festival calendar and updates
    the context provider with information about today's festivals and upcoming
    festivals. It can trigger reminders at specified times.

    The task runs continuously, checking for festivals at the configured interval
    and updating the system context so that the LLM can be aware of festival
    information and trigger appropriate greetings.
    """

    def __init__(self, config: FestivalReminderConfig):
        """
        Initialize Festival Reminder background task with configuration.

        Parameters
        ----------
        config : FestivalReminderConfig
            Configuration object containing check interval, reminder settings, etc.
        """
        super().__init__(config)

        self.festival_provider = FestivalProvider()
        self.context_provider = ContextProvider()
        self.last_check_time = 0
        self.last_reminder_date = None

        logging.info(
            f"Festival Reminder Background initialized with check interval: {config.check_interval_seconds}s"
        )

    def run(self) -> None:
        """
        Run the festival reminder background task.

        This method checks for festivals periodically and updates the context.
        It should be called in a loop by the runtime system.
        """
        current_time = time.time()

        # Check if it's time to check for festivals
        if current_time - self.last_check_time < self.config.check_interval_seconds:
            time.sleep(60)  # Sleep for 1 minute if not time to check yet
            return

        self.last_check_time = current_time

        if not self.config.enable_reminders:
            return

        try:
            # Get today's festivals
            today_festivals = self.festival_provider.get_today_festivals()

            # Get upcoming festivals
            upcoming_festivals = self.festival_provider.get_upcoming_festivals(days_ahead=7)

            # Get festivals that need reminders today
            reminder_festivals = self.festival_provider.get_reminder_festivals()

            # Update context with festival information
            context_update = {
                "today_festivals": [f["name"] for f in today_festivals],
                "upcoming_festivals": [
                    {
                        "name": f["name"],
                        "type": f["type"],
                        "days_until": f.get("days_until", 0),
                    }
                    for f in upcoming_festivals
                ],
                "reminder_festivals": [
                    {
                        "name": f["name"],
                        "type": f["type"],
                        "days_until": f.get("days_until", 0),
                    }
                    for f in reminder_festivals
                ],
            }

            # Only update context if there are festivals or reminders
            if (
                today_festivals
                or upcoming_festivals
                or reminder_festivals
            ):
                self.context_provider.update_context(context_update)
                logging.info(
                    f"Updated festival context: {len(today_festivals)} today, "
                    f"{len(upcoming_festivals)} upcoming, {len(reminder_festivals)} reminders"
                )

                # Log reminder information for LLM awareness
                if reminder_festivals:
                    for festival in reminder_festivals:
                        days = festival.get("days_until", 0)
                        logging.info(
                            f"Festival reminder: {festival['name']} in {days} days"
                        )

        except Exception as e:
            logging.error(f"Error in Festival Reminder background task: {e}")

        time.sleep(60)  # Sleep for 1 minute before next check

