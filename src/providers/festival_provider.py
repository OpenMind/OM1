import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from .singleton import singleton


@singleton
class FestivalProvider:
    """
    Singleton provider for managing festival information and reminders.

    This provider maintains a calendar of festivals and can check if any festivals
    are approaching or occurring today. It supports both Chinese and Western festivals.
    """

    def __init__(self):
        """
        Initialize the FestivalProvider with festival calendar.
        """
        self.festivals: List[Dict] = [
            # Chinese Festivals (lunar calendar - simplified to solar for demo)
            {
                "name": "春节",
                "english_name": "Chinese New Year",
                "type": "chinese_new_year",
                "date": "2025-01-29",  # Example date, should be calculated from lunar calendar
                "reminder_days": [7, 3, 1],  # Remind 7 days, 3 days, and 1 day before
            },
            {
                "name": "中秋节",
                "english_name": "Mid-Autumn Festival",
                "type": "mid_autumn",
                "date": "2025-09-29",  # Example date
                "reminder_days": [7, 3, 1],
            },
            {
                "name": "端午节",
                "english_name": "Dragon Boat Festival",
                "type": "dragon_boat",
                "date": "2025-05-31",  # Example date
                "reminder_days": [7, 3, 1],
            },
            {
                "name": "国庆节",
                "english_name": "National Day",
                "type": "national_day",
                "date": "2025-10-01",
                "reminder_days": [7, 3, 1],
            },
            # Western Festivals
            {
                "name": "圣诞节",
                "english_name": "Christmas",
                "type": "christmas",
                "date": "2025-12-25",
                "reminder_days": [7, 3, 1],
            },
            {
                "name": "新年",
                "english_name": "New Year",
                "type": "new_year",
                "date": "2026-01-01",
                "reminder_days": [7, 3, 1],
            },
            {
                "name": "情人节",
                "english_name": "Valentine's Day",
                "type": "valentine",
                "date": "2025-02-14",
                "reminder_days": [7, 3, 1],
            },
        ]

    def get_today_festivals(self) -> List[Dict]:
        """
        Get festivals that occur today.

        Returns
        -------
        List[Dict]
            List of festivals occurring today.
        """
        today = datetime.now().date()
        today_festivals = []

        for festival in self.festivals:
            try:
                festival_date = datetime.strptime(festival["date"], "%Y-%m-%d").date()
                if festival_date == today:
                    today_festivals.append(festival)
            except ValueError:
                logging.warning(f"Invalid date format for festival: {festival['name']}")

        return today_festivals

    def get_upcoming_festivals(self, days_ahead: int = 7) -> List[Dict]:
        """
        Get festivals that are coming up within the specified number of days.

        Parameters
        ----------
        days_ahead : int
            Number of days to look ahead. Default is 7.

        Returns
        -------
        List[Dict]
            List of upcoming festivals with days until the festival.
        """
        today = datetime.now().date()
        upcoming = []

        for festival in self.festivals:
            try:
                festival_date = datetime.strptime(festival["date"], "%Y-%m-%d").date()
                days_until = (festival_date - today).days

                if 0 < days_until <= days_ahead:
                    festival_copy = festival.copy()
                    festival_copy["days_until"] = days_until
                    upcoming.append(festival_copy)
            except ValueError:
                logging.warning(f"Invalid date format for festival: {festival['name']}")

        return sorted(upcoming, key=lambda x: x["days_until"])

    def get_reminder_festivals(self) -> List[Dict]:
        """
        Get festivals that should be reminded today based on reminder_days configuration.

        Returns
        -------
        List[Dict]
            List of festivals that need reminders today.
        """
        today = datetime.now().date()
        reminder_festivals = []

        for festival in self.festivals:
            try:
                festival_date = datetime.strptime(festival["date"], "%Y-%m-%d").date()
                days_until = (festival_date - today).days

                if days_until in festival.get("reminder_days", []):
                    festival_copy = festival.copy()
                    festival_copy["days_until"] = days_until
                    reminder_festivals.append(festival_copy)
            except ValueError:
                logging.warning(f"Invalid date format for festival: {festival['name']}")

        return reminder_festivals

    def add_custom_festival(
        self,
        name: str,
        english_name: str,
        festival_type: str,
        date: str,
        reminder_days: Optional[List[int]] = None,
    ):
        """
        Add a custom festival to the calendar.

        Parameters
        ----------
        name : str
            Local name of the festival.
        english_name : str
            English name of the festival.
        festival_type : str
            Type identifier for the festival.
        date : str
            Date in "YYYY-MM-DD" format.
        reminder_days : Optional[List[int]]
            Days before the festival to send reminders. Default is [7, 3, 1].
        """
        if reminder_days is None:
            reminder_days = [7, 3, 1]

        custom_festival = {
            "name": name,
            "english_name": english_name,
            "type": festival_type,
            "date": date,
            "reminder_days": reminder_days,
        }

        self.festivals.append(custom_festival)
        logging.info(f"Added custom festival: {english_name} ({name}) on {date}")

    def get_festival_by_type(self, festival_type: str) -> Optional[Dict]:
        """
        Get festival information by type.

        Parameters
        ----------
        festival_type : str
            The festival type identifier.

        Returns
        -------
        Optional[Dict]
            Festival information if found, None otherwise.
        """
        for festival in self.festivals:
            if festival["type"] == festival_type:
                return festival
        return None

