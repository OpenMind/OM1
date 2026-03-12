"""
Simple conversation history provider.

Records user voice input and robot spoken responses,
keeps the last N rounds, and formats them for the Fuser
to inject into the LLM prompt.

"""

import json
import logging
from dataclasses import dataclass
from typing import List, Optional

from .singleton import singleton


@dataclass
class ConversationRound:
    """A single round of user input and robot response."""

    user_input: str
    robot_response: str


@singleton
class ConversationHistoryProvider:
    """
    Singleton that tracks conversation rounds between user and robot.

    Parameters
    ----------
    max_rounds : int
        Maximum number of conversation rounds to keep (default: 3).
    agent_name : str
        The robot's name for formatting (default: "Bits").
    """

    def __init__(self, max_rounds: int = 3, agent_name: str = "Bits"):
        self.max_rounds = max_rounds
        self.agent_name = agent_name
        self.rounds: List[ConversationRound] = []
        logging.info(
            f"ConversationHistoryProvider initialized: max_rounds={max_rounds}, agent={agent_name}"
        )

    def add_user_input(self, user_input: str) -> None:
        """
        Record what the user said. Creates a new round with an empty robot response.

        Parameters
        ----------
        user_input : str
            The voice input from the user.
        """
        text = user_input.strip()
        if not text:
            return
        self.rounds.append(ConversationRound(user_input=text, robot_response=""))
        logging.debug(f"ConversationHistory: recorded user input: {text[:80]}")

    def add_robot_response(self, action_value: str) -> None:
        """
        Record what the robot said. Updates the most recent round.
        Handles both plain text and JSON format like: {"response": "Hello!", "confidence": 0.9, ...}.


        Parameters
        ----------
        action_value : str
            The raw action value from the LLM output.
        """
        response = self._extract_response(action_value)
        if not response:
            return

        if self.rounds and not self.rounds[-1].robot_response:
            self.rounds[-1].robot_response = response
        else:
            # Edge case: response without a preceding user input
            self.rounds.append(
                ConversationRound(user_input="", robot_response=response)
            )

        # Truncate to max_rounds
        if len(self.rounds) > self.max_rounds:
            self.rounds = self.rounds[-self.max_rounds :]

        logging.debug(
            f"ConversationHistory: {len(self.rounds)}/{self.max_rounds} rounds"
        )

    def format(self) -> str:
        """
        Format conversation history as a string for LLM prompt injection.

        Returns
        -------
        str
            Formatted conversation history, or empty string if no history.
        """
        if not self.rounds:
            return ""

        lines = []
        for r in self.rounds:
            if r.user_input:
                lines.append(f"User: {r.user_input}")
            if r.robot_response:
                lines.append(f"{self.agent_name}: {r.robot_response}")

        if not lines:
            return ""

        return "CONVERSATION HISTORY:\n" + "\n".join(lines)

    def clear(self) -> None:
        """Clear all conversation history (e.g., on mode transition)."""
        self.rounds.clear()
        logging.debug("ConversationHistory: cleared")

    def _extract_response(self, action_value: str) -> Optional[str]:
        """
        Extract spoken text from action value.

        Parameters
        ----------
        action_value : str
            Raw action value, either plain text or JSON with "response" field.

        Returns
        -------
        Optional[str]
            The extracted spoken text, or None.
        """
        if not action_value:
            return None
        try:
            parsed = json.loads(action_value)
            if isinstance(parsed, dict) and "response" in parsed:
                return parsed["response"].strip()
        except (ValueError, TypeError):
            pass
        return action_value.strip() or None
