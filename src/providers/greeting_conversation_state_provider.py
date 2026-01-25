import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, TypedDict

from .io_provider import IOProvider
from .singleton import singleton


class ConversationState(Enum):
    """
    States for greeting conversation.
    """

    IDLE = "idle"
    APPROACHING = "approaching"
    ENGAGING = "engaging"
    CONVERSING = "conversing"
    CONCLUDING = "concluding"
    FINISHED = "finished"


@dataclass
class ConfidenceFactors:
    """All factors that contribute to conversation completion confidence."""

    conversation_state: ConversationState  # From LLM output
    llm_confidence: float  # From LLM output
    silence_duration: float  # Seconds of silence
    speech_clarity: float  # Audio quality/recognition confidence
    person_distance: float  # Meters from robot
    conversation_duration: float  # Total conversation time
    turn_count: int  # Number of back-and-forth exchanges
    last_user_utterance_length: int  # Words in last user response


class ConfidenceBreakdown(TypedDict):
    """Breakdown of individual confidence components."""

    llm: float
    silence: float
    engagement: float
    quality: float


class ConfidenceResult(TypedDict):
    """Complete confidence calculation result."""

    overall: float
    breakdown: ConfidenceBreakdown
    factors: ConfidenceFactors


class ConfidenceCalculator:
    """Calculates overall confidence score for conversation completion."""

    def __init__(self) -> None:
        self.silence_threshold_soft = 5.0  # seconds
        self.silence_threshold_hard = 10.0
        self.min_conversation_duration = 15.0  # seconds
        self.engagement_distance_max = 2.5  # meters
        self.min_turn_count = 2  # At least some back-and-forth

        self.weights = {
            "llm": 0.35,
            "silence": 0.25,
            "engagement": 0.20,  # distance + facing
            "conversation_quality": 0.20,  # duration + turns
        }

    def calculate_completion_confidence(
        self, factors: ConfidenceFactors
    ) -> ConfidenceResult:
        """
        Calculate comprehensive confidence that conversation should end.

        Parameters
        ----------
        factors : ConfidenceFactors
            The various factors contributing to confidence.

        Returns
        -------
        ConfidenceResult
            Dictionary with overall confidence score and breakdown.
        """
        # LLM confidence
        llm_score = factors.llm_confidence

        # Silence factor
        if factors.silence_duration >= self.silence_threshold_hard:
            silence_score = 1.0
        elif factors.silence_duration >= self.silence_threshold_soft:
            silence_score = 0.5 + 0.5 * (
                (factors.silence_duration - self.silence_threshold_soft)
                / (self.silence_threshold_hard - self.silence_threshold_soft)
            )
        else:
            silence_score = factors.silence_duration / self.silence_threshold_soft * 0.5

        # 3. Engagement score (lower engagement = higher confidence to end)
        engagement_score = min(
            factors.person_distance / self.engagement_distance_max, 1.0
        )

        # 4. Conversation quality score (has it been substantial enough to end naturally?)
        # If conversation was very short, we're less confident it's really done
        duration_score = min(
            factors.conversation_duration / self.min_conversation_duration, 1.0
        )
        turn_score = min(factors.turn_count / self.min_turn_count, 1.0)

        # Short user utterances might indicate disengagement
        utterance_score = 1.0 if factors.last_user_utterance_length < 3 else 0.5

        quality_score = (duration_score + turn_score + utterance_score) / 3

        # 5. Combine weighted scores
        overall_confidence = (
            self.weights["llm"] * llm_score
            + self.weights["silence"] * silence_score
            + self.weights["engagement"] * engagement_score
            + self.weights["conversation_quality"] * quality_score
        )

        return {
            "overall": overall_confidence,
            "breakdown": {
                "llm": llm_score,
                "silence": silence_score,
                "engagement": engagement_score,
                "quality": quality_score,
            },
            "factors": factors,
        }

    def should_transition_to_concluding(
        self, confidence_result: ConfidenceResult
    ) -> bool:
        """
        Decide if we should move to concluding state.

        Parameters
        ----------
        confidence_result : ConfidenceResult
            The result from calculate_completion_confidence.

        Returns
        -------
        bool
            True if we should transition to concluding state, False otherwise.
        """
        overall = confidence_result["overall"]
        breakdown = confidence_result["breakdown"]
        llm_state = confidence_result["factors"].conversation_state

        # LLM explicitly wants to conclude with decent confidence
        llm_wants_to_conclude = llm_state in [
            ConversationState.CONCLUDING,
            ConversationState.FINISHED,
        ]

        if llm_wants_to_conclude and breakdown["llm"] >= 0.7:
            if overall >= 0.5:
                return True

        # High overall confidence
        if overall >= 0.75:
            return True

        # Medium confidence but strong signals from multiple sources
        if overall >= 0.6:
            strong_signals = sum(
                [
                    breakdown["llm"] >= 0.7,
                    breakdown["silence"] >= 0.6,
                    breakdown["engagement"] >= 0.6,
                ]
            )
            if strong_signals >= 2:
                return True

        return False

    def should_transition_to_finished(
        self, confidence_result: ConfidenceResult, time_in_concluding: float
    ) -> bool:
        """
        Decide if we should move to finished state.

        Parameters
        ----------
        confidence_result : ConfidenceResult
            The result from calculate_completion_confidence.
        time_in_concluding : float
            Time spent in concluding state in seconds.

        Returns
        -------
        bool
            True if we should transition to finished state, False otherwise.
        """
        overall = confidence_result["overall"]
        breakdown = confidence_result["breakdown"]

        # Very high confidence - immediate transition
        if overall >= 0.9:
            return True

        # High silence after we've been concluding
        if time_in_concluding >= 3.0 and breakdown["silence"] >= 0.8:
            return True

        # Person is clearly leaving
        if breakdown["engagement"] >= 0.8 and time_in_concluding >= 2.0:
            return True

        # Timeout with reasonable confidence
        if time_in_concluding >= 5.0 and overall >= 0.6:
            return True

        return False


@singleton
class GreetingConversationStateMachineProvider:
    """Manages greeting conversation state transitions based on confidence scores."""

    def __init__(self):
        self.current_state = ConversationState.IDLE
        self.previous_state = None
        self.state_entry_time = time.time()
        self.conversation_start_time = None
        self.turn_count = 0
        self.last_user_utterance = ""

        self.confidence_calculator = ConfidenceCalculator()

        self.confidence_history = []
        self.max_history = 5

        self.io_provider = IOProvider()

    def process_conversation(self, llm_output: Dict[str, Any]) -> Dict:
        """
        Process greeting conversation state based on LLM output and other factors.

        Parameters
        ----------
        llm_output : Dict[str, Any]
            Output from LLM containing confidence and other info.

        Returns
        -------
        Dict
            Current state and confidence details.
        """
        # TODO: Integration person distance, speech clarity, etc.
        person_distance = 1.5  # Placeholder

        silence_duration = time.time() - self.state_entry_time
        voice_data = self.io_provider.get_input("Voice")
        if voice_data:
            silence_duration = (
                time.time() - voice_data.timestamp
                if voice_data.timestamp
                else silence_duration
            )

        last_user_utterance_length = len(self.last_user_utterance)

        factors = ConfidenceFactors(
            conversation_state=llm_output.get(
                "conversation_state", ConversationState.CONVERSING
            ),
            llm_confidence=llm_output.get("confidence", 0.0),
            silence_duration=silence_duration,
            speech_clarity=llm_output.get("speech_clarity", 1.0),
            person_distance=person_distance,
            conversation_duration=(
                (time.time() - self.conversation_start_time)
                if self.conversation_start_time
                else 0.0
            ),
            turn_count=self.turn_count,
            last_user_utterance_length=last_user_utterance_length,
        )

        confidence_result = self.confidence_calculator.calculate_completion_confidence(
            factors
        )

        self.confidence_history.append(confidence_result["overall"])
        if len(self.confidence_history) > self.max_history:
            self.confidence_history.pop(0)

        new_state = self._determine_next_state(confidence_result)
        self.previous_state = self.current_state
        if new_state != self.current_state:
            self.current_state = new_state
            self.state_entry_time = time.time()
            logging.info(
                f"State transition: {self.previous_state.value} -> {self.current_state.value}"
            )
            logging.info(f"Confidence: {confidence_result['overall']:.2f}")
            logging.info(f"Breakdown: {confidence_result['breakdown']}")

        if voice_data:
            if self.io_provider.tick_counter == voice_data.tick:
                self.turn_count += 1
                self.last_user_utterance = voice_data.input or ""

        command = self._generate_command(confidence_result)

        return {
            "current_state": self.current_state.value,
            "confidence": confidence_result,
            "command": command,
            "time_in_state": time.time() - self.state_entry_time,
            "confidence_trend": self._get_confidence_trend(),
        }

    def _determine_next_state(
        self, confidence_result: ConfidenceResult
    ) -> ConversationState:
        """
        Determine the next conversation state based on LLM output and confidence.

        Parameters
        ----------
        confidence_result : ConfidenceResult
            The confidence calculation result.

        Returns
        -------
        ConversationState
            The next conversation state.
        """
        llm_conversation_state = confidence_result["factors"].conversation_state
        time_in_state = time.time() - self.state_entry_time

        # From CONVERSING to CONCLUDING
        if self.current_state == ConversationState.CONVERSING:
            # LLM thinks we should conclude with high confidence
            if (
                llm_conversation_state == ConversationState.CONCLUDING
                and confidence_result["breakdown"]["llm"] >= 0.7
            ):
                logging.info(
                    "LLM indicates concluding with high confidence to conclude"
                )
                return ConversationState.CONCLUDING

            # Traditional confidence-based transition
            if self.confidence_calculator.should_transition_to_concluding(
                confidence_result
            ):
                logging.info(
                    "Transitioning to CONCLUDING based on confidence to conclude"
                )
                return ConversationState.CONCLUDING

        # From CONCLUDING to FINISHED
        elif self.current_state == ConversationState.CONCLUDING:
            if self.confidence_calculator.should_transition_to_finished(
                confidence_result, time_in_state
            ):
                logging.info("Transitioning to FINISHED based on confidence to finish")
                return ConversationState.FINISHED

            # Revert to CONVERSING if:
            # 1. LLM explicitly says we're still conversing, OR
            # 2. Low overall confidence with strong engagement
            if llm_conversation_state == ConversationState.CONVERSING:
                logging.info(
                    "LLM indicates conversation continuing, reverting to conversing"
                )
                return ConversationState.CONVERSING

            if (
                confidence_result["overall"] < 0.4
                and confidence_result["breakdown"]["engagement"] < 0.3
            ):
                logging.info(
                    "Low confidence and strong engagement, reverting to conversing"
                )
                return ConversationState.CONVERSING

        # Emergency timeout - force finish after very long concluding
        if self.current_state == ConversationState.CONCLUDING and time_in_state > 15.0:
            logging.info("Emergency timeout - forcing finish")
            return ConversationState.FINISHED

        return self.current_state

    def _get_confidence_trend(self) -> str:
        """
        Analyze if confidence is increasing or decreasing.

        Returns
        -------
        str
            "increasing", "decreasing", or "stable"
        """
        if len(self.confidence_history) < 3:
            return "insufficient_data"

        recent = self.confidence_history[-3:]
        if recent[-1] > recent[0] + 0.1:
            return "increasing"
        elif recent[-1] < recent[0] - 0.1:
            return "decreasing"
        return "stable"

    def _generate_command(self, confidence_result: ConfidenceResult) -> Dict:
        """
        Generate command with confidence metadata.

        Parameters
        ----------
        confidence_result : ConfidenceResult
            The confidence calculation result.

        Returns
        -------
        Dict
            Command dictionary including confidence metadata.
        """
        base_command = self._get_base_command()

        return {
            **base_command,
            "confidence_metadata": {
                "overall": confidence_result["overall"],
                "breakdown": confidence_result["breakdown"],
                "trend": self._get_confidence_trend(),
            },
        }

    def _get_conversation_duration(self) -> float:
        """
        Get total conversation duration.

        Returns
        -------
        float
            Duration in seconds.
        """
        if self.conversation_start_time is None:
            return 0.0
        return time.time() - self.conversation_start_time

    def _get_base_command(self) -> Dict:
        """
        Get base command for robot based on current conversation state.

        Returns
        -------
        Dict
            Command dictionary for robot behavior.
        """
        if self.current_state == ConversationState.FINISHED:
            return {
                "action": "move_to_next_person",
                "track_current": False,
                "find_new_target": True,
            }
        elif self.current_state == ConversationState.CONCLUDING:
            return {
                "action": "maintain_position",
                "track_current": True,
                "find_new_target": False,
            }
        elif self.current_state == ConversationState.CONVERSING:
            return {
                "action": "stay_with_person",
                "track_current": True,
                "find_new_target": False,
            }
        else:
            return {"action": "wait", "track_current": True, "find_new_target": False}

    def update_state_without_llm(self) -> Dict:
        """
        Update conversation state based on current factors without requiring LLM input.
        This should be called periodically (e.g., in tick()) to handle timeouts and
        state transitions based on silence and other factors.

        Returns
        -------
        Dict
            Current state and confidence details.
        """
        # Get current factors without new LLM input
        person_distance = 1.5  # Placeholder - should be updated from actual sensor data

        silence_duration = time.time() - self.state_entry_time
        voice_data = self.io_provider.get_input("Voice")
        if voice_data:
            silence_duration = (
                time.time() - voice_data.timestamp
                if voice_data.timestamp
                else silence_duration
            )

        last_user_utterance_length = len(self.last_user_utterance)

        # Use previous LLM confidence or default
        factors = ConfidenceFactors(
            conversation_state=self.current_state,  # Use current state
            llm_confidence=0.5,  # Neutral confidence when no LLM input
            silence_duration=silence_duration,
            speech_clarity=1.0,  # Assume good clarity
            person_distance=person_distance,
            conversation_duration=(
                (time.time() - self.conversation_start_time)
                if self.conversation_start_time
                else 0.0
            ),
            turn_count=self.turn_count,
            last_user_utterance_length=last_user_utterance_length,
        )

        confidence_result = self.confidence_calculator.calculate_completion_confidence(
            factors
        )

        # Update confidence history
        self.confidence_history.append(confidence_result["overall"])
        if len(self.confidence_history) > self.max_history:
            self.confidence_history.pop(0)

        # Determine next state based on confidence
        new_state = self._determine_next_state(confidence_result)
        self.previous_state = self.current_state
        if new_state != self.current_state:
            logging.info(
                f"State transition: {self.current_state.value} -> {new_state.value}"
            )
            self.current_state = new_state
            self.state_entry_time = time.time()
            logging.info(f"Entered {self.current_state.value} state")

        # Generate command
        command = self._generate_command(confidence_result)

        return {
            "current_state": self.current_state.value,
            "confidence": confidence_result,
            "command": command,
            "time_in_state": time.time() - self.state_entry_time,
            "confidence_trend": self._get_confidence_trend(),
            "silence_duration": silence_duration,
        }

    def start_conversation(self):
        """
        Call when starting a new conversation.
        """
        self.conversation_start_time = time.time()
        self.turn_count = 0
        self.confidence_history = []
        self.current_state = ConversationState.CONVERSING

    def reset_state(
        self, initial_state: ConversationState = ConversationState.CONVERSING
    ):
        """
        Reset the conversation state machine to initial values.
        Typically called when transitioning from approaching to greeting.

        Parameters
        ----------
        initial_state : ConversationState, optional
            The initial state to set. Defaults to CONVERSING.
        """
        self.current_state = initial_state
        self.previous_state = None
        self.state_entry_time = time.time()
        self.conversation_start_time = time.time()
        self.turn_count = 0
        self.last_user_utterance = ""
        self.confidence_history = []
        self.current_state = ConversationState.CONVERSING
