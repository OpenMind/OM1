import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import json5
import zenoh

from runtime.multi_mode.config import (
    LifecycleHookType,
    ModeConfig,
    ModeSystemConfig,
    TransitionRule,
    TransitionType,
    mode_config_to_dict,
)
from zenoh_msgs import (
    ModeStatusRequest,
    ModeStatusResponse,
    String,
    open_zenoh_session,
    prepare_header,
)


@dataclass
class ModeState:
    """
    Current state of the mode system.

    Parameters
    ----------
    current_mode : str
        The current active mode
    previous_mode : Optional[str]
        The previous mode before the current one
    mode_start_time : float
        Timestamp when the current mode was activated
    transition_history : List[str]
        History of mode transitions
    last_transition_time : float
        Timestamp of the last mode transition
    user_context : Dict
        Contextual information for context-aware transitions
    """

    current_mode: str
    previous_mode: Optional[str] = None
    mode_start_time: float = field(default_factory=time.time)
    transition_history: List[str] = field(default_factory=list)
    last_transition_time: float = 0.0
    user_context: Dict = field(default_factory=dict)


class ModeManager:
    """
    Manages mode transitions and state for the OM1 system.
    """

    def __init__(self, config: ModeSystemConfig):
        """
        Initialize the mode manager.

        Parameters
        ----------
        config : ModeSystemConfig
            The mode system configuration
        """
        self.config = config
        self.state = ModeState(current_mode=config.default_mode)
        self.transition_cooldowns: Dict[str, float] = {}
        self.pending_transitions: List[TransitionRule] = []
        self._transition_callbacks: List = []
        self._main_event_loop: Optional[asyncio.AbstractEventLoop] = None
        self._transition_lock = asyncio.Lock()
        self._is_transitioning = False

        # Validate configuration
        if config.default_mode not in config.modes:
            raise ValueError(
                f"Default mode '{config.default_mode}' not found in available modes"
            )

        # Load persisted state if enabled
        if config.mode_memory_enabled:
            self._load_mode_state()

        # Start zenoh controller
        self.mode_status_request = "om/mode/request"
        self.mode_status_response = "om/mode/response"
        self.context_update_topic = "om/mode/context"

        try:
            self.session = open_zenoh_session()
            self.session.declare_subscriber(
                self.mode_status_request, self._zenoh_mode_status_request
            )
            self.session.declare_subscriber(
                self.context_update_topic, self._zenoh_context_update
            )
            self._zenoh_mode_status_response_pub = self.session.declare_publisher(
                self.mode_status_response
            )
        except Exception as e:
            logging.error(f"Error opening Zenoh client: {e}")
            self.session = None
            self.pub = None

        logging.info(
            f"Mode Manager initialized with current mode: {self.state.current_mode}"
        )

        self._create_runtime_config_file()

    def _get_runtime_config_path(self) -> str:
        """
        Get the path to the runtime config file.

        Returns
        -------
        str
            The absolute path to the runtime config file
        """
        memory_folder_path = os.path.join(
            os.path.dirname(__file__), "../../../config", "memory"
        )
        if not os.path.exists(memory_folder_path):
            os.makedirs(memory_folder_path, mode=0o755, exist_ok=True)

        return os.path.join(memory_folder_path, ".runtime.json5")

    def _create_runtime_config_file(self):
        """
        Create/update the runtime config file with the current configuration.

        This file is used for hot reload monitoring. When this file changes,
        the system will reload the configuration.
        """
        runtime_config_path = self._get_runtime_config_path()

        try:
            runtime_config = mode_config_to_dict(self.config)

            temp_file = runtime_config_path + ".tmp"
            with open(temp_file, "w") as f:
                json5.dump(runtime_config, f, indent=2)

            os.rename(temp_file, runtime_config_path)
            logging.debug(f"Runtime config file created/updated: {runtime_config_path}")

        except Exception as e:
            logging.error(f"Error creating runtime config file: {e}")

    def set_event_loop(self, loop: asyncio.AbstractEventLoop):
        """
        Set the main event loop reference for thread-safe task scheduling.

        Parameters
        ----------
        loop : asyncio.AbstractEventLoop
            The main event loop reference
        """
        self._main_event_loop = loop

    @property
    def current_mode_config(self) -> ModeConfig:
        """
        Get the configuration for the current mode.

        Returns
        -------
        ModeConfig
            The current mode configuration
        """
        return self.config.modes[self.state.current_mode]

    @property
    def current_mode_name(self) -> str:
        """
        Get the name of the current mode.

        Returns
        -------
        str
            The current mode name
        """
        return self.state.current_mode

    def add_transition_callback(self, callback: Callable):
        """
        Add a callback to be called when mode transitions occur.

        Parameters
        ----------
        callback : Callable
            The callback function to add
        """
        self._transition_callbacks.append(callback)

    def remove_transition_callback(self, callback: Callable):
        """
        Remove a transition callback.

        Parameters
        ----------
        callback : Callable
            The callback function to remove
        """
        if callback in self._transition_callbacks:
            self._transition_callbacks.remove(callback)

    async def _notify_transition_callbacks(self, from_mode: str, to_mode: str):
        """
        Notify all transition callbacks of a mode change.

        Parameters
        ----------
        from_mode : str
            The mode being transitioned from
        to_mode : str
            The mode being transitioned to
        """
        for callback in self._transition_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(from_mode, to_mode)
                else:
                    callback(from_mode, to_mode)
            except Exception as e:
                logging.error(f"Error in transition callback: {e}")

    async def check_time_based_transitions(self) -> Optional[str]:
        """
        Check if any time-based transitions should be triggered.

        Returns
        -------
        Optional[str]
            The target mode if a transition should occur, None otherwise
        """
        current_time = time.time()
        mode_duration = current_time - self.state.mode_start_time

        # Check if current mode has a timeout
        current_config = self.current_mode_config
        if (
            current_config.timeout_seconds
            and mode_duration >= current_config.timeout_seconds
        ):
            timeout_context = {
                "mode_name": self.state.current_mode,
                "timeout_seconds": current_config.timeout_seconds,
                "actual_duration": mode_duration,
                "timestamp": current_time,
            }

            try:
                await current_config.execute_lifecycle_hooks(
                    LifecycleHookType.ON_TIMEOUT, timeout_context
                )
            except Exception as e:
                logging.error(f"Error executing timeout lifecycle hooks: {e}")

            for rule in self.config.transition_rules:
                if (
                    rule.from_mode == self.state.current_mode or rule.from_mode == "*"
                ) and rule.transition_type == TransitionType.TIME_BASED:
                    if self._can_transition(rule):
                        logging.info(
                            f"Time-based transition triggered: {self.state.current_mode} -> {rule.to_mode}"
                        )
                        return rule.to_mode

        return None

    async def check_context_aware_transitions(self) -> Optional[str]:
        """
        Check if any context-aware transitions should be triggered based on current user context.

        Returns
        -------
        Optional[str]
            The target mode if a transition should occur, None otherwise
        """
        # Find matching transition rules sorted by priority (higher priority first)
        matching_rules = []
        for rule in self.config.transition_rules:
            if (
                rule.from_mode == self.state.current_mode or rule.from_mode == "*"
            ) and rule.transition_type == TransitionType.CONTEXT_AWARE:

                if self._can_transition(rule) and self._evaluate_context_conditions(
                    rule
                ):
                    matching_rules.append(rule)

        if matching_rules:
            # Sort by priority (higher first) and select the best match
            matching_rules.sort(key=lambda r: r.priority, reverse=True)
            target_rule = matching_rules[0]
            logging.info(
                f"Context-aware transition triggered: {self.state.current_mode} -> {target_rule.to_mode} "
                f"(priority: {target_rule.priority}, conditions: {target_rule.context_conditions})"
            )
            return target_rule.to_mode

        return None

    def check_input_triggered_transitions(self, input_text: str) -> Optional[str]:
        """
        Check if any input-triggered transitions should be activated.

        Parameters
        ----------
        input_text : str
            The input text to check for trigger keywords

        Returns
        -------
        Optional[str]
            The target mode if a transition should occur, None otherwise
        """
        if not input_text:
            return None

        input_lower = input_text.lower()

        # Find matching transition rules sorted by priority (higher priority first)
        matching_rules = []
        for rule in self.config.transition_rules:
            if (
                rule.from_mode == self.state.current_mode or rule.from_mode == "*"
            ) and rule.transition_type == TransitionType.INPUT_TRIGGERED:

                # Check if any trigger keywords are present
                for keyword in rule.trigger_keywords:
                    if keyword.lower() in input_lower:
                        if self._can_transition(rule):
                            matching_rules.append(rule)
                        break

        if matching_rules:
            # Sort by priority (higher first) and select the best match
            matching_rules.sort(key=lambda r: r.priority, reverse=True)
            best_rule = matching_rules[0]
            logging.info(
                f"Input-triggered transition: {self.state.current_mode} -> {best_rule.to_mode}"
            )
            logging.info(f"Triggered by keywords: {best_rule.trigger_keywords}")
            return best_rule.to_mode

        return None

    def _can_transition(self, rule: TransitionRule) -> bool:
        """
        Check if a transition rule can be executed based on cooldowns and other constraints.

        Parameters
        ----------
        rule : TransitionRule
            The transition rule to check

        Returns
        -------
        bool
            True if the transition can occur, False otherwise
        """
        current_time = time.time()

        transition_key = f"{rule.from_mode}->{rule.to_mode}"
        if transition_key in self.transition_cooldowns:
            if (
                current_time - self.transition_cooldowns[transition_key]
                < rule.cooldown_seconds
            ):
                logging.debug(f"Transition {transition_key} still in cooldown")
                return False

        if rule.to_mode not in self.config.modes:
            logging.warning(f"Target mode '{rule.to_mode}' not found in configuration")
            return False

        return True

    def _evaluate_context_conditions(self, rule: TransitionRule) -> bool:
        """
        Evaluate if the context conditions for a transition rule are met.

        Parameters
        ----------
        rule : TransitionRule
            The transition rule with context conditions to evaluate

        Returns
        -------
        bool
            True if all context conditions are satisfied, False otherwise
        """
        if not rule.context_conditions:
            return True

        user_context = self.state.user_context

        for condition_key, condition_value in rule.context_conditions.items():
            if not self._evaluate_single_condition(
                condition_key, condition_value, user_context
            ):
                logging.debug(
                    f"Context condition failed: {condition_key}={condition_value}, "
                    f"actual context: {user_context.get(condition_key)}"
                )
                return False

        return True

    def _evaluate_single_condition(
        self, key: str, expected_value, user_context: Dict
    ) -> bool:
        """
        Evaluate a single context condition.

        Parameters
        ----------
        key : str
            The context key to check
        expected_value : Any
            The expected value for the condition
        user_context : Dict
            The current user context

        Returns
        -------
        bool
            True if the condition is satisfied, False otherwise
        """
        if key not in user_context:
            return False

        actual_value = user_context[key]

        # Handle different condition types
        if isinstance(expected_value, dict):
            # Support for complex conditions like {"min": 5, "max": 10} or {"contains": "pattern"}
            if "min" in expected_value or "max" in expected_value:
                # Numeric range condition
                if not isinstance(actual_value, (int, float)):
                    return False
                if "min" in expected_value and actual_value < expected_value["min"]:
                    return False
                if "max" in expected_value and actual_value > expected_value["max"]:
                    return False
                return True

            elif "contains" in expected_value:
                # String contains condition
                if not isinstance(actual_value, str):
                    return False
                return expected_value["contains"].lower() in actual_value.lower()

            elif "one_of" in expected_value:
                # Value must be one of the specified options
                return actual_value in expected_value["one_of"]

            elif "not" in expected_value:
                # Negation condition
                return actual_value != expected_value["not"]

        elif isinstance(expected_value, list):
            # List membership condition
            return actual_value in expected_value

        else:
            # Simple equality condition
            return actual_value == expected_value

        return False

    async def request_transition(
        self, target_mode: str, reason: str = "manual"
    ) -> bool:
        """
        Request a manual transition to a specific mode.

        Parameters
        ----------
        target_mode : str
            The name of the target mode
        reason : str
            The reason for the transition

        Returns
        -------
        bool
            True if the transition was successful, False otherwise
        """
        if not self.config.allow_manual_switching and reason == "manual":
            logging.warning("Manual mode switching is disabled")
            return False

        if target_mode not in self.config.modes:
            logging.error(f"Target mode '{target_mode}' not found")
            return False

        if target_mode == self.state.current_mode:
            logging.info(f"Already in mode '{target_mode}'")
            return True

        return await self._execute_transition(target_mode, reason)

    async def _execute_transition(self, target_mode: str, reason: str) -> bool:
        """
        Execute a mode transition.

        Parameters
        ----------
        target_mode : str
            The name of the target mode
        reason : str
            The reason for the transition

        Returns
        -------
        bool
            True if the transition was successful, False otherwise
        """
        async with self._transition_lock:
            if self._is_transitioning:
                logging.debug(
                    f"Transition already in progress, skipping transition to {target_mode}"
                )
                return True

            self._is_transitioning = True

        from_mode = self.state.current_mode

        try:

            if from_mode == target_mode:
                logging.debug(
                    f"Already in target mode '{target_mode}', skipping transition"
                )
                return True

            transition_key = f"{from_mode}->{target_mode}"
            self.transition_cooldowns[transition_key] = time.time()

            from_config = self.config.modes.get(from_mode)
            to_config = self.config.modes[target_mode]

            transition_context = {
                "from_mode": from_mode,
                "to_mode": target_mode,
                "reason": reason,
                "timestamp": time.time(),
                "transition_key": transition_key,
            }

            # Execute exit hooks for the current mode
            if from_config:
                logging.debug(f"Executing exit hooks for mode: {from_mode}")
                exit_success = await from_config.execute_lifecycle_hooks(
                    LifecycleHookType.ON_EXIT, transition_context.copy()
                )
                if not exit_success:
                    logging.warning(f"Some exit hooks failed for mode: {from_mode}")

            # Execute global exit hooks
            global_exit_success = await self.config.execute_global_lifecycle_hooks(
                LifecycleHookType.ON_EXIT, transition_context.copy()
            )
            if not global_exit_success:
                logging.warning("Some global exit hooks failed")

            # Update state
            self.state.previous_mode = from_mode
            self.state.current_mode = target_mode
            self.state.mode_start_time = time.time()
            self.state.last_transition_time = time.time()
            self.state.transition_history.append(f"{from_mode}->{target_mode}:{reason}")

            if len(self.state.transition_history) > 50:
                self.state.transition_history = self.state.transition_history[-25:]

            logging.info(
                f"Mode transition: {from_mode} -> {target_mode} (reason: {reason})"
            )

            # Execute entry hooks for the new mode
            logging.debug(f"Executing entry hooks for mode: {target_mode}")
            entry_success = await to_config.execute_lifecycle_hooks(
                LifecycleHookType.ON_ENTRY, transition_context.copy()
            )
            if not entry_success:
                logging.warning(f"Some entry hooks failed for mode: {target_mode}")

            # Execute global entry hooks
            global_entry_success = await self.config.execute_global_lifecycle_hooks(
                LifecycleHookType.ON_ENTRY, transition_context.copy()
            )
            if not global_entry_success:
                logging.warning("Some global entry hooks failed")

            await self._notify_transition_callbacks(from_mode, target_mode)

            self._save_mode_state()

            return True

        except Exception as e:
            logging.error(
                f"Failed to execute transition {from_mode} -> {target_mode}: {e}"
            )
            return False
        finally:
            self._is_transitioning = False

    def get_available_transitions(self) -> List[str]:
        """
        Get a list of modes that can be transitioned to from the current mode.

        Returns
        -------
        List[str]
            List of available target mode names
        """
        available = set()

        for rule in self.config.transition_rules:
            if rule.from_mode == self.state.current_mode or rule.from_mode == "*":
                if self._can_transition(rule):
                    available.add(rule.to_mode)

        return list(available)

    def get_mode_info(self) -> Dict:
        """
        Get information about the current mode and system state.

        Returns
        -------
        Dict
            Dictionary containing mode information
        """
        current_config = self.current_mode_config
        current_time = time.time()
        mode_duration = current_time - self.state.mode_start_time

        return {
            "current_mode": self.state.current_mode,
            "display_name": current_config.display_name,
            "description": current_config.description,
            "mode_duration": mode_duration,
            "previous_mode": self.state.previous_mode,
            "available_transitions": self.get_available_transitions(),
            "all_modes": list(self.config.modes.keys()),
            "transition_history": self.state.transition_history[-5:],
            "timeout_seconds": current_config.timeout_seconds,
            "time_remaining": (
                current_config.timeout_seconds - mode_duration
                if current_config.timeout_seconds
                else None
            ),
        }

    def update_user_context(self, context: Dict):
        """
        Update the user context for context-aware transitions.

        Parameters
        ----------
        context : Dict
            The context information to update
        """
        self.state.user_context.update(context)

    def get_user_context(self) -> Dict:
        """Get the current user context."""
        return self.state.user_context.copy()

    async def process_tick(
        self, input_text: Optional[str]
    ) -> Optional[tuple[str, str]]:
        """
        Process a tick and check for any needed transitions.

        Parameters
        ----------
        input_text : Optional[str]
            Any input text to check for triggered transitions

        Returns
        -------
        Optional[tuple[str, str]]
            A tuple of (target_mode, reason) if a transition occurred, None otherwise
        """
        time_target = await self.check_time_based_transitions()
        if time_target:
            logging.info(f"Time-based transition to mode: {time_target}")
            return (time_target, "time_based")

        context_target = await self.check_context_aware_transitions()
        if context_target:
            logging.info(f"Context-aware transition to mode: {context_target}")
            return (context_target, "context_aware")

        if input_text:
            target_mode = self.check_input_triggered_transitions(input_text)
            if target_mode:
                logging.info(f"Input-triggered transition to mode: {target_mode}")
                return (target_mode, "input_triggered")

        return None

    def _zenoh_mode_status_request(self, data: zenoh.Sample):
        """
        Process incoming mode status requests via Zenoh.

        Parameters
        ----------
        data : zenoh.Sample
            The incoming Zenoh sample containing the request.
        """
        mode_status = ModeStatusRequest.deserialize(data.payload.to_bytes())
        logging.debug(f"Received mode status request: {mode_status}")

        code = mode_status.code
        request_id = mode_status.request_id
        target_mode = mode_status.mode

        # Switch to specified mode
        if code == 0 and target_mode:
            try:
                if self._main_event_loop and self._main_event_loop.is_running():
                    self._main_event_loop.call_soon_threadsafe(
                        lambda: asyncio.create_task(
                            self._handle_mode_switch_request(
                                mode_status.header.frame_id,
                                request_id.data,
                                target_mode.data,
                            )
                        )
                    )
                else:
                    logging.error("Main event loop is not set or not running")
            except Exception as e:
                logging.error(f"Error scheduling mode switch request: {e}")
            return

        # Request current mode info
        if code == 1:
            mode_status_response = ModeStatusResponse(
                header=prepare_header(mode_status.header.frame_id),
                request_id=request_id,
                code=ModeStatusResponse.Code.SUCCESS.value,
                current_mode=String(self.state.current_mode),
                message=String(json.dumps(self.get_mode_info())),
            )
            return self._zenoh_mode_status_response_pub.put(
                mode_status_response.serialize()
            )

    def _zenoh_context_update(self, data: zenoh.Sample):
        """
        Process incoming context update messages via Zenoh.

        Parameters
        ----------
        data : zenoh.Sample
            The incoming Zenoh sample containing the context update.
        """
        try:
            context_data = json.loads(data.payload.to_string())
            logging.debug(f"Received context update: {context_data}")

            if isinstance(context_data, dict):
                self.update_user_context(context_data)
                logging.info(f"Updated user context with: {context_data}")
            else:
                logging.warning(f"Invalid context data format: {context_data}")

        except (json.JSONDecodeError, Exception) as e:
            logging.error(f"Error processing context update: {e}")

    async def _handle_mode_switch_request(
        self, frame_id: str, request_id: str, target_mode: str
    ):
        """
        Handle mode switch request asynchronously and send appropriate response.

        Parameters
        ----------
        frame_id : str
            The frame ID for the response header
        request_id : str
            The request ID
        target_mode : str
            The target mode to switch to
        """
        success = await self.request_transition(target_mode, "manual")

        if success:
            mode_status_response = ModeStatusResponse(
                header=prepare_header(frame_id),
                request_id=String(request_id),
                code=ModeStatusResponse.Code.SUCCESS.value,
                current_mode=String(self.state.current_mode),
                message=String(f"Successfully switched to mode {target_mode}"),
            )
        else:
            mode_status_response = ModeStatusResponse(
                header=prepare_header(frame_id),
                request_id=String(request_id),
                code=ModeStatusResponse.Code.FAILURE.value,
                current_mode=String(self.state.current_mode),
                message=String(f"Failed to switch to mode {target_mode}"),
            )

        self._zenoh_mode_status_response_pub.put(mode_status_response.serialize())

    def _get_state_file_path(self) -> str:
        """
        Get the path to the mode state file.

        Returns
        -------
        str
            The absolute path to the state file
        """
        memory_folder_path = os.path.join(
            os.path.dirname(__file__), "../../../config", "memory"
        )
        if not os.path.exists(memory_folder_path):
            os.makedirs(memory_folder_path, mode=0o755, exist_ok=True)

        config_name = getattr(self.config, "config_name", "default")
        state_filename = (
            f"{config_name}.memory.json5"
            if config_name.startswith(".")
            else f".{config_name}.memory.json5"
        )

        return os.path.join(memory_folder_path, state_filename)

    def _load_mode_state(self):
        """
        Load the persisted mode state from file.

        If the state file exists and contains a valid last active mode,
        set it as the current mode. Otherwise, use the default mode.
        """
        state_file = self._get_state_file_path()

        try:
            with open(state_file, "r") as f:
                state_data = json.load(f)

            last_active_mode = state_data.get("last_active_mode")

            if (
                last_active_mode
                and last_active_mode in self.config.modes
                and last_active_mode != self.config.default_mode
            ):

                logging.info(f"Restoring last active mode: {last_active_mode}")
                self.state.current_mode = last_active_mode
                self.state.previous_mode = state_data.get("previous_mode")

                saved_history = state_data.get("transition_history", [])
                if saved_history:
                    self.state.transition_history.extend(saved_history)
                    if len(self.state.transition_history) > 50:
                        self.state.transition_history = self.state.transition_history[
                            -25:
                        ]

                logging.info(f"Mode state restored from {state_file}")
            else:
                logging.info(f"Using default mode: {self.config.default_mode}")

        except FileNotFoundError:
            logging.debug(f"No state file found at {state_file}, using default mode")
        except (json.JSONDecodeError, KeyError) as e:
            logging.warning(f"Invalid state file format: {e}, using default mode")
        except Exception as e:
            logging.error(f"Error loading mode state: {e}, using default mode")

    def _save_mode_state(self):
        """
        Save the current mode state to file.

        This method is called after successful mode transitions to persist
        the current state for restoration on next startup.
        """
        if not self.config.mode_memory_enabled:
            return

        state_file = self._get_state_file_path()

        try:
            os.makedirs(os.path.dirname(state_file), exist_ok=True)

            state_data = {
                "last_active_mode": self.state.current_mode,
                "previous_mode": self.state.previous_mode,
                "timestamp": time.time(),
                "transition_history": self.state.transition_history[-10:],
            }

            temp_file = state_file + ".tmp"
            with open(temp_file, "w") as f:
                json.dump(state_data, f, indent=2)

            os.rename(temp_file, state_file)
            logging.debug(f"Mode state saved to {state_file}")

        except Exception as e:
            logging.error(f"Error saving mode state: {e}")
