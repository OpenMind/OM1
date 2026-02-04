from abc import ABC, abstractmethod
from typing import Any, Dict

from inputs.base import Message
from providers.io_provider import IOProvider


class ManageResourcesInterface(ABC):
    """
    Interface for managing system resources dynamically based on runtime needs.
    This could involve adjusting network QoS, CPU scheduling priorities, etc.
    Currently focused on Zenoh-based network resource management.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.io_provider = IOProvider()

    @abstractmethod
    async def adjust_network_qos(
        self,
        target: str,
        priority: str,
        reliability: str = "reliable",
        durability: str = "volatile",
    ) -> bool:
        """
        Adjusts Quality of Service parameters for a specific target/resource over the network.

        Args:
            target (str): The Zenoh key expression or resource identifier to apply QoS to.
                          Example: "sensor/lidar/data", "actuator/arm/command"
            priority (str): Priority level. Options: "realtime", "high", "medium", "low".
            reliability (str): Reliability level. Options: "reliable", "best_effort". Default: "reliable".
            durability (str): Durability level. Options: "volatile", "transient". Default: "volatile".

        Returns
        -------
            bool: True if adjustment was successful, False otherwise.

        """
        pass

    @abstractmethod
    async def adjust_cpu_priority(self, task_name: str, priority: str) -> bool:
        """
        Adjusts the CPU scheduling priority for a specific task/service.

        Args:
            task_name (str): Name or identifier of the task/service.
            priority (str): Priority level. Options: "critical", "high", "normal", "low".

        Returns
        -------
            bool: True if adjustment was successful, False otherwise.

        """
        pass

    async def execute(self, message: Message) -> bool:
        """
        Parses the message to determine the desired resource adjustment and executes it.

        Args:
            message (Message): The message containing resource adjustment command.

        Returns
        -------
            bool: True if execution was successful, False otherwise.

        """
        command_text = message.message.lower().strip()

        if "adjust qos for" in command_text and "to" in command_text:
            try:
                target_start = command_text.find("for") + 4
                target_end = command_text.find(" ", target_start)
                if target_end == -1:
                    target_end = len(command_text)
                target = command_text[target_start:target_end]

                priority_start = command_text.find("to") + 3
                priority_end = command_text.find(" ", priority_start)
                if priority_end == -1:
                    priority_end = len(command_text)
                priority = command_text[priority_start:priority_end]

                valid_priorities = ["realtime", "high", "medium", "low"]
                if priority in valid_priorities:
                    success = await self.adjust_network_qos(target, priority)
                    if success:
                        print(
                            f"[ManageResources] Successfully adjusted QoS for {target} to {priority}."
                        )
                        self.io_provider.add_input(
                            "Resource Manager",
                            f"Adjusted QoS for {target} to {priority}.",
                            message.timestamp,
                        )
                        return True
                    else:
                        print(f"[ManageResources] Failed to adjust QoS for {target}.")
                        self.io_provider.add_input(
                            "Resource Manager",
                            f"Failed to adjust QoS for {target}.",
                            message.timestamp,
                        )
                        return False
            except (ValueError, IndexError) as e:
                print(
                    f"[ManageResources] Could not parse QoS command: {command_text}. Error: {e}"
                )
                return False

        elif "adjust cpu priority for" in command_text and "to" in command_text:
            try:
                task_start = command_text.find("for") + 4
                task_end = command_text.find(" ", task_start)
                if task_end == -1:
                    task_end = len(command_text)
                task_name = command_text[task_start:task_end]

                priority_start = command_text.find("to") + 3
                priority_end = command_text.find(" ", priority_start)
                if priority_end == -1:
                    priority_end = len(command_text)
                priority = command_text[priority_start:priority_end]

                valid_priorities = ["critical", "high", "normal", "low"]
                if priority in valid_priorities:
                    success = await self.adjust_cpu_priority(task_name, priority)
                    if success:
                        print(
                            f"[ManageResources] Successfully adjusted CPU priority for {task_name} to {priority}."
                        )
                        self.io_provider.add_input(
                            "Resource Manager",
                            f"Adjusted CPU priority for {task_name} to {priority}.",
                            message.timestamp,
                        )
                        return True
                    else:
                        print(
                            f"[ManageResources] Failed to adjust CPU priority for {task_name}."
                        )
                        self.io_provider.add_input(
                            "Resource Manager",
                            f"Failed to adjust CPU priority for {task_name}.",
                            message.timestamp,
                        )
                        return False
            except (ValueError, IndexError) as e:
                print(
                    f"[ManageResources] Could not parse CPU priority command: {command_text}. Error: {e}"
                )
                return False

        else:
            print(f"[ManageResources] Unknown command: {command_text}")
            return False

        return False
