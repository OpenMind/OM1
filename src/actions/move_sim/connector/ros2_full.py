import logging
import time

from actions.base import ActionConfig, ActionConnector
from actions.move.interface import MoveInput
from providers.ros2_publisher_provider import ROS2PublisherProvider


class ROS2Connector(ActionConnector[MoveInput]):
    """
    A connector that publishes Move messages using ROS2PublisherProvider.

    When a Move input is received, the connector publishes the message via the
    ROS2 SDK channel publisher. Optionally, it can coordinate with an ASR provider
    by registering a callback.
    """

    def __init__(self, config: ActionConfig):
        super().__init__(config)
        move_topic = getattr(self.config, "move_topic", None)

        if move_topic is None:
            move_topic = "move_topic"
            # Log the domain id being used
            logging.info(f"Move topic not provided. Using default topic: {move_topic}")

        # Initialize the ROS2 publisher provider (using a topic of your choice)
        self.publisher = ROS2PublisherProvider(topic=move_topic)
        self.publisher.start()

    async def connect(self, output_interface: MoveInput) -> None:
        new_msg = {"move": ""}

        # stub to show how to do this
        if output_interface.action == "stand still":
            new_msg["move"] = "stand still"
        elif output_interface.action == "run":
            new_msg["move"] = "run"
        elif output_interface.action == "walk forward":
            new_msg["move"] = "walk forward"
        elif output_interface.action == "walk backward":
            new_msg["move"] = "walk backward"
        elif output_interface.action == "turn left":
            new_msg["move"] = "turn left"
        elif output_interface.action == "turn right":
            new_msg["move"] = "turn right"
        elif output_interface.action == "look left":
            new_msg["move"] = "look left"
        elif output_interface.action == "look right":
            new_msg["move"] = "look right"
        elif output_interface.action == "move left":
            new_msg["move"] = "move left"
        elif output_interface.action == "move right":
            new_msg["move"] = "move right"
        else:
            logging.info(f"Other move type: {output_interface.action}")
            # raise ValueError(f"Unknown move type: {output_interface.action}")

        logging.info(f"SendThisToROS2: {new_msg}")

        # Publish the Move message using ROS2PublisherProvider.
        self.publisher.add_pending_message(new_msg)  # type: ignore

    def tick(self) -> None:
        time.sleep(0.1)
