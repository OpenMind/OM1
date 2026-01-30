import logging
from pydantic import Field

from backgrounds.base import Background, BackgroundConfig

try:
    from providers.ros2_publisher_provider import ROS2PublisherProvider
except Exception:
    ROS2PublisherProvider = None

try:
    import zenoh
except Exception:
    zenoh = None

try:
    from zenoh_msgs import String, open_zenoh_session
except Exception:
    String = None
    open_zenoh_session = None


class WelcomeStatusPublisherConfig(BackgroundConfig):
    """Configuration for welcome status publisher."""

    ros2_topic: str = Field(
        default="om/welcome",
        description="ROS2 topic name to publish to (if ROS2 available)",
    )
    zenoh_topic: str = Field(
        default="om/welcome", description="Zenoh topic name to publish to (fallback)"
    )
    message: str = Field(
        default="welcome mode", description="Status message to publish"
    )
    interval: float = Field(default=1.0, description="Seconds between publishes")


class WelcomeStatusPublisher(Background[WelcomeStatusPublisherConfig]):
    """
    Background that periodically publishes a simple status message.

    Attempts to use the ROS2 publisher provider (`ROS2PublisherProvider`) if
    available so a standard ROS2 node can subscribe. If ROS2 isn't available
    in the environment, falls back to publishing a `zenoh_msgs.String` on the
    configured Zenoh topic.
    """

    def __init__(self, config: WelcomeStatusPublisherConfig):
        super().__init__(config)

        self.ros2_provider = None
        self.zenoh_session = None
        self.zenoh_pub = None

        # Try ROS2 provider
        if ROS2PublisherProvider is not None:
            try:
                self.ros2_provider = ROS2PublisherProvider(topic=self.config.ros2_topic)
                try:
                    self.ros2_provider.start()
                except Exception:
                    # start may already have been called elsewhere
                    pass
                logging.info(
                    "WelcomeStatusPublisher: using ROS2 publisher on %s",
                    self.config.ros2_topic,
                )
            except Exception as e:
                logging.warning("Could not initialize ROS2 publisher provider: %s", e)

        # If ROS2 not available or failed, try zenoh
        if (
            self.ros2_provider is None
            and zenoh is not None
            and open_zenoh_session is not None
            and String is not None
        ):
            try:
                self.zenoh_session = open_zenoh_session()
                self.zenoh_pub = self.zenoh_session.declare_publisher(
                    self.config.zenoh_topic
                )
                logging.info(
                    "WelcomeStatusPublisher: using Zenoh publisher on %s",
                    self.config.zenoh_topic,
                )
            except Exception as e:
                logging.warning("Could not initialize Zenoh publisher: %s", e)

    def run(self) -> None:
        """Main execution loop for publisher."""
        if self.should_stop():
            return

        text = self.config.message

        # Prefer ROS2
        if self.ros2_provider is not None:
            try:
                self.ros2_provider.add_pending_message(text)
                logging.debug("Published welcome message via ROS2: %s", text)
            except Exception as e:
                logging.warning("Failed to publish via ROS2 provider: %s", e)

        # Fallback to Zenoh
        elif self.zenoh_pub is not None and String is not None:
            try:
                msg = String(text)
                # Some msg types expect .serialize()
                try:
                    payload = msg.serialize()
                except Exception:
                    # fallback to sending raw string bytes
                    payload = str(text).encode("utf-8")

                self.zenoh_pub.put(payload)
                logging.debug("Published welcome message via Zenoh: %s", text)
            except Exception as e:
                logging.warning("Failed to publish via Zenoh: %s", e)

        else:
            logging.debug("No publisher available for WelcomeStatusPublisher")

        # Sleep until next publish or until stopped
        self.sleep(self.config.interval)
