import rclpy

# Import messages from both packages
# from om1_interfaces.msg import AI as AIFromInterfaces
from om1_msgs.msg import AI as AIFromMsgs
from rclpy.node import Node
from std_msgs.msg import String


class BridgeNode(Node):
    def __init__(self):
        super().__init__("ai_topic_bridge")

        # Subscription to the om1_interfaces/AI topic
        self.subscription = self.create_subscription(
            String,
            "/move_topic",  # Update with your actual topic name
            self.listener_callback,
            10,  # QoS
        )
        self.subscription  # Prevent unused variable warning

        # Publisher to the om1_msgs/AI topic
        self.publisher = self.create_publisher(
            AIFromMsgs,
            "/bridged_movecmd",  # Update with your actual bridged topic name
            10,  # QoS
        )

        self.get_logger().info("AI Topic Bridge Node Initialized")

    def listener_callback(self, string_msg):
        # Manually map fields from om1_interfaces/AI to om1_msgs/AI
        msg_to_msgs = AIFromMsgs()

        match string_msg.data:
            case msg if "run" in msg:
                vx, vy, vyaw = 1.0, 0.0, 0.0
            case msg if "walk forward" in msg:
                vx, vy, vyaw = 0.3, 0.0, 0.0
            case msg if "walk backward" in msg:
                vx, vy, vyaw = -0.3, 0.0, 0.0
            case msg if "turn left" in msg:
                vx, vy, vyaw = 0.0, 0.0, -0.4
            case msg if "turn right" in msg:
                vx, vy, vyaw = 0.0, 0.0, 0.4
            case msg if "look left" in msg:
                vx, vy, vyaw = 0.0, 0.0, -0.1
            case msg if "look right" in msg:
                vx, vy, vyaw = 0.0, 0.0, 0.1
            case msg if "move left" in msg:
                vx, vy, vyaw = 0.0, -0.2, 0.0
            case msg if "move right" in msg:
                vx, vy, vyaw = 0.0, 0.2, 0.0
            case _:
                vx, vy, vyaw = 0.0, 0.0, 0.0

        # Copy float32 fields
        msg_to_msgs.vx = vx
        msg_to_msgs.vy = vy
        msg_to_msgs.vyaw = vyaw

        # Publish the converted message
        self.publisher.publish(msg_to_msgs)

        self.get_logger().info(
            f"Relayed message: source={msg_to_msgs.source}, "
            f"thought={msg_to_msgs.thought}, vx={msg_to_msgs.vx}, "
            f"vy={msg_to_msgs.vy}, vyaw={msg_to_msgs.vyaw}"
        )


def main(args=None):
    rclpy.init(args=args)

    bridge_node = BridgeNode()

    try:
        rclpy.spin(bridge_node)
    except KeyboardInterrupt:
        pass

    # Cleanup
    bridge_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
