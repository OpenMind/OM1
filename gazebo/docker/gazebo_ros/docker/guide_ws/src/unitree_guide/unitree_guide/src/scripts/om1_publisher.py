#!/usr/bin/env python3

import signal
import time

import rospy
from custom_msgs.msg import UserInput, Vector4
from om1_msgs.msg import AI  # Assuming /bridged_movecmd uses om1_msgs/AI

# Flag to control the publishing loop
running = True


def signal_handler(sig, frame):
    """
    Handle termination signals (e.g., Ctrl+C) to enable graceful shutdown.
    """
    global running
    rospy.loginfo("Shutting down user_input_publisher gracefully...")
    running = False


def vx_vy_vyaw_to_userinput(vx, vy, vyaw):
    """
    Function to compute userValue based on vx, vy, and vyaw.
    Example formula (update this as per your requirements).
    """
    userValue_x = vy
    userValue_y = vx
    userValue_z = vyaw
    userValue_w = 0.0

    return Vector4(x=userValue_x, y=userValue_y, z=userValue_z, w=userValue_w)


def bridged_movecmd_callback(msg, publisher):
    """
    Callback function for the /bridged_movecmd subscriber.
    Converts the received message to UserInput and publishes it.
    """
    # Compute userValue based on the received vx, vy, vyaw
    userValue = vx_vy_vyaw_to_userinput(msg.vx, msg.vy, msg.vyaw)

    # Create and populate the UserInput message
    user_input_msg = UserInput()
    user_input_msg.userCmd = 1  # Example command, modify as needed
    user_input_msg.userValue = userValue

    # Log and publish the message
    rospy.loginfo(
        "Received from /bridged_movecmd: vx=%.2f, vy=%.2f, vyaw=%.2f. "
        "Publishing UserInput: userCmd=%d, userValue=(%.2f, %.2f, %.2f, %.2f)",
        msg.vx,
        msg.vy,
        msg.vyaw,
        user_input_msg.userCmd,
        user_input_msg.userValue.x,
        user_input_msg.userValue.y,
        user_input_msg.userValue.z,
        user_input_msg.userValue.w,
    )
    publisher.publish(user_input_msg)


def publish_message(pub, user_cmd, user_value, duration):
    """
    Publish a UserInput message with the specified command and value for a given duration.
    """
    start_time = time.time()
    rate = rospy.Rate(10)  # 10 Hz publishing rate

    while running and not rospy.is_shutdown() and time.time() - start_time < duration:
        # Create and populate the UserInput message
        user_input_msg = UserInput()
        user_input_msg.userCmd = user_cmd
        user_input_msg.userValue = Vector4(
            x=user_value[0], y=user_value[1], z=user_value[2], w=user_value[3]
        )

        # Log and publish the message
        rospy.loginfo(
            "Publishing UserInput: userCmd=%d, userValue=(%.2f, %.2f, %.2f, %.2f)",
            user_input_msg.userCmd,
            user_input_msg.userValue.x,
            user_input_msg.userValue.y,
            user_input_msg.userValue.z,
            user_input_msg.userValue.w,
        )
        pub.publish(user_input_msg)
        rate.sleep()


def user_input_publisher():
    # Initialize the ROS node
    rospy.init_node("user_input_publisher", anonymous=True)

    # Create a publisher for the custom UserInput message
    pub = rospy.Publisher("/user_command_topic", UserInput, queue_size=10)

    # Publish initial messages
    rospy.loginfo("Publishing initial UserInput messages...")
    publish_message(pub, user_cmd=2, user_value=(0.0, 0.0, 0.0, 0.0), duration=3)
    publish_message(pub, user_cmd=1, user_value=(0.0, 0.0, 0.0, 0.0), duration=3)

    # Subscribe to /bridged_movecmd and process messages with the callback
    rospy.Subscriber(
        "/bridged_movecmd", AI, bridged_movecmd_callback, callback_args=pub
    )

    # Spin to keep the node running and processing callbacks
    rospy.spin()


if __name__ == "__main__":
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        user_input_publisher()
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS node shutting down.")
    except Exception as e:
        rospy.logerr(f"An unexpected error occurred: {e}")
