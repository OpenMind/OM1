#!/usr/bin/env python

import rospy
from om1_msgs.msg import AI
from std_msgs.msg import Header


def main():
    # Initialize the ROS node
    rospy.init_node("move_publisher", anonymous=True)

    # Create a publisher to the /bridged_movecmd topic
    pub = rospy.Publisher("/bridged_movecmd", AI, queue_size=10)

    rospy.loginfo("Publisher for /bridged_movecmd is running...")
    rospy.loginfo("Enter vx, vy, and vyaw values to publish (Ctrl+C to exit).")

    try:
        while not rospy.is_shutdown():
            # Get user input for vx, vy, and vyaw
            try:
                # Prompt user to enter vx, vy, and vyaw in a single line
                user_input = input(
                    "Enter vx, vy, vyaw (separated by spaces, values between -1.0 and 1.0). E.g. 0.5 0.0 0.0 to begin walking forward: "
                )
                vx, vy, vyaw = map(float, user_input.split())
            except ValueError:
                rospy.logwarn(
                    "Invalid input. Please enter three numeric values separated by spaces."
                )
                continue

            # Create the AI message
            msg = AI()
            msg.header = Header(stamp=rospy.Time.now(), frame_id="frame1")
            msg.source = "robot1"
            msg.thought = "I am moving"
            msg.vx = vx
            msg.vy = vy
            msg.vyaw = vyaw

            # Publish the message
            pub.publish(msg)
            rospy.loginfo(f"Published: vx={vx}, vy={vy}, vyaw={vyaw}")

    except rospy.ROSInterruptException:
        rospy.loginfo("Node terminated.")


if __name__ == "__main__":
    main()
