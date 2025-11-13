# stream_camera.py
import socket
import struct

import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

# Initialize ROS node
rospy.init_node("camera_stream_node")
bridge = CvBridge()

# Set up socket for streaming
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.settimeout(3600)  # Set to 1 hour

# Bind to the IP of Machine 1 and port 5000 (where it will listen)
host = "0.0.0.0"  # 0.0.0.0 allows it to accept connections from any IP
port = 5000
server_socket.bind((host, port))

# Set the server to listen for incoming connections
server_socket.listen(1)
print("Waiting for a connection...")
conn, addr = server_socket.accept()
print(f"Connection from: {addr}")


def image_callback(data):
    try:
        # Convert ROS image message to OpenCV format
        cv_image = bridge.imgmsg_to_cv2(data, "bgr8")

        # Encode the image as JPEG in MJPEG format
        _, buffer = cv2.imencode(".jpg", cv_image)

        # Send the JPEG-encoded image size followed by the image data
        size = len(buffer)
        conn.sendall(struct.pack(">L", size) + buffer.tobytes())
    except Exception as e:
        rospy.loginfo(f"Error during streaming: {e}")
        conn.close()
        server_socket.close()
        rospy.signal_shutdown("Server closed")


# Subscribe to the ROS topic for camera images
rospy.Subscriber("/camera/image_raw", Image, image_callback)

# Keep the script running
rospy.spin()
