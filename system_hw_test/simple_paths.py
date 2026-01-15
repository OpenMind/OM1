import logging
import sys
import time

sys.path.insert(0, "../src")

from zenoh_msgs import open_zenoh_session, sensor_msgs

logging.basicConfig(level=logging.INFO)


class SimplePaths:
    def __init__(self):
        self.session = open_zenoh_session()
        self.session.declare_subscriber("om/paths", self.paths_callback)

        logging.info("Zenoh is open for Paths")

    def paths_callback(self, msg):
        try:
            paths_msg = sensor_msgs.Paths.deserialize(msg.payload.to_bytes())
            msg_time = (
                paths_msg.header.stamp.sec + paths_msg.header.stamp.nanosec * 1e-9
            )
            current_time = time.time()
            latency = current_time - msg_time
            logging.debug(f"Received paths with latency: {latency:.6f} seconds")
            logging.debug(f"Received paths: {paths_msg.paths}")
        except Exception as e:
            logging.error(f"Error processing paths: {e}")


def main():
    paths_detector = SimplePaths()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Shutting down Paths detector")
        paths_detector.session.close()


if __name__ == "__main__":
    main()
