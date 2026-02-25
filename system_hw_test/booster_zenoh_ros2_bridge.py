import json

import rclpy

# Import ROS 2 service type
from booster_interface.srv import RpcService
from rclpy.node import Node
from rosidl_runtime_py.utilities import get_message

# Import your custom message wrappers
# Ensure these are in your PYTHONPATH
from zenoh_msgs import (
    BoosterApiRespMsg,
    RpcServiceRequest,
    RpcServiceResponse,
    open_zenoh_session,
)
from zenoh_msgs import sensor_msgs as zenoh_sensor_msgs
from zenoh_msgs import std_msgs as zenoh_std_msgs


class BoosterZenohBridge(Node):
    def __init__(self):
        super().__init__("booster_zenoh_bridge")

        # 1. Initialize ROS 2 Service Client
        # This bridge acts as a CLIENT to the real ROS 2 service
        self.ros2_service_name = "/booster_rpc_service"
        self.cli = self.create_client(RpcService, self.ros2_service_name)

        # Wait for service to be available
        print(f"Waiting for ROS 2 service: {self.ros2_service_name}")
        while not self.cli.wait_for_service(timeout_sec=1.0):
            print("Service not available, waiting...")
        print(f"ROS 2 service {self.ros2_service_name} is ready")

        # 2. Initialize Zenoh Session
        print("Opening Zenoh session...")
        self.zenoh_session = open_zenoh_session()
        self.zenoh_key = "booster_rpc_service"

        # 2b. Bridge ROS2 /om/paths -> Zenoh om/paths (for SimplePathsProvider)
        self.ros2_paths_topic = "/om/paths"
        self.zenoh_paths_key = "om/paths"
        self._paths_sub = None

        # 3. Register Zenoh Query Responder
        # This listens for the session.get() calls from your test script
        print(f"Registering Zenoh responder on: {self.zenoh_key}")
        self.queryable = self.zenoh_session.declare_queryable(
            self.zenoh_key, self.zenoh_query_handler
        )

        # Try to subscribe immediately; if /om/paths isn't available yet, retry.
        self._try_subscribe_paths()
        self.create_timer(1.0, self._try_subscribe_paths)

        print("Bridge is ready. Waiting for Zenoh requests...")

    def _try_subscribe_paths(self):
        if self._paths_sub is not None:
            return

        topics = dict(self.get_topic_names_and_types())
        types = topics.get(self.ros2_paths_topic)
        type_str = types[0] if types else None
        if not type_str:
            return

        try:
            msg_type = get_message(type_str)
        except Exception as e:
            print(
                f"Cannot resolve ROS2 message type for {self.ros2_paths_topic}: {type_str} ({e})"
            )
            return

        print(
            f"Subscribing to ROS2 {self.ros2_paths_topic} [{type_str}] -> Zenoh {self.zenoh_paths_key}"
        )
        self._paths_sub = self.create_subscription(
            msg_type,
            self.ros2_paths_topic,
            self._ros2_paths_callback,
            10,
        )

    def _ros2_paths_callback(self, msg):
        try:
            header = getattr(msg, "header", None)
            frame_id = getattr(header, "frame_id", "") if header else ""

            stamp = getattr(header, "stamp", None) if header else None
            sec = int(getattr(stamp, "sec", 0)) if stamp else 0
            nanosec = int(getattr(stamp, "nanosec", 0)) if stamp else 0

            z_header = zenoh_std_msgs.Header(
                stamp=zenoh_std_msgs.Time(sec=sec, nanosec=nanosec),
                frame_id=frame_id,
            )

            paths = list(getattr(msg, "paths", []) or [])
            blocked_by_obstacle_idx = list(
                getattr(msg, "blocked_by_obstacle_idx", []) or []
            )
            blocked_by_hazard_idx = list(
                getattr(msg, "blocked_by_hazard_idx", []) or []
            )

            z_paths = zenoh_sensor_msgs.Paths(
                header=z_header,
                paths=paths,
                blocked_by_obstacle_idx=blocked_by_obstacle_idx,
                blocked_by_hazard_idx=blocked_by_hazard_idx,
            )

            self.zenoh_session.put(self.zenoh_paths_key, z_paths.serialize())
        except Exception as e:
            print(f"Error bridging {self.ros2_paths_topic}: {e}")

    def zenoh_query_handler(self, query):
        """Callback when Zenoh client calls session.get()"""
        print(f"Received Zenoh query on {query.selector}")

        payload = query.payload.to_bytes()
        try:
            # Deserialize the Zenoh request
            zenoh_req = RpcServiceRequest.deserialize(payload)
            inner_msg = zenoh_req.msg  # This is the BoosterApiReqMsg

            # Convert Zenoh request to ROS 2 service request
            ros2_req = RpcService.Request()
            ros2_req.msg.api_id = inner_msg.api_id
            ros2_req.msg.body = inner_msg.body

            # Call the ROS 2 service synchronously
            print("Calling ROS 2 service...")
            future = self.cli.call_async(ros2_req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)

            if future.result() is not None:
                ros2_resp = future.result()
                print(
                    f"ROS 2 Response - Status: {ros2_resp.msg.status}, Body: {ros2_resp.msg.body}"
                )

                # Convert ROS 2 response to Zenoh response
                api_resp = BoosterApiRespMsg(
                    status=ros2_resp.msg.status, body=ros2_resp.msg.body
                )
            else:
                print("ROS 2 service call failed or timed out")
                error_body = json.dumps(
                    {"status": "error", "message": "ROS 2 service call failed"}
                )
                api_resp = BoosterApiRespMsg(status=-1, body=error_body)

            # Wrap back into the RpcServiceResponse expected by your client
            zenoh_resp = RpcServiceResponse(msg=api_resp)

            # Send the reply back to the Zenoh client
            query.reply(self.zenoh_key, zenoh_resp.serialize())
            print("Reply sent back via Zenoh.")

        except Exception as e:
            print(f"Error handling query: {e}")

    def destroy_node(self):
        self.queryable.undeclare()
        self.zenoh_session.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    bridge = BoosterZenohBridge()
    try:
        rclpy.spin(bridge)
    except KeyboardInterrupt:
        pass
    finally:
        bridge.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
