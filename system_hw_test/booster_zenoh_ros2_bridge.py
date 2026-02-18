import json

import rclpy
import zenoh

# Import ROS 2 service type
from booster_interface.srv import RpcService
from rclpy.node import Node

# Import your custom message wrappers
# Ensure these are in your PYTHONPATH
from zenoh_msgs import BoosterApiRespMsg, RpcServiceRequest, RpcServiceResponse


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
        self.zenoh_session = zenoh.open(zenoh.Config())
        self.zenoh_key = "booster_rpc_service"

        # 3. Register Zenoh Query Responder
        # This listens for the session.get() calls from your test script
        print(f"Registering Zenoh responder on: {self.zenoh_key}")
        self.queryable = self.zenoh_session.declare_queryable(
            self.zenoh_key, self.zenoh_query_handler
        )

        print("Bridge is ready. Waiting for Zenoh requests...")

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
