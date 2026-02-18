import json

import rclpy
import zenoh
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
        # Note: Replace 'YourServiceType' with the actual ROS 2 .srv type
        # from your_custom_interfaces.srv import YourServiceType
        # self.cli = self.create_client(YourServiceType, self.ros2_service_name)

        # 2. Initialize Zenoh Session
        print("Opening Zenoh session...")
        self.zenoh_session = zenoh.open()
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

            print(f"Request API ID: {inner_msg.api_id}")
            print(f"Request Body: {inner_msg.body}")

            # --- ROS 2 Integration Logic ---
            # In a real scenario, you would convert inner_msg to a ROS 2 Service Request:
            # ros2_req = YourServiceType.Request()
            # ros2_req.data = inner_msg.body

            # For this example, we simulate the ROS 2 service success:
            response_body = json.dumps({"status": "success", "message": "Robot moved"})
            api_resp = BoosterApiRespMsg(status=0, body=response_body)

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
