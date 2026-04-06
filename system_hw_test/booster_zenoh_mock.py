import json
import os
import sys
import time

import zenoh

# Add src directory to path to import local zenoh_msgs if needed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from zenoh_msgs import BoosterApiRespMsg, RpcServiceRequest, RpcServiceResponse


class BoosterZenohMock:
    def __init__(self):
        print("Opening Zenoh session...")
        self.zenoh_session = zenoh.open(zenoh.Config())
        self.zenoh_key = "booster_rpc_service"

        # Register Zenoh Query Responder
        print(f"Registering Zenoh responder on: {self.zenoh_key}")
        self.queryable = self.zenoh_session.declare_queryable(self.zenoh_key, self.zenoh_query_handler)
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

            # --- Mock Response Logic ---
            # Simulate ROS 2 service success:
            response_body = json.dumps({"status": "success", "message": "Robot moved (MOCKED)"})
            api_resp = BoosterApiRespMsg(status=0, body=response_body)

            # Wrap back into the RpcServiceResponse expected by client
            zenoh_resp = RpcServiceResponse(msg=api_resp)

            # Send the reply back to the Zenoh client
            query.reply(self.zenoh_key, zenoh_resp.serialize())
            print("Reply sent back via Zenoh.")

        except Exception as e:
            print(f"Error handling query: {e}")

    def close(self):
        self.queryable.undeclare()
        self.zenoh_session.close()


def main():
    bridge = BoosterZenohMock()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        bridge.close()


if __name__ == "__main__":
    main()
