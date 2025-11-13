import logging
from uuid import uuid4

from actions.base import ActionConfig, ActionConnector
from actions.face.interface import FaceInput
from zenoh_msgs import AvatarFace, open_zenoh_session, prepare_header


class FaceAvatarConnector(ActionConnector[FaceInput]):
    def __init__(self, config: ActionConfig):
        """
        Initialize the FaceAvatarConnector with Zenoh publisher.

        Parameters:
        ----------
        config : ActionConfig
            Configuration parameters for the connector.
        """
        super().__init__(config)

        # Face mapping to AvatarFace enum values
        self.face_map = {
            "happy": (AvatarFace.Face.HAPPY.value, "Happy"),
            "sad": (AvatarFace.Face.SAD.value, "Sad"),
            "curious": (AvatarFace.Face.CURIOUS.value, "Curious"),
            "confused": (AvatarFace.Face.CONFUSED.value, "Confused"),
            "think": (AvatarFace.Face.THINK.value, "Think"),
            "excited": (AvatarFace.Face.EXCITED.value, "Excited"),
        }

        # Initialize Zenoh
        self.avatar_topic = "om/avatar/face"
        self.session = None
        self.avatar_publisher = None

        try:
            self.session = open_zenoh_session()
            self.avatar_publisher = self.session.declare_publisher(self.avatar_topic)
            logging.info("Zenoh Avatar publisher initialized on topic 'om/avatar/face'")
        except Exception as e:
            logging.error(f"Could not initialize Zenoh for Avatar: {e}")
            self.session = None
            self.avatar_publisher = None

        logging.info("Face system initiated with Zenoh")

    async def connect(self, output_interface: FaceInput) -> None:
        """
        Publish face to Zenoh for Avatar system.

        Parameters:
        ----------
        output_interface : FaceInput
            The face input containing the action to be performed.
        """
        action = output_interface.action
        
        if action in self.face_map:
            face_code, face_text = self.face_map[action]
            
            # Publish to Zenoh
            if self.avatar_publisher:
                try:
                    face_msg = AvatarFace(
                        header=prepare_header(str(uuid4())),
                        face=face_code,
                        face_text=face_text,
                    )
                    self.avatar_publisher.put(face_msg.serialize())
                    logging.info(f"Published Avatar face to Zenoh: {face_text}")
                except Exception as e:
                    logging.warning(f"Failed to publish Avatar face to Zenoh: {e}")
        else:
            logging.warning(f"Unknown face: {action}")

        logging.info(f"Avatar face command: {output_interface.action}")

    def stop(self):
        """
        Stop and cleanup Zenoh session.
        """
        if self.session:
            self.session.close()
            logging.info("Zenoh Avatar session closed")

    def tick(self) -> None:
        time.sleep(60)