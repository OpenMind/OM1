import logging
from uuid import uuid4

from zenoh_msgs import AvatarFaceRequest, open_zenoh_session, prepare_header

from .singleton import singleton


@singleton
class AvatarProvider:
    """
    Singleton provider for Avatar communication via Zenoh.
    
    """

    def __init__(self):
        """
        Initialize the AvatarProvider.
        """
        self.session = None
        self.avatar_publisher = None
        self.running = False
        
        self.face_map = {
            "happy": (AvatarFaceRequest.Face.HAPPY.value, "Happy"),
            "sad": (AvatarFaceRequest.Face.SAD.value, "Sad"),
            "curious": (AvatarFaceRequest.Face.CURIOUS.value, "Curious"),
            "confused": (AvatarFaceRequest.Face.CONFUSED.value, "Confused"),
            "think": (AvatarFaceRequest.Face.THINK.value, "Think"),
            "excited": (AvatarFaceRequest.Face.EXCITED.value, "Excited"),
        }
        
        self._initialize_zenoh()

    def _initialize_zenoh(self):
        """
        Initialize Zenoh session and publisher.
        """
        try:
            self.session = open_zenoh_session()
            self.avatar_publisher = self.session.declare_publisher("om/avatar/face")
            self.running = True
            logging.info("AvatarProvider initialized with Zenoh on topic 'om/avatar/face'")
        except Exception as e:
            logging.error(f"Failed to initialize AvatarProvider Zenoh session: {e}")
            self.session = None
            self.avatar_publisher = None
            self.running = False

    def send_avatar_command(self, command: str) -> bool:
        """
        Send avatar command via Zenoh.

        Parameters
        ----------
        command : str

        Returns
        -------
        bool
            True if command was sent successfully, False otherwise
        """
        if not self.running or not self.avatar_publisher:
            logging.warning(
                f"AvatarProvider not running, cannot send command: {command}"
            )
            return False

        command = command.lower()
        
        if command not in self.face_map:
            logging.warning(f"Unknown avatar command: {command}")
            return False

        try:
            face_code, face_text = self.face_map[command]
            
            face_msg = AvatarFaceRequest(
                header=prepare_header(str(uuid4())),
                face=face_code,
                face_text=face_text,
            )
            self.avatar_publisher.put(face_msg.serialize())
            logging.info(f"AvatarProvider sent command to Zenoh: {face_text}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to send avatar command via Zenoh: {e}")
            return False

    def stop(self):
        """
        Stop the AvatarProvider and cleanup Zenoh session.
        """
        self.running = False
        if self.session:
            try:
                self.session.close()
                logging.info("AvatarProvider Zenoh session closed")
            except Exception as e:
                logging.error(f"Error closing AvatarProvider Zenoh session: {e}")
