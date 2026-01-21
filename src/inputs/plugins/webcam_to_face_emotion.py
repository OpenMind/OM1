"""
Code example is from:
https://github.com/manish-9245/Facial-Emotion-Recognition-using-OpenCV-and-Deepface
Thank you @manish-9245.
"""

import asyncio
import logging
import random
import time
from typing import Optional

import cv2
from deepface import DeepFace

from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput
from providers.io_provider import IOProvider


def check_webcam():
    """
    Checks if a webcam is available and returns True if found, False otherwise.
    """
    cap = cv2.VideoCapture(0)  # 0 is the default camera index
    if not cap.isOpened():
        logging.info("No webcam found")
        return False
    logging.info("Found cam(0)")
    return True


class FaceEmotionCapture(FuserInput[SensorConfig, Optional[cv2.typing.MatLike]]):
    """
    Real-time facial emotion recognition using webcam input.

    Uses OpenCV for face detection and DeepFace for emotion analysis.
    Processes video frames to detect faces and classify emotions.
    """

    def __init__(self, config: SensorConfig):
        """
        Initialize FaceEmotionCapture instance.
        """
        super().__init__(config)

        # Track IO
        self.io_provider = IOProvider()

        # Load face cascade classifier
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"  # type: ignore
        )

        self.have_cam = check_webcam()

        # Start capturing video, if we have a webcam
        self.cap = None
        if self.have_cam:
            self.cap = cv2.VideoCapture(0)

        # Initialize emotion label
        self.emotion = ""

        # Messages buffer
        self.messages: list[Message] = []

    async def _poll(self) -> Optional[cv2.typing.MatLike]:
        """
        Capture frame from webcam.

        Returns
        -------
        cv2.typing.MatLike
            Captured video frame
        """
        await asyncio.sleep(0.5)

        # Capture a frame every 500 ms
        if self.have_cam and self.cap is not None:
            ret, frame = self.cap.read()
            return frame

    async def _raw_to_text(
        self, raw_input: Optional[cv2.typing.MatLike]
    ) -> Optional[Message]:
        """
        Process video frame for emotion detection.

        Parameters
        ----------
        raw_input : Optional[cv2.typing.MatLike]
            Captured video frame

        Returns
        -------
        Optional[Message]
            Message containing detected emotion or status
        """
        if not self.have_cam:
            # simulate a model response
            emotions = ["happy", "angry", "sad", "bored"]
            random_emotion = random.choice(emotions)
            message = f"I see a person. Their emotion is {random_emotion}."
            return Message(timestamp=time.time(), message=message)

        frame = raw_input

        # If no frame is captured, return None
        if frame is None:
            return None

        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Convert grayscale frame to RGB format
        rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

        # Detect faces in the frame
        faces = self.face_cascade.detectMultiScale(
            gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        for x, y, w, h in faces:
            # Extract the face ROI (Region of Interest)
            face_roi = rgb_frame[y : y + h, x : x + w]

            # Perform emotion analysis on the face ROI
            result = DeepFace.analyze(
                face_roi, actions=["emotion"], enforce_detection=False
            )

            # Determine the dominant emotion
            if isinstance(result, list) and len(result) > 0:
                emotion_data = result[0]
                if isinstance(emotion_data, dict):
                    self.emotion = emotion_data.get("dominant_emotion", "")

        if self.emotion == "":
            message = "I do not see anyone, so I can't estimate their emotion."
        else:
            message = f"I see a person. Their emotion is {self.emotion}."

        logging.info(f"EmotionCapture: {message}")

        return Message(timestamp=time.time(), message=message)

    async def raw_to_text(self, raw_input: Optional[cv2.typing.MatLike]):
        """
        Convert raw input to processed text and manage buffer.

        Parameters
        ----------
        raw_input : Optional[cv2.typing.MatLike]
            Raw video frame to be processed
        """
        pending_message = await self._raw_to_text(raw_input)

        if pending_message is not None:
            self.messages.append(pending_message)

    def formatted_latest_buffer(self) -> Optional[str]:
        """
        Format and clear the latest buffer contents.

        Returns
        -------
        Optional[str]
            Formatted string of buffer contents or None if buffer is empty
        """
        if len(self.messages) == 0:
            return None

        latest_message = self.messages[-1]

        result = f"""
{self.__class__.__name__} INPUT
// START
{latest_message.message}
// END
"""

        self.io_provider.add_input(
            self.__class__.__name__, latest_message.message, latest_message.timestamp
        )
        self.messages = []

        return result
