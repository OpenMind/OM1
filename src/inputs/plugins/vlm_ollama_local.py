import asyncio
import base64
import logging
import time
from typing import Optional

import aiohttp
import cv2
import numpy as np
from pydantic import Field

from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput
from providers.io_provider import IOProvider


class VLM_Ollama_LocalConfig(SensorConfig):
    """
    Configuration for Ollama-based local VLM sensor.

    Parameters
    ----------
    camera_index : int
        Index of the camera device to capture frames from.
    base_url : str
        Base URL for the Ollama API service.
    model : str
        Ollama multimodal model name (e.g., llava, llava-phi3, moondream).
    prompt : str
        Text prompt sent alongside the image to the VLM.
    timeout : int
        Request timeout in seconds for Ollama inference.
    """

    camera_index: int = Field(default=0, description="Index of the camera device")
    base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL for the Ollama API service",
    )
    model: str = Field(
        default="llava",
        description="Ollama multimodal model name (e.g., llava, llava-phi3, moondream)",
    )
    prompt: str = Field(
        default="Briefly describe what you see in one or two sentences.",
        description="Text prompt sent alongside the image to the VLM",
    )
    timeout: int = Field(
        default=30,
        description="Request timeout in seconds for Ollama inference",
    )


def check_webcam(index_to_check: int) -> bool:
    """
    Check if a webcam is available at the given index.

    Parameters
    ----------
    index_to_check : int
        The camera index to check.

    Returns
    -------
    bool
        True if the webcam is available, False otherwise.
    """
    cap = cv2.VideoCapture(index_to_check)
    if not cap.isOpened():
        logging.error(f"VLM Ollama Local: camera not found at index {index_to_check}")
        cap.release()
        return False
    logging.info(f"VLM Ollama Local: camera found at index {index_to_check}")
    cap.release()
    return True


class VLM_Ollama_Local(FuserInput[VLM_Ollama_LocalConfig, Optional[np.ndarray]]):
    """
    Vision Language Model input using a local Ollama multimodal model.

    Captures frames from a webcam, encodes them as base64, and sends
    them to a locally running Ollama instance (e.g., llava, moondream)
    for visual reasoning. The resulting text description is forwarded
    to the fuser for use by the main LLM.

    Requires Ollama to be running locally with a multimodal model pulled:

        ollama pull llava
        ollama serve
    """

    def __init__(self, config: VLM_Ollama_LocalConfig):
        """
        Initialize the Ollama VLM input handler.

        Parameters
        ----------
        config : VLM_Ollama_LocalConfig
            Configuration settings for the Ollama VLM sensor.
        """
        super().__init__(config)

        self.io_provider = IOProvider()

        self.messages: list[Message] = []

        self.descriptor_for_LLM = "Vision"

        base_url = self.config.base_url.rstrip("/")
        self._chat_url = f"{base_url}/api/chat"

        self.have_cam = check_webcam(self.config.camera_index)
        self.cap: Optional[cv2.VideoCapture] = None

        if self.have_cam:
            self.cap = cv2.VideoCapture(self.config.camera_index)
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logging.info(f"VLM Ollama Local: camera resolution {width}x{height}")

        logging.info(f"VLM Ollama Local: initialized with model='{self.config.model}' endpoint='{self._chat_url}'")

    async def _poll(self) -> Optional[np.ndarray]:
        """
        Poll for a new frame from the camera.

        Returns
        -------
        Optional[np.ndarray]
            Captured frame as a numpy array, or None if unavailable.
        """
        await asyncio.sleep(0.5)

        if not self.have_cam or self.cap is None:
            return None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            logging.warning("VLM Ollama Local: failed to read frame from camera")
            return None

        return frame

    async def _raw_to_text(self, raw_input: Optional[np.ndarray]) -> Optional[Message]:
        """
        Send a camera frame to Ollama and return the text response.

        Encodes the frame as a JPEG base64 string, posts it to the
        Ollama /api/chat endpoint with the configured prompt, and
        wraps the response in a timestamped Message.

        Parameters
        ----------
        raw_input : Optional[np.ndarray]
            Camera frame to process.

        Returns
        -------
        Optional[Message]
            Timestamped message containing the VLM description,
            or None if processing fails.
        """
        if raw_input is None:
            return None

        success, buffer = cv2.imencode(".jpg", raw_input)
        if not success:
            logging.error("VLM Ollama Local: failed to encode frame to JPEG")
            return None

        image_b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")

        payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "user",
                    "content": self.config.prompt,
                    "images": [image_b64],
                }
            ],
            "stream": False,
        }

        try:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(self._chat_url, json=payload) as response:
                    if response.status != 200:
                        logging.error(f"VLM Ollama Local: API error {response.status}")
                        return None

                    result = await response.json()

            description = result.get("message", {}).get("content", "").strip()

            if not description:
                logging.warning("VLM Ollama Local: received empty response")
                return None

            logging.info(f"VLM Ollama Local: {description}")
            return Message(timestamp=time.time(), message=description)

        except aiohttp.ClientConnectorError:
            logging.error("VLM Ollama Local: cannot connect to Ollama. Is Ollama running? Start with: ollama serve")
            return None
        except asyncio.TimeoutError:
            logging.error(
                f"VLM Ollama Local: request timed out after {self.config.timeout}s. "
                "Try increasing timeout or using a smaller model."
            )
            return None
        except Exception as e:
            logging.error(f"VLM Ollama Local: unexpected error: {e}")
            return None

    async def raw_to_text(self, raw_input: Optional[np.ndarray]):
        """
        Convert a camera frame to text and append to the message buffer.

        Parameters
        ----------
        raw_input : Optional[np.ndarray]
            Camera frame to process.
        """
        pending_message = await self._raw_to_text(raw_input)

        if pending_message is not None:
            self.messages.append(pending_message)

    def formatted_latest_buffer(self) -> Optional[str]:
        """
        Format the latest buffered message for the fuser and clear the buffer.

        Retrieves the most recent VLM description, formats it with the
        standard INPUT block structure, records it in the IO provider,
        and clears the internal message buffer.

        Returns
        -------
        Optional[str]
            Formatted input string for the fuser, or None if buffer is empty.
        """
        if len(self.messages) == 0:
            return None

        latest_message = self.messages[-1]

        logging.info(f"VLM_Ollama_Local: {latest_message.message}")

        result = f"\nINPUT: {self.descriptor_for_LLM}\n// START\n{latest_message.message}\n// END\n"

        self.io_provider.add_input(
            self.descriptor_for_LLM,
            latest_message.message,
            latest_message.timestamp,
        )
        self.messages = []

        return result
