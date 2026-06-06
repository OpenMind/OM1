import asyncio
import logging
import time
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image as PILImage
from pydantic import Field

from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput
from providers.io_provider import IOProvider

SmolVLMForConditionalGeneration = None  # type: ignore[assignment]
SmolVLMProcessor = None  # type: ignore[assignment]
HAS_TRANSFORMERS = False

try:
    from transformers import (
        SmolVLMForConditionalGeneration,
        SmolVLMProcessor,
    )

    HAS_TRANSFORMERS = True
except ImportError:
    logging.warning(
        "SmolVLM local: 'transformers' not installed. "
        "Install with: pip install transformers num2words"
    )


class VLM_SmolVLM_LocalConfig(SensorConfig):
    """
    Configuration for SmolVLM2 local VLM sensor.

    Parameters
    ----------
    camera_index : int
        Index of the camera device to capture frames from.
    model_id : str
        HuggingFace model ID for SmolVLM2.
    prompt : str
        Text prompt sent alongside the image to the VLM.
    """

    camera_index: int = Field(default=0, description="Index of the camera device")
    model_id: str = Field(
        default="HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
        description="HuggingFace model ID for SmolVLM2",
    )
    prompt: str = Field(
        default="Briefly describe what you see in one or two sentences.",
        description="Text prompt sent alongside the image to the VLM",
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
        logging.error(f"VLM SmolVLM Local: camera not found at index {index_to_check}")
        cap.release()
        return False
    logging.info(f"VLM SmolVLM Local: camera found at index {index_to_check}")
    cap.release()
    return True


class VLM_SmolVLM_Local(FuserInput[VLM_SmolVLM_LocalConfig, Optional[np.ndarray]]):
    """
    Vision Language Model input using a local SmolVLM2 model.

    Captures frames from a webcam and runs inference using SmolVLM2
    directly via the HuggingFace transformers library — no internet
    connection or external server required after the initial model download.

    The model is downloaded automatically from HuggingFace on first run
    and cached locally. Default model is SmolVLM2-256M which requires
    less than 1GB of VRAM and can run on CPU as a fallback.

    Requires the transformers package:

        pip install transformers num2words

    For GPU acceleration, a CUDA-capable device is recommended but not required.
    """

    def __init__(self, config: VLM_SmolVLM_LocalConfig):
        """
        Initialize the SmolVLM2 local VLM input handler.

        Parameters
        ----------
        config : VLM_SmolVLM_LocalConfig
            Configuration settings for the SmolVLM2 sensor.
        """
        super().__init__(config)

        self.io_provider = IOProvider()
        self.messages: list[Message] = []
        self.descriptor_for_LLM = "Vision"

        self.model = None
        self.processor = None

        if not HAS_TRANSFORMERS:
            logging.error(
                "VLM SmolVLM Local: transformers not installed. Plugin disabled. "
                "Install with: pip install transformers num2words"
            )
            self.have_cam = False
            self.cap = None
            return

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"VLM SmolVLM Local: using device: {self.device}")

        logging.info(
            f"VLM SmolVLM Local: loading model '{config.model_id}' "
            "(downloading if not cached)..."
        )
        try:
            self.processor = SmolVLMProcessor.from_pretrained(config.model_id)  # type: ignore[union-attr]
            self.model = SmolVLMForConditionalGeneration.from_pretrained(  # type: ignore[union-attr]
                config.model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device,
            )
            self.model.eval()
            logging.info("VLM SmolVLM Local: model loaded successfully")
        except Exception as e:
            logging.error(f"VLM SmolVLM Local: failed to load model: {e}")
            self.have_cam = False
            self.cap = None
            return

        self.have_cam = check_webcam(self.config.camera_index)
        self.cap: Optional[cv2.VideoCapture] = None

        if self.have_cam:
            self.cap = cv2.VideoCapture(self.config.camera_index)
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logging.info(f"VLM SmolVLM Local: camera resolution {width}x{height}")

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
            logging.warning("VLM SmolVLM Local: failed to read frame from camera")
            return None

        return frame

    async def _raw_to_text(self, raw_input: Optional[np.ndarray]) -> Optional[Message]:
        """
        Run SmolVLM2 inference on a camera frame and return a text description.

        Converts the numpy frame to a PIL image, runs the SmolVLM2 model,
        and wraps the response in a timestamped Message.

        Parameters
        ----------
        raw_input : Optional[np.ndarray]
            Camera frame to process.

        Returns
        -------
        Optional[Message]
            Timestamped message containing the VLM description,
            or None if processing fails or input is None.
        """
        if raw_input is None:
            return None

        if self.model is None or self.processor is None:
            return None

        try:
            image = PILImage.fromarray(cv2.cvtColor(raw_input, cv2.COLOR_BGR2RGB))

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": self.config.prompt},
                    ],
                }
            ]

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(text=text, images=[image], return_tensors="pt").to(
                self.device
            )

            with torch.no_grad():
                output = self.model.generate(**inputs, max_new_tokens=128)

            description = self.processor.decode(
                output[0][inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            ).strip()

            if not description:
                logging.warning("VLM SmolVLM Local: received empty response")
                return None

            logging.info(f"VLM SmolVLM Local: {description}")
            return Message(timestamp=time.time(), message=description)

        except Exception as e:
            logging.error(f"VLM SmolVLM Local: inference error: {e}")
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

        logging.info(f"VLM_SmolVLM_Local: {latest_message.message}")

        result = f"""
INPUT: {self.descriptor_for_LLM}
// START
{latest_message.message}
// END
"""

        self.io_provider.add_input(
            self.descriptor_for_LLM,
            latest_message.message,
            latest_message.timestamp,
        )
        self.messages = []

        return result
