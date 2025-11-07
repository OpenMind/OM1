import argparse
import json
import logging
import os
import tempfile
import threading
import time
from io import BytesIO
from typing import Any, Callable, List, Optional

import torch
from PIL import Image as PILImage

from .vila_model import VILAModelSingleton

# llava is only on VILA server
# The dependency (bitsandbytes) is not available for Mac M chips
try:
    import llava
    from llava import conversation as clib
    from llava.media import Image, Video

    # PreTrainedModel doesn't work on Mac M chips
    from transformers import GenerationConfig
except ModuleNotFoundError:
    llava = None
    clib = None
    Image = None
    Video = None

root_package_name = __name__.split(".")[0] if "." in __name__ else __name__
logger = logging.getLogger(root_package_name)


class VILAProcessor:
    """
    Vision Language Model (VLM) processor for real-time video analysis.

    Processes video frames through a vision-language model to generate text descriptions
    of interesting aspects in the video stream. Designed specifically for NVIDIA Jetson
    devices using the nano_llm framework.

    Parameters
    ----------
    model_args : argparse.Namespace
        Command line arguments for model configuration.
    callback : Optional[Callable[[str], None]], optional
        Callback function for processing model responses,
        by default None
    """

    def __init__(
        self,
        model_args: argparse.Namespace,
        callback: Optional[Callable[[str], None]] = None,
    ):
        # Get or create the singleton model instance
        self.model_singleton = VILAModelSingleton()

        try:
            # Initialize the model if not already initialized
            self.model_singleton.initialize_model(model_args)
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise

        # Get the shared model instance
        self.model = self.model_singleton.model

        # Rest of the initialization
        self.model_args = model_args
        self.callback = callback
        self.image_buffer = []
        self.response = ""
        self.running = True

        # Add locks for thread safety
        self.model_lock = threading.Lock()
        self.buffer_lock = threading.Lock()

        # Create a dedicated CUDA stream for this processor
        self.cuda_stream = torch.cuda.Stream() if torch.cuda.is_available() else None

        try:
            self.model_config = GenerationConfig(
                max_new_tokens=48,
                temperature=0.7,
                top_p=0.95,
                do_sample=True,
            )
        except Exception as e:
            logger.error(f"Error initializing model configuration: {e}")
            raise

    def on_video(self, image: bytes) -> Any:
        """
        Process incoming video frames.

        Parameters
        ----------
        image : Any
            Input video frame to process

        Returns
        -------
        Any
            Annotated video frame
        """
        # Make sure image buffer always gets the latest image but stays the same size
        if len(self.image_buffer) == self.model_args.vila_batch_size:
            self.image_buffer.pop(0)
        self.image_buffer.append(image)

    def process_frames(self, video_output: Any, video_source: Any):
        """
        Main frame processing loop.

        Parameters
        ----------
        video_output : Any
            Video output stream handler
        video_source : Any
            Video input source handler
        """
        while self.running:
            # Send accumulated images to VILA server
            try:
                # Create a safe copy of the image buffer
                with self.buffer_lock:
                    # Only process if we have enough images
                    if len(self.image_buffer) < self.model_args.vila_batch_size:
                        # Not enough images yet, sleep briefly and continue
                        time.sleep(0.01)
                        continue

                    # Create a copy of the buffer to process
                    images_to_process = self.image_buffer.copy()
                    self.image_buffer.clear()

                logger.info(f"Processing {len(images_to_process)} images")
                message = {
                    "images": images_to_process,
                    "prompt": "What is the most interesting aspect in this series of images?",
                }

                # Process the images with proper synchronization
                self.response = self.generate_with_images(
                    message["images"], message["prompt"]
                )

                if self.callback:
                    self.callback(json.dumps({"vlm_reply": self.response}))
            except Exception as e:
                logger.error(f"Error processing frames: {e}")
                # Sleep briefly to avoid tight loop on error
                time.sleep(0.1)

    def generate_with_images(self, images: List[bytes], prompt: str):
        temp_files = []
        try:
            # Prepare multi-modal prompt
            prompt_list = []

            # Convert base64 images to Image objects
            start = time.time() * 1000
            for img_bytes in images:
                try:
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                    temp_files.append(temp_file.name)

                    img = PILImage.open(BytesIO(img_bytes))
                    img.save(temp_file.name)
                    temp_file.close()

                    prompt_list.append(Image(temp_file.name))
                except Exception as e:
                    # Clean up any temporary files created so far
                    logger.error(f"Error processing image: {e}")
                    for temp_path in temp_files:
                        if os.path.exists(temp_path):
                            os.unlink(temp_path)
                    raise Exception(f"Invalid image data: {str(e)}")
            end = time.time() * 1000
            logger.debug(f"Time taken to construct image files: {end - start} ms")

            # Add text prompt
            prompt_list.append(prompt)

            # Generate response using shared model with proper synchronization
            with self.model_lock:
                # Set the current CUDA stream
                if self.cuda_stream:
                    with torch.cuda.stream(self.cuda_stream):
                        # Add explicit CUDA synchronization before processing
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()

                        with torch.inference_mode():
                            response = self.model.generate_content(
                                prompt_list, self.model_config
                            )

                        # Synchronize again after processing
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                else:
                    with torch.inference_mode():
                        response = self.model.generate_content(
                            prompt_list, self.model_config
                        )

            end = time.time() * 1000
            logger.debug(f"Time taken to infer: {end - start} ms")

            return response
        finally:
            # Clean up temporary files
            for temp_path in temp_files:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

    def stop(self):
        """
        Stop frame processing.

        Sets the running flag to False to terminate processing loop.
        """
        self.running = False

        # Ensure CUDA resources are properly cleaned up
        if torch.cuda.is_available():
            try:
                # Wait for all CUDA operations to complete
                torch.cuda.synchronize()

                # Clean up CUDA cache
                torch.cuda.empty_cache()
            except Exception as e:
                logger.error(f"Error cleaning up CUDA resources: {e}")
