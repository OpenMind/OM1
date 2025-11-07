import argparse
import json
import logging
from typing import Any, Callable, Optional

# nano_llm and jestson_utils are only available on Jetson devices
try:
    import nano_llm
    from jetson_utils import cudaFont, cudaMemcpy
    from nano_llm import ChatHistory, NanoLLM, remove_special_tokens
    from nano_llm.utils import wrap_text
except ModuleNotFoundError:
    NanoLLM = None
    ChatHistory = None
    remove_special_tokens = None
    wrap_text = None
    cudaMemcpy = None
    cudaFont = None

root_package_name = __name__.split(".")[0] if "." in __name__ else __name__
logger = logging.getLogger(root_package_name)


class VLMProcessor:
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
        self.model: nano_llm.NanoLLM = self._initialize_model(model_args)
        self.model_args = model_args
        self.chat_history: nano_llm.ChatHistory = self._initialize_chat_history(
            model_args
        )
        self.callback = callback
        self.last_image: Any = None
        self.num_images: int = 0
        self.raw_response: str = "test"
        self.response: str = ""
        self.font: Any = cudaFont()
        self.running: bool = True

        # Warm up the model
        self._warmup_model()

    def _initialize_model(self, args: argparse.Namespace):
        """
        Initialize the vision-language model.

        Parameters
        ----------
        args : argparse.Namespace
            Model configuration arguments

        Returns
        -------
        nano_llm.NanoLLM
            Initialized model instance

        Raises
        ------
        AssertionError
            If the model does not have vision capabilities
        """
        model = NanoLLM.from_pretrained(
            args.model,
            api=args.api,
            quantization=args.quantization,
            max_context_len=args.max_context_len,
            vision_api=args.vision_api,
            vision_model=args.vision_model,
            vision_scaling=args.vision_scaling,
        )
        assert model.has_vision
        return model

    def _initialize_chat_history(self, args: argparse.Namespace):
        """
        Initialize chat history for the model.

        Parameters
        ----------
        args : argparse.Namespace
            Arguments containing chat template and system prompt

        Returns
        -------
        nano_llm.ChatHistory
            Initialized chat history instance
        """
        return ChatHistory(self.model, args.chat_template, args.system_prompt)

    def _warmup_model(self):
        """
        Perform model warmup with a simple query.

        Sends a basic arithmetic query to ensure the model is loaded
        and ready for processing.
        """
        self.chat_history.append(role="user", text="What is 2+2?")
        logging.info(
            f"Warmup response: '{self.model.generate(self.chat_history.embed_chat()[0], streaming=False)}'".replace(
                "\n", "\\n"
            )
        )
        self.chat_history.reset()

    def cleanup(self, text: str) -> str:
        """
        Clean up model response text.

        Parameters
        ----------
        text : str
            Raw model response text

        Returns
        -------
        str
            Cleaned and formatted response text
        """
        response = remove_special_tokens(text)
        response = response.lower()
        response = response.replace(
            "the most interesting aspect of this series of images is", "You see"
        )
        response = response.replace(
            "the most interesting aspect of the series of images is", "You see"
        )
        response = response.replace(
            "the most interesting aspect of the images is", "You see"
        )
        response = response.replace(
            "the most interesting aspect of these images is", "You see"
        )
        response = response.replace(
            "the most interesting aspect of the video is", "You see"
        )
        return response

    def on_video(self, image: Any) -> Any:
        """
        Process incoming video frames.

        Parameters
        ----------
        image : Any
            Input video frame in CUDA memory

        Returns
        -------
        Any
            Annotated video frame
        """
        self.last_image = cudaMemcpy(image)

        annotation = "Accumulating:" + str(self.num_images)
        if self.num_images >= 5:
            annotation = "VLM:" + self.response

        wrap_text(
            self.font,
            image,
            text=annotation,
            x=5,
            y=5,
            color=(255, 0, 0),
            background=self.font.Gray50,
        )
        return image

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
        skip = 0
        while self.running:
            if self.last_image is None:
                continue

            if skip < 5:
                skip += 1
                continue
            else:
                skip = 0

            self.chat_history.append("user", text=f"Image {self.num_images + 1}:")
            self.chat_history.append("user", image=self.last_image)
            self.last_image = None

            if self.num_images < 5:
                self.num_images += 1
                continue
            else:
                self.num_images = 0

            self.chat_history.append(
                "user", "What is the most interesting aspect in this series of images?"
            )
            embedding, _ = self.chat_history.embed_chat()

            reply = self.model.generate(
                embedding,
                kv_cache=self.chat_history.kv_cache,
                max_new_tokens=self.model_args.max_new_tokens,
                min_new_tokens=self.model_args.min_new_tokens,
                do_sample=self.model_args.do_sample,
                repetition_penalty=self.model_args.repetition_penalty,
                temperature=self.model_args.temperature,
                top_p=self.model_args.top_p,
            )

            for token in reply:
                if len(reply.tokens) == 1:
                    self.raw_response = token
                else:
                    self.raw_response = self.raw_response + token

            self.response = self.cleanup(self.raw_response)
            logger.info(f"VLM response: {self.response}")

            if self.callback:
                self.callback(json.dumps({"vlm_reply": self.response}))

            self.chat_history.reset()

            if video_source.eos:
                video_output.stream.Close()
                break

    def stop(self):
        """
        Stop frame processing.

        Sets the running flag to False to terminate processing loop.
        """
        self.running = False
