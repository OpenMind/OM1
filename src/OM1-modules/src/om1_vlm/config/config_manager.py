import argparse
from typing import Dict, List, Optional

from om1_utils import singleton

from .config import (
    MODEL_CONFIGS,
    MODEL_PARSERS,
    VIDEO_DEVICE_INPUT,
    VIDEO_STREAM_INPUT,
    VLM_PROCESSOR,
    T_Parser,
)


@singleton
class ConfigManager:
    """
    Configuration manager for Vision Language Models (VLM).

    A singleton class that manages configuration, argument parsing, and component access
    for different Vision Language Models. Handles model-specific configurations and
    provides access to model processors and video input components.
    """

    def __init__(self):
        self.model_configs: Dict[str, List[str]] = MODEL_CONFIGS
        self.model_parsers: Dict[str, type[T_Parser]] = MODEL_PARSERS
        self.model_name: Optional[str] = None

    def get_parser_for_model(self, model_name: str) -> argparse.ArgumentParser:
        """
        Get argument parser for specified model.

        Parameters
        ----------
        model_name : str
            Name of the model to get parser for

        Returns
        -------
        argparse.ArgumentParser
            Argument parser configured for the specified model

        Raises
        ------
        ValueError
            If model_name is not found in model configurations
        """
        if model_name not in self.model_configs:
            raise ValueError(f"Model '{model_name}' not found in model configurations")

        self.model_name = model_name
        extras = self.model_configs[model_name]
        return self.model_parsers[model_name](extras=extras)

    def parse_model_arguments(self) -> argparse.ArgumentParser:
        """Parse command line arguments to determine model and get its parser.

        Returns
        -------
        argparse.ArgumentParser
            Argument parser for the model specified in command line arguments
        """
        temp_parser = argparse.ArgumentParser(add_help=False)
        temp_parser.add_argument(
            "--model-name", type=str, default="vila", help="VLM model path/identifier"
        )
        temp_args, _ = temp_parser.parse_known_args()

        parser = self.get_parser_for_model(temp_args.model_name)
        parser.add_argument(
            "--model-name", type=str, default="vila", help="VLM model path/identifier"
        )

        return parser

    @property
    def vlm_processor(self):
        """
        Get the VLM processor for the currently selected model.

        Returns
        -------
        Type
            VLM processor class for the current model

        Raises
        ------
        ValueError
            If no model is selected or if current model has no VLM processor
        """
        if self.model_name not in VLM_PROCESSOR:
            raise ValueError(
                f"Model '{self.model_name}' not found in model configurations"
            )
        return VLM_PROCESSOR[self.model_name]

    @property
    def video_device_input(self):
        """
        Get the video device input class for the currently selected model.

        Returns
        -------
        Type
            Video device input class for the current model

        Raises
        ------
        ValueError
            If no model is selected or if current model has no video device input
        """
        if self.model_name not in VIDEO_DEVICE_INPUT:
            raise ValueError(
                f"Model '{self.model_name}' not found in model configurations"
            )
        return VIDEO_DEVICE_INPUT[self.model_name]

    @property
    def video_stream_input(self):
        """
        Get the video stream input class for the currently selected model.

        Returns
        -------
        Type
            Video stream input class for the current model

        Raises
        ------
        ValueError
            If no model is selected or if current model has no video stream input
        """
        if self.model_name not in VIDEO_STREAM_INPUT:
            raise ValueError(
                f"Model '{self.model_name}' not found in model configurations"
            )
        return VIDEO_STREAM_INPUT[self.model_name]
