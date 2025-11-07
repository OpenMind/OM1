import argparse
from typing import Dict, List, Type, TypeVar

# Video stream input
# Video device input
# VLM Processors
from ..nv_nano_llm import VideoDeviceInput as NanoLLMVideoDeviceInput
from ..nv_nano_llm import VideoStreamInput as NanoLLMVideoStreamInput
from ..nv_nano_llm import VLMProcessor as NanoLLMProcessor
from ..nv_nano_llm.args import NanoLLMArgParser
from ..vila_vlm import VideoStreamInput as VILAVideoStreamInput
from ..vila_vlm import VILAProcessor as VILAProcessor
from ..vila_vlm.args import VILAArgParser

T_Parser = TypeVar("T_Parser", bound=argparse.ArgumentParser)
T_Processor = TypeVar("T_Processor", NanoLLMProcessor, VILAProcessor)
T_VideoDeviceInput = TypeVar("T_VideoDeviceInput", bound=NanoLLMVideoDeviceInput)
T_VideoStreamInput = TypeVar(
    "T_VideoStreamInput", NanoLLMVideoStreamInput, VILAVideoStreamInput
)

MODEL_CONFIGS: Dict[str, List[str]] = {
    "vila": ["vila", "standalone"],
    "nano_llm": ["model", "chat", "generation"],
}

MODEL_PARSERS: Dict[str, Type[T_Parser]] = {
    "vila": VILAArgParser,
    "nano_llm": NanoLLMArgParser,
}

VLM_PROCESSOR: Dict[str, Type[T_Processor]] = {
    "vila": VILAProcessor,
    "nano_llm": NanoLLMProcessor,
}

VIDEO_DEVICE_INPUT: Dict[str, Type[T_VideoDeviceInput]] = {
    "nano_llm": NanoLLMVideoDeviceInput,
}

VIDEO_STREAM_INPUT: Dict[str, Type[T_VideoStreamInput]] = {
    "vila": VILAVideoStreamInput,
    "nano_llm": NanoLLMVideoStreamInput,
}
