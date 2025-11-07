import argparse
import logging
from typing import Any, Optional

from om1_utils import singleton

# llava is only on VILA server
# The dependency (bitsandbytes) is not available for Mac M chips
try:
    import llava
    from llava import conversation as clib

    # PreTrainedModel doesn't work on Mac M chips
    from transformers import PreTrainedModel
except ModuleNotFoundError:
    llava = None

root_package_name = __name__.split(".")[0] if "." in __name__ else __name__
logger = logging.getLogger(root_package_name)


@singleton
class VILAModelLoader:
    """
    Singleton class for loading and managing the VILA model.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments for model configuration.
    """

    def __init__(self, args: argparse.Namespace):
        """
        Initialize the VILA model loader.

        Parameters
        ----------
        args : argparse.Namespace
            Command line arguments for model configuration.
        """
        # Load the VILA model arguments
        self.args = args

        # Initialize model instance
        self._model: Optional[PreTrainedModel] = None

        # Load the model
        self._model = self._load_model()

    def _load_model(self) -> Optional[Any]:
        """
        Load the VILA model and move it to CUDA device.

        Returns
        -------
        PreTrainedModel
            Loaded VILA model instance.

        Raises
        ------
        Exception
            If model loading fails.
        """
        try:
            model = llava.load(self.args.vila_model_path)
            model.to("cuda")
            clib.default_conversation = clib.conv_templates["vicuna_v1"].copy()
            assert model
            return model
        except Exception as e:
            raise Exception("Failed to load VILA model") from e

    @property
    def model(self) -> Optional[Any]:
        """
        Get the VILA model instance, loading it if not already loaded.

        Returns
        -------
        PreTrainedModel
            The loaded VILA model instance.
        """
        if not self._model:
            self._model = self._load_model()
        return self._model

    @model.setter
    def model(self, value):
        """
        Set the VILA model instance.

        Parameters
        ----------
        value : PreTrainedModel
            The VILA model instance to set.
        """
        self._model = value
