import logging
import threading

from om1_utils import singleton

from .model_loader import VILAModelLoader

# llava is only on VILA server
# The dependency (bitsandbytes) is not available for Mac M chips
try:
    import llava
    from llava import conversation as clib
    from llava.media import Image, Video
except ModuleNotFoundError:
    llava = None
    clib = None
    Image = None
    Video = None

root_package_name = __name__.split(".")[0] if "." in __name__ else __name__
logger = logging.getLogger(root_package_name)


@singleton
class VILAModelSingleton:
    def __init__(self):
        self._lock: threading = threading.Lock()
        self._model = None
        self._is_initialized: bool = False

    def initialize_model(self, model_args):
        """Thread-safe model initialization"""
        with self._lock:
            if not self._is_initialized:
                try:
                    model_loader = VILAModelLoader(model_args)
                    self._model = model_loader.model
                    self._warmup_model(model_args)
                    self._is_initialized = True
                except Exception as e:
                    logger.error(f"Model initialization failed: {e}")
                    raise

    def _warmup_model(self, model_args):
        """Warm up the model with dummy data to ensure it's ready for inference"""
        pass

    @property
    def model(self):
        """Get the initialized model"""
        if not self._is_initialized:
            raise RuntimeError("Model not initialized. Call initialize_model first.")
        return self._model
