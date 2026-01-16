import logging
import sys
from types import ModuleType

from tests.integration.mock_inputs.mock_rplidar import MockRPLidar
from tests.integration.mock_inputs.mock_vlm_coco import MockVLM_COCO
from tests.integration.mock_inputs.mock_vlm_gemini import MockVLM_Gemini
from tests.integration.mock_inputs.mock_vlm_openai import MockVLM_OpenAI
from tests.integration.mock_inputs.mock_vlm_vila import MockVLM_Vila

# Store original classes to restore them later
_original_classes = {}


def register_mock_inputs():
    """
    Register mock inputs by directly replacing the classes in the inputs module.

    This approach is more direct and reliable than patching the load_input function.
    """
    # Import all the modules we need to modify
    import inputs.plugins.rplidar
    import inputs.plugins.vlm_coco_local
    import inputs.plugins.vlm_gemini
    import inputs.plugins.vlm_openai
    import inputs.plugins.vlm_vila

    # Save original classes for later restoration
    global _original_classes
    _original_classes = {
        "VLM_COCO_Local": inputs.plugins.vlm_coco_local.VLM_COCO_Local,
        "VLMOpenAI": inputs.plugins.vlm_openai.VLMOpenAI,
        "VLMGemini": inputs.plugins.vlm_gemini.VLMGemini,
        "VLMVila": inputs.plugins.vlm_vila.VLMVila,
        "RPLidar": inputs.plugins.rplidar.RPLidar,
    }

    # Replace with mock classes
    inputs.plugins.vlm_coco_local.VLM_COCO_Local = MockVLM_COCO
    inputs.plugins.vlm_openai.VLMOpenAI = MockVLM_OpenAI
    inputs.plugins.vlm_gemini.VLMGemini = MockVLM_Gemini
    inputs.plugins.vlm_vila.VLMVila = MockVLM_Vila
    inputs.plugins.rplidar.RPLidar = MockRPLidar

    # Add mock modules to namespace for discoverability
    mock_modules = {
        "inputs.plugins.mock_vlm_coco": {"MockVLM_COCO": MockVLM_COCO},
        "inputs.plugins.mock_vlm_openai": {"MockVLM_OpenAI": MockVLM_OpenAI},
        "inputs.plugins.mock_vlm_gemini": {"MockVLM_Gemini": MockVLM_Gemini},
        "inputs.plugins.mock_vlm_vila": {"MockVLM_Vila": MockVLM_Vila},
        "inputs.plugins.mock_rplidar": {"MockRPLidar": MockRPLidar},
    }

    for module_name, mock_classes in mock_modules.items():
        mock_module = ModuleType(module_name)
        for class_name, class_obj in mock_classes.items():
            setattr(mock_module, class_name, class_obj)
        sys.modules[module_name] = mock_module

    logging.info("Registered mock inputs by directly replacing classes")


def unregister_mock_inputs():
    """
    Restore the original input classes.
    """
    global _original_classes

    if _original_classes:
        # Restore original classes
        import inputs.plugins.rplidar
        import inputs.plugins.vlm_coco_local
        import inputs.plugins.vlm_gemini
        import inputs.plugins.vlm_openai
        import inputs.plugins.vlm_vila

        # Restore original classes
        for plugin_name, original_class in _original_classes.items():
            if plugin_name == "VLM_COCO_Local":
                inputs.plugins.vlm_coco_local.VLM_COCO_Local = original_class
            elif plugin_name == "VLMOpenAI":
                inputs.plugins.vlm_openai.VLMOpenAI = original_class
            elif plugin_name == "VLMGemini":
                inputs.plugins.vlm_gemini.VLMGemini = original_class
            elif plugin_name == "VLMVila":
                inputs.plugins.vlm_vila.VLMVila = original_class
            elif plugin_name == "RPLidar":
                inputs.plugins.rplidar.RPLidar = original_class
        # Remove mock modules
        mock_modules = [
            "inputs.plugins.mock_vlm_coco",
            "inputs.plugins.mock_vlm_openai",
            "inputs.plugins.mock_vlm_gemini",
            "inputs.plugins.mock_vlm_vila",
            "inputs.plugins.mock_rplidar",
        ]
        for module in mock_modules:
            sys.modules.pop(module, None)

        _original_classes = {}
        logging.info("Unregistered mock inputs and restored original classes")
