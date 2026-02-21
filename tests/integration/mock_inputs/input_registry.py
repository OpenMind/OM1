import logging
import sys
from types import ModuleType

from tests.integration.mock_inputs.mock_battery import MockUnitreeGo2Battery
from tests.integration.mock_inputs.mock_google_asr import MockGoogleASR
from tests.integration.mock_inputs.mock_gps import MockGps
from tests.integration.mock_inputs.mock_odometry import MockUnitreeGo2Odom
from tests.integration.mock_inputs.mock_unitree_go2_rplidar import MockUnitreeGo2RPLidar
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
    import inputs.plugins.google_asr
    import inputs.plugins.gps
    import inputs.plugins.unitree_go2_battery
    import inputs.plugins.unitree_go2_odom
    import inputs.plugins.unitree_go2_rplidar
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
        "UnitreeGo2RPLidar": inputs.plugins.unitree_go2_rplidar.UnitreeGo2RPLidar,
        "GoogleASRInput": inputs.plugins.google_asr.GoogleASRInput,
        "UnitreeGo2Battery": inputs.plugins.unitree_go2_battery.UnitreeGo2Battery,
        "UnitreeGo2Odom": inputs.plugins.unitree_go2_odom.UnitreeGo2Odom,
        "Gps": inputs.plugins.gps.Gps,
    }

    # Replace with mock classes
    inputs.plugins.vlm_coco_local.VLM_COCO_Local = MockVLM_COCO
    inputs.plugins.vlm_openai.VLMOpenAI = MockVLM_OpenAI
    inputs.plugins.vlm_gemini.VLMGemini = MockVLM_Gemini
    inputs.plugins.vlm_vila.VLMVila = MockVLM_Vila
    inputs.plugins.unitree_go2_rplidar.UnitreeGo2RPLidar = MockUnitreeGo2RPLidar
    inputs.plugins.google_asr.GoogleASRInput = MockGoogleASR
    inputs.plugins.unitree_go2_battery.UnitreeGo2Battery = MockUnitreeGo2Battery
    inputs.plugins.unitree_go2_odom.UnitreeGo2Odom = MockUnitreeGo2Odom
    inputs.plugins.gps.Gps = MockGps

    # Add mock modules to namespace for discoverability
    mock_modules = {
        "inputs.plugins.mock_vlm_coco": {"MockVLM_COCO": MockVLM_COCO},
        "inputs.plugins.mock_vlm_openai": {"MockVLM_OpenAI": MockVLM_OpenAI},
        "inputs.plugins.mock_vlm_gemini": {"MockVLM_Gemini": MockVLM_Gemini},
        "inputs.plugins.mock_vlm_vila": {"MockVLM_Vila": MockVLM_Vila},
        "inputs.plugins.mock_unitree_go2_rplidar": {
            "MockUnitreeGo2RPLidar": MockUnitreeGo2RPLidar
        },
        "inputs.plugins.mock_google_asr": {"MockGoogleASR": MockGoogleASR},
        "inputs.plugins.mock_battery": {"MockUnitreeGo2Battery": MockUnitreeGo2Battery},
        "inputs.plugins.mock_odometry": {"MockUnitreeGo2Odom": MockUnitreeGo2Odom},
        "inputs.plugins.mock_gps": {"MockGps": MockGps},
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
        import inputs.plugins.google_asr
        import inputs.plugins.gps
        import inputs.plugins.unitree_go2_battery
        import inputs.plugins.unitree_go2_odom
        import inputs.plugins.unitree_go2_rplidar
        import inputs.plugins.vlm_coco_local
        import inputs.plugins.vlm_gemini
        import inputs.plugins.vlm_openai
        import inputs.plugins.vlm_vila

        # Map class names to (module, attribute) for restoration
        class_to_module = {
            "VLM_COCO_Local": (inputs.plugins.vlm_coco_local, "VLM_COCO_Local"),
            "VLMOpenAI": (inputs.plugins.vlm_openai, "VLMOpenAI"),
            "VLMGemini": (inputs.plugins.vlm_gemini, "VLMGemini"),
            "VLMVila": (inputs.plugins.vlm_vila, "VLMVila"),
            "UnitreeGo2RPLidar": (
                inputs.plugins.unitree_go2_rplidar,
                "UnitreeGo2RPLidar",
            ),
            "GoogleASRInput": (inputs.plugins.google_asr, "GoogleASRInput"),
            "UnitreeGo2Battery": (
                inputs.plugins.unitree_go2_battery,
                "UnitreeGo2Battery",
            ),
            "UnitreeGo2Odom": (inputs.plugins.unitree_go2_odom, "UnitreeGo2Odom"),
            "Gps": (inputs.plugins.gps, "Gps"),
        }

        for plugin_name, original_class in _original_classes.items():
            if plugin_name in class_to_module:
                module, attr = class_to_module[plugin_name]
                setattr(module, attr, original_class)

        # Remove mock modules
        mock_modules = [
            "inputs.plugins.mock_vlm_coco",
            "inputs.plugins.mock_vlm_openai",
            "inputs.plugins.mock_vlm_gemini",
            "inputs.plugins.mock_vlm_vila",
            "inputs.plugins.mock_unitree_go2_rplidar",
            "inputs.plugins.mock_google_asr",
            "inputs.plugins.mock_battery",
            "inputs.plugins.mock_odometry",
            "inputs.plugins.mock_gps",
        ]
        for module in mock_modules:
            sys.modules.pop(module, None)

        _original_classes = {}
        logging.info("Unregistered mock inputs and restored original classes")
