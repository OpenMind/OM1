# Troubleshooting Guide

## Optional Plugin Dependencies

### Overview

OM1 is designed with modularity in mind. Many input plugins (such as VLM-based sensors) are **optional** and require additional dependencies that may not be needed for all use cases. OM1 gracefully handles missing optional dependencies by registering stub sensors that log warnings instead of crashing the application.

### Common Issue: ImportError for VLM Plugins

**Problem:** When running OM1 (e.g., `uv run src/run.py spot`), you may encounter warnings about missing optional input modules:

```
WARNING: Optional input module 'vlm_coco_local' could not be imported: No module named 'torch'
This is likely due to missing dependencies. A stub sensor will be registered for 'VLM_COCO_Local'.
To enable this sensor, install the required dependencies.
```

**What this means:** The VLM plugin requires dependencies (like PyTorch, transformers, etc.) that are not installed in your environment. This is **expected behavior** if you don't need these specific plugins.

**Impact:** 
- ✅ OM1 continues to run normally
- ✅ Other configured sensors work as expected
- ⚠️ The specific VLM plugin with missing dependencies will not provide data
- ⚠️ A stub sensor is used instead, which returns None and logs warnings

### Solution: Install Optional Dependencies

If you need to use VLM plugins with local models, install the required dependencies:

#### For VLM plugins (PyTorch-based):

```bash
# Install PyTorch (CPU version)
pip install torch torchvision torchaudio

# Install transformers and related packages
pip install transformers pillow
```

#### For specific VLM models:

```bash
# For COCO-based VLM
pip install pycocotools

# For YOLO-based VLM
pip install ultralytics

# For VILA-based VLM
pip install transformers accelerate
```

### Verifying Plugin Status

After installing dependencies, restart OM1:

```bash
uv run src/run.py spot
```

Check the logs:
- ✅ **Success:** `Loaded input VLM_COCO_Local from vlm_coco_local.py`
- ⚠️ **Still using stub:** `Optional input module 'vlm_coco_local' could not be imported...`

### Running OM1 Without Optional Plugins

OM1 is designed to work with minimal dependencies. If you don't need VLM plugins:

1. **No action required** - The stub sensors allow OM1 to run normally
2. **Modify your config** - Edit your agent configuration (e.g., `config/spot.json5`) to remove unused VLM inputs
3. **Use alternative inputs** - Switch to plugins with fewer dependencies:
   - `vlm_openai` - Uses OpenAI's API (no local dependencies)
   - `vlm_gemini` - Uses Google's Gemini API
   - `vlm_dummy_local` - Minimal test plugin

### Configuration Best Practices

When designing agents, consider dependency requirements:

```json5
{
  "inputs": [
    // ✅ Minimal dependencies - works everywhere
    {
      "type": "vlm_openai",
      "name": "vision"
    },
    
    // ⚠️ Heavy dependencies - requires PyTorch, etc.
    {
      "type": "vlm_coco_local",
      "name": "local_vision"
    }
  ]
}
```

### Understanding Stub Sensors

When optional dependencies are missing, OM1 automatically creates stub sensors:

- **Purpose:** Allow the application to start without crashing
- **Behavior:** 
  - Logs clear warnings at load time and instantiation
  - Returns `None` from `read()` method
  - Does not process any actual sensor data
- **When to use:** During development, testing with minimal dependencies, or when specific sensors aren't needed

### Getting Help

If you encounter issues:

1. **Check the logs** - OM1 provides detailed warnings about missing dependencies
2. **Review your config** - Ensure your agent configuration matches your installed dependencies
3. **Consult the docs** - Visit [docs.openmind.org](https://docs.openmind.org/) for more information
4. **Ask for help** - Join our [Discord](https://discord.gg/VUjpg4ef5n) community

### Related Issues

- [Issue #585](https://github.com/OpenMind/OM1/issues/585): ImportError for vlm_coco_local in fresh environment

---

*Last updated: 2025-10-25*
