# OM1 Configuration Examples

This directory contains example configuration files for different OM1 use cases.

## Available Examples

### minimal_config.json5

The simplest working OM1 configuration.

**Use case**: Testing, development, learning OM1 basics

**Features**:
- Minimal API usage (0.1 Hz)
- No inputs (audio/voice only)
- WebSim on port 8001
- No background tasks
- Basic speak action

**Estimated cost**: ~$0.50/month

**To use**:
```bash
cp examples/minimal_config.json5 config/my_robot.json5
uv run src/run.py my_robot
```

---

### production_config.json5

Production-ready configuration with monitoring and reliability features.

**Use case**: Production deployments, long-running systems

**Features**:
- Optimized API usage (0.1 Hz)
- VLM vision input (COCO object detection)
- WebSim with auto-reconnect
- TeleopsConnection for portal visibility
- HealthCheck for resource monitoring
- AutoRestart for crash recovery
- APILimiter for cost control

**Estimated cost**: ~$2-3/month

**To use**:
```bash
cp examples/production_config.json5 config/my_robot.json5
# Edit config/my_robot.json5 to add your API key and URID
uv run src/run.py my_robot
```

---

## Configuration Templates

### Development Configuration

For development and testing:
```json5
{
  "hertz": 0.1,  // Minimize API calls
  "agent_inputs": [],  // Fast iteration without VLM
  "backgrounds": [
    {
      "type": "ConfigWatcher",  // Auto-reload on changes
      "config": {
        "config_paths": ["config/dev.json5"]
      }
    }
  ]
}
```

### Cost-Optimized Configuration

For minimal API costs:
```json5
{
  "hertz": 0.05,  // Very low frequency
  "cortex_llm": {
    "config": {
      "history_length": 1  // Reduced context
    }
  }
}
```

### High-Performance Configuration

For maximum responsiveness:
```json5
{
  "hertz": 1.0,  // Higher frequency
  "agent_inputs": [
    {"type": "VLM_COCO_Local"},  // Vision
    {"type": "Microphone_Input"}  // Audio
  ],
  "cortex_llm": {
    "config": {
      "history_length": 5  // More context
    }
  }
}
```

---

## Port Selection

Avoid port conflicts:
- **Port 8000**: Often used by other tools (perp dex tool, etc.)
- **Port 8001**: Recommended for WebSim
- **Custom ports**: Change in `simulators.config.port`

```json5
{
  "simulators": [
    {
      "type": "WebSim",
      "config": {
        "port": 8001  // Avoid port 8000
      }
    }
  ]
}
```

---

## API Key Setup

1. Get your API key from https://fabric.openmind.org
2. Add to `.env` file:
   ```
   OM_API_KEY=om1_live_your_api_key_here
   URID=your_urid_here
   ```
3. Or add directly to config:
   ```json5
   {
     "api_key": "om1_live_your_api_key_here",
     "URID": "your_urid_here"
   }
   ```

---

## Cost Estimation

Before running a configuration, estimate costs:

```bash
python scripts/api_cost_estimator.py examples/production_config.json5
```

---

## Validation

Validate your configuration before using:

```bash
python scripts/validate_config.py config/my_robot.json5
```

---

## Customization

To create your own configuration:

1. **Start with an example**: Copy the closest example
2. **Edit for your needs**: Modify fields as needed
3. **Validate**: Run validation script
4. **Estimate costs**: Check API costs
5. **Test**: Run in development first

---

## Troubleshooting

### Port Already in Use

If you see "port already in use":
- Change the WebSim port in your config
- Or stop the conflicting service

### API Key Errors

If you get "unauthorized" or "invalid API key":
- Check your API key in `.env` or config
- Verify your key is active at https://fabric.openmind.org
- Ensure URID matches your API key

### High API Costs

If API costs are higher than expected:
- Reduce `hertz` value
- Reduce `history_length` in cortex_llm
- Remove unused inputs
- Use cost estimator to identify issues

---

## More Examples

For more examples, see:
- [Main Documentation](../README.md)
- [Background Plugins Guide](../docs/BACKGROUND_PLUGINS.md)
- [Contributing Guidelines](../CONTRIBUTING.md)
