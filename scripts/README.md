# OM1 Utility Scripts

This directory contains utility scripts to help with OM1 development, deployment, and management.

## Available Scripts

### validate_config.py

Validates OM1 configuration files for common errors and issues.

**Usage**:
```bash
python scripts/validate_config.py config/spot.json5
```

**Checks**:
- Required fields presence
- Data type validation
- Value ranges (e.g., hertz > 0)
- Deprecated fields
- Configuration structure

**Example Output**:
```
Validating: config/spot.json5
------------------------------------------------------------
✅ Valid JSON

⚠️  Warnings (1):
  • High hertz value: 10.0. This may increase API costs.

------------------------------------------------------------
```

---

### api_cost_estimator.py

Estimates OpenMind API costs based on configuration and usage patterns.

**Usage**:
```bash
python scripts/api_cost_estimator.py config/spot.json5
```

**Provides**:
- Requests per hour/day/month
- Token usage estimates
- Hourly/daily/monthly cost projections
- Cost optimization suggestions

**Example Output**:
```
📊 API Cost Estimation for: spot.json5
============================================================

Configuration:
  Hertz: 0.1 Hz
  Requests per hour: 360
  Requests per day: 8,640

Token Estimation (per request):
  Input: ~800 tokens
  Output: ~100 tokens
  Total: ~900 tokens

💰 Cost Estimates:
  Per hour:  $0.0036
    ├─ Input:  $0.0006
    └─ Output: $0.0030

  Per day:   $0.0864
  Per month: $2.5920

💡 Cost Optimization Tips:
  ✓ Hertz is already optimized
  ✓ TeleopsConnection uses minimal API (status updates only)
```

---

### quickstart.sh

Interactive setup and launch script for new OM1 installations.

**Usage**:
```bash
./scripts/quickstart.sh
```

**Features**:
- Checks dependencies (Python, uv)
- Creates .env template if missing
- Validates configuration
- Estimates API costs
- Optionally starts OM1

**For New Users**:
```bash
# Clone repository
git clone https://github.com/OpenMind/OM1.git
cd OM1

# Run quickstart
./scripts/quickstart.sh
```

---

## Development Scripts

### Creating New Scripts

When adding new utility scripts:

1. **Make it executable**:
   ```bash
   chmod +x scripts/your_script.py
   ```

2. **Add shebang for Python scripts**:
   ```python
   #!/usr/bin/env python3
   ```

3. **Include help text**:
   ```python
   """
   Script description.
   
   Usage: python scripts/your_script.py [args]
   """
   ```

4. **Update this README** with documentation

---

## Script Guidelines

1. **Error Handling**: Always include proper error handling and user-friendly messages
2. **Exit Codes**: Use standard exit codes (0 = success, 1 = error)
3. **Dependencies**: Check for required dependencies before running
4. **Documentation**: Include clear usage examples in script docstrings
5. **Testing**: Test scripts on both macOS and Linux

---

## Common Tasks

### Validate All Configs

```bash
for config in config/*.json5; do
    echo "Validating $config..."
    python scripts/validate_config.py "$config"
done
```

### Cost Comparison

```bash
# Compare different configurations
python scripts/api_cost_estimator.py config/spot.json5
python scripts/api_cost_estimator.py config/spot_low_cost.json5
```

### Batch Processing

```bash
# Validate all configs in a directory
find config -name "*.json5" -exec python scripts/validate_config.py {} \;
```

---

## Troubleshooting

### Permission Denied

If you get "Permission denied" when running scripts:

```bash
chmod +x scripts/*.py scripts/*.sh
```

### Python Not Found

Make sure Python 3 is installed and in your PATH:

```bash
which python3
python3 --version
```

### Module Not Found

Ensure you're running from the OM1 root directory with uv:

```bash
cd /path/to/OM1
uv run python scripts/your_script.py
```

---

## Contributing

To add new utility scripts:

1. Follow the script guidelines above
2. Add comprehensive documentation
3. Include usage examples
4. Update this README
5. Test on multiple platforms

For major contributions, consider opening a PR first to discuss the approach.
