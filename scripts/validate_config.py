#!/usr/bin/env python3
"""
Configuration Validation Script

Validates OM1 configuration files for common errors and issues.
Usage: python scripts/validate_config.py [config_file]
"""

import json
import sys
from pathlib import Path


def validate_json5(path):
    """Check if file is valid JSON5 (or JSON)."""
    try:
        with open(path, 'r') as f:
            content = f.read()
            # Try JSON first
            try:
                json.loads(content)
                return True, "Valid JSON"
            except json.JSONDecodeError:
                # JSON5 allows trailing commas, comments, etc.
                # For now, accept it as potentially valid
                return True, "Valid JSON5 (not fully validated)"
    except Exception as e:
        return False, f"Invalid: {e}"


def validate_config_structure(config):
    """Validate configuration structure."""
    errors = []
    warnings = []

    # Required top-level fields
    required_fields = ["hertz", "name"]
    for field in required_fields:
        if field not in config:
            errors.append(f"Missing required field: {field}")

    # Validate hertz
    if "hertz" in config:
        hertz = config["hertz"]
        if not isinstance(hertz, (int, float)) or hertz <= 0:
            errors.append(f"Invalid hertz value: {hertz}. Must be positive number.")
        elif hertz > 10:
            warnings.append(f"High hertz value: {hertz}. This may increase API costs.")

    # Validate agent_inputs
    if "agent_inputs" in config:
        agent_inputs = config["agent_inputs"]
        if not isinstance(agent_inputs, list):
            errors.append("agent_inputs must be a list")
        else:
            for i, inp in enumerate(agent_inputs):
                if not isinstance(inp, dict):
                    errors.append(f"agent_inputs[{i}] must be a dict")
                elif "type" not in inp:
                    errors.append(f"agent_inputs[{i}] missing 'type' field")

    # Validate simulators
    if "simulators" in config:
        simulators = config["simulators"]
        if not isinstance(simulators, list):
            errors.append("simulators must be a list")
        else:
            for i, sim in enumerate(simulators):
                if not isinstance(sim, dict):
                    errors.append(f"simulators[{i}] must be a dict")
                elif "type" not in sim:
                    errors.append(f"simulators[{i}] missing 'type' field")

    # Validate backgrounds
    if "backgrounds" in config:
        backgrounds = config["backgrounds"]
        if not isinstance(backgrounds, list):
            errors.append("backgrounds must be a list")
        else:
            for i, bg in enumerate(backgrounds):
                if not isinstance(bg, dict):
                    errors.append(f"backgrounds[{i}] must be a dict")
                elif "type" not in bg:
                    errors.append(f"backgrounds[{i}] missing 'type' field")

    # Validate agent_actions
    if "agent_actions" in config:
        actions = config["agent_actions"]
        if not isinstance(actions, list):
            errors.append("agent_actions must be a list")

    # Check for deprecated fields
    deprecated_fields = ["robot_ip"]  # Add more as needed
    for field in deprecated_fields:
        if field in config:
            warnings.append(f"Field '{field}' is deprecated")

    return errors, warnings


def main():
    if len(sys.argv) < 2:
        print("Usage: python validate_config.py <config_file>")
        print("\nExample: python validate_config.py config/spot.json5")
        sys.exit(1)

    config_path = Path(sys.argv[1])

    if not config_path.exists():
        print(f"❌ Error: Config file not found: {config_path}")
        sys.exit(1)

    print(f"Validating: {config_path}")
    print("-" * 60)

    # Check file format
    valid, msg = validate_json5(config_path)
    if not valid:
        print(f"❌ {msg}")
        sys.exit(1)
    print(f"✅ {msg}")

    # Load config
    try:
        with open(config_path, 'r') as f:
            config = json.loads(f.read().replace(',\n}', '\n}'))  # Basic JSON5 cleanup
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        sys.exit(1)

    # Validate structure
    errors, warnings = validate_config_structure(config)

    # Print results
    if errors:
        print(f"\n❌ Errors found ({len(errors)}):")
        for error in errors:
            print(f"  • {error}")

    if warnings:
        print(f"\n⚠️  Warnings ({len(warnings)}):")
        for warning in warnings:
            print(f"  • {warning}")

    if not errors and not warnings:
        print("\n✅ Configuration is valid!")

    print("-" * 60)

    # Exit with appropriate code
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
