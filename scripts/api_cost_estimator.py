#!/usr/bin/env python3
"""
API Cost Estimator Script

Estimates OpenMind API costs based on configuration and usage patterns.
Usage: python scripts/api_cost_estimator.py [config_file]
"""

import json
import sys
from pathlib import Path


# Approximate costs (adjust based on actual OpenMind pricing)
COST_PER_1M_TOKENS = {
    "input": 2.0,   # USD per million input tokens
    "output": 6.0,  # USD per million output tokens
}


def estimate_tokens_per_request(config):
    """Estimate tokens per OM1 request based on config."""
    # Base tokens for system prompt + context
    base_tokens = 500

    # Add tokens for inputs
    agent_inputs = config.get("agent_inputs", [])
    input_tokens = len(agent_inputs) * 200  # Approximate per input

    # Add tokens for LLM history
    cortex_llm = config.get("cortex_llm", {})
    history_length = cortex_llm.get("config", {}).get("history_length", 3)
    history_tokens = history_length * 300

    total_input = base_tokens + input_tokens + history_tokens
    total_output = 100  # Approximate output tokens

    return total_input, total_output


def estimate_requests_per_hour(hertz):
    """Estimate number of API requests per hour."""
    return hertz * 3600


def calculate_hourly_cost(input_tokens, output_tokens, requests_per_hour):
    """Calculate estimated hourly cost."""
    input_cost = (input_tokens * requests_per_hour / 1_000_000) * COST_PER_1M_TOKENS["input"]
    output_cost = (output_tokens * requests_per_hour / 1_000_000) * COST_PER_1M_TOKENS["output"]
    total = input_cost + output_cost
    return total, input_cost, output_cost


def format_cost(cents):
    """Format cost in USD."""
    return f"${cents:.4f}"


def main():
    if len(sys.argv) < 2:
        print("Usage: python api_cost_estimator.py <config_file>")
        print("\nExample: python api_cost_estimator.py config/spot.json5")
        sys.exit(1)

    config_path = Path(sys.argv[1])

    if not config_path.exists():
        print(f"❌ Error: Config file not found: {config_path}")
        sys.exit(1)

    # Load config
    try:
        with open(config_path, 'r') as f:
            content = f.read()
            # Basic JSON5 cleanup
            content = content.replace(',\n}', '\n}').replace(',\n]', '\n]')
            config = json.loads(content)
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        sys.exit(1)

    hertz = config.get("hertz", 1.0)
    print(f"📊 API Cost Estimation for: {config_path.name}")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Hertz: {hertz} Hz")
    print(f"  Requests per hour: {estimate_requests_per_hour(hertz):,.0f}")
    print(f"  Requests per day: {estimate_requests_per_hour(hertz) * 24:,.0f}")

    # Estimate tokens
    input_tokens, output_tokens = estimate_tokens_per_request(config)
    print(f"\nToken Estimation (per request):")
    print(f"  Input: ~{input_tokens:,} tokens")
    print(f"  Output: ~{output_tokens:,} tokens")
    print(f"  Total: ~{input_tokens + output_tokens:,} tokens")

    # Calculate costs
    requests_per_hour = estimate_requests_per_hour(hertz)
    hourly_total, hourly_input, hourly_output = calculate_hourly_cost(
        input_tokens, output_tokens, requests_per_hour
    )

    daily_total = hourly_total * 24
    monthly_total = daily_total * 30

    print(f"\n💰 Cost Estimates:")
    print(f"  Per hour:  {format_cost(hourly_total)}")
    print(f"    ├─ Input:  {format_cost(hourly_input)}")
    print(f"    └─ Output: {format_cost(hourly_output)}")
    print(f"\n  Per day:   {format_cost(daily_total)}")
    print(f"  Per month: {format_cost(monthly_total)}")

    # Optimization suggestions
    print(f"\n💡 Cost Optimization Tips:")
    if hertz > 1.0:
        print(f"  • Reduce hertz from {hertz} to 0.1:")
        print(f"    Estimated savings: {format_cost(monthly_total * 0.9)} / month")
    else:
        print(f"  ✓ Hertz is already optimized")

    cortex_config = config.get("cortex_llm", {}).get("config", {})
    history_length = cortex_config.get("history_length", 3)
    if history_length > 3:
        print(f"  • Reduce history_length from {history_length} to 3:")
        reduction = (history_length - 3) / history_length
        print(f"    Estimated savings: {format_cost(monthly_total * reduction)} / month")

    agent_inputs = config.get("agent_inputs", [])
    if len(agent_inputs) > 2:
        print(f"  • Consider reducing active inputs ({len(agent_inputs)} configured)")

    backgrounds = config.get("backgrounds", [])
    for bg in backgrounds:
        if bg.get("type") == "TeleopsConnection":
            print(f"  ✓ TeleopsConnection uses minimal API (status updates only)")

    print("\n" + "=" * 60)
    print("Note: These are estimates based on typical usage patterns.")
    print("Actual costs may vary based on:")
    print("  • Actual prompt and response lengths")
    print("  • Specific model pricing")
    print("  • VLM input processing")
    print("  • Background plugin activity")


if __name__ == "__main__":
    main()
