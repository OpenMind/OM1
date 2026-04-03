---
name: weather-report
description: "Query weather for New York and San Francisco and report the results"
requires_tools:
  - mcp_weather_get-forecast
max_rounds: 5
priority: 10
---

# Weather Report Skill

Query weather for New York and San Francisco, then report the results.

## Steps

1. Call `mcp_weather_get-forecast` for **New York** (latitude: 40.7128, longitude: -74.0060).
2. Call `mcp_weather_get-forecast` for **San Francisco** (latitude: 37.7749, longitude: -122.4194).
3. Use `speak` to report both forecasts in a concise, friendly summary.
