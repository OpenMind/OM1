# Apple Silicon (M1/M2) Performance & Setup Optimization

This guide helps Apple Silicon users reduce latency and avoid context bloat
when running OM1 locally.

## Why Latency Happens on Apple Silicon
- Large default context sizes
- Unoptimized model settings
- Docker running with limited resources
- Background Rosetta (x86) processes

## Recommended Baseline Settings
- Use native arm64 Python
- Keep context size minimal for local testing
- Close unused agents/modules
- Ensure Docker Desktop is running natively (Apple Silicon build)

## Reducing Context Bloat
- Avoid long-running sessions during testing
- Restart agents between experiments
- Keep prompts concise
- Disable unused integrations

## Common Issues & Fixes

### High latency on first run
- This is expected during initial model warm-up

### Increasing memory usage over time
- Restart the process to clear accumulated context

### Docker consuming excessive resources
- Limit CPU and memory in Docker Desktop settings

---

These recommendations are intended to improve the local developer experience
on Apple Silicon devices.
