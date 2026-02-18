## Summary

This PR consolidates **9 bug fixes** cherry-picked from upstream development, addressing crash prevention, correctness issues, and logging improvements across the robot runtime.

### Fixes Included

| Commit | Issue | Fix |
|--------|-------|-----|
| `dca89cdf` | #2323 | UbTTS: Fix base_url f-string evaluating FieldInfo instead of robot_ip |
| `d29c96b3` | #2322 | DualLLM: Use correct LLMConfig subclass instead of base class |
| `b91decd5` | #2321 | wallet_ethereum: Replace init crash with warning log |
| `8b47940b` | #2320 | Simulator: Enforce ABC contract and reject abstract plugins |
| `9ee3b968` | #2319 | TurtleBot4 RPLidar: Fix blanked angle filtering logic |
| `0115081f` | #2318 | Version: Narrow try-except scope to preserve error context |
| `e8d61052` | #2317 | SleepTicker: Remove no-op pass, add duration to logging |
| `25d10eff` | #2315 | Docker: Enable reproducible builds with uv.lock |
| `287ad373` | #2314 | Function Schemas: Fix type checking (isinstance → is) |

### Key Changes

1. **src/actions/speak/connector/ub_tts.py**: Use `@model_validator(mode='after')` to build base_url after field values resolve
2. **src/llm/__init__.py**: Add `get_llm_config_class()` helper for config subclass discovery
3. **src/llm/plugins/dual_llm.py**: Call `get_llm_config_class()` instead of using base `LLMConfig`
4. **src/llm/function_schemas.py**: Fix type checking guard from `isinstance` to `is`
5. **src/providers/io_provider.py**: Add timestamp warning logging for missing keys
6. **src/providers/sleep_ticker_provider.py**: Replace pass with informative logging
7. **src/providers/turtlebot4_rplidar_provider.py**: Replace nested loop `any()` pattern for blanked angles
8. **src/runtime/version.py**: Narrow try-except scope to preserve actual error messages
9. **src/simulators/base.py**: Add `@abstractmethod` decorators to enforce ABC contract
10. **src/simulators/__init__.py**: Reject abstract simulator subclasses
11. **Dockerfile**: Pin uv version, use `uv sync --frozen` for reproducible builds
12. **.dockerignore**: Include uv.lock in build context

### Testing
- All existing unit tests pass
- New tests added for abstract class rejection
- Manual verification for bug reproduction cases

### Related Issues
Closes #2323, #2322, #2321, #2320, #2319, #2318, #2317, #2315, #2314
