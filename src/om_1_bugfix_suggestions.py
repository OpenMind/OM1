# 🧾 OM1 Project – Example Pull Request (PR) Template
# This file contains a ready-to-use PR description you can paste into GitHub.
# Branch name suggestion: bugfix/logger-async

---

## 🧠 Title:
**Fix: Add logger, async improvements & better error handling**

---

## 📌 Description
This PR addresses multiple issues found during a source code review of the `src/` folder.  
The goal is to make the codebase more stable, maintainable, and easier to debug.

### ✅ Changes:
- Added centralized **logging infrastructure** to replace `print()` statements.
- Wrapped **blocking calls inside async functions** with `asyncio.to_thread()` to avoid freezing the event loop.
- Added **error handling for ROS2 connection failures** with fallback to simulation mode.
- Replaced **hardcoded paths** with environment-variable based paths using `Pathlib`.
- Added **type hints** and improved static analysis coverage.
- Modularized utility functions into a new `utils/` folder to reduce code repetition.
- Improved **API key management** with environment variables.
- Added a basic **GitHub Actions CI workflow** for linting and testing.

---

## 🐞 Related Issues
- Fixes: Connection failures without handling
- Fixes: Async blocking
- Fixes: Hardcoded paths portability problem

---

## 🧪 How to Test
1. Create a `.env` file with `OM1_API_KEY`.
2. Run the main agent: `uv run src/run.py spot`
3. Verify logs are printed with timestamps (no more plain `print()`).
4. Simulate a ROS2 failure → confirm the system falls back to simulation mode.
5. Run `pytest` and `flake8` locally to ensure all checks pass.

---

## 🧰 Additional Notes
- Added comments and docstrings for new helper functions.
- This PR improves code structure but **does not change core logic**.
- Future work: extend logger with file storage and log levels per module.

---

## 📸 Screenshots / Logs (Optional)
```
[2025-10-19 13:42:10] INFO: System started
[2025-10-19 13:42:11] ERROR: [ROS2] Connection failed: Connection refused
[2025-10-19 13:42:11] INFO: Switching to simulation mode...
```

---

## 🚀 Checklist
- [x] Code builds and runs
- [x] No breaking changes
- [x] Code is linted and type-checked
- [x] Tests pass
- [x] Documentation updated

---

## 🔀 Merge Strategy
Squash & merge (recommended)

---

## 📅 Versioning
Patch release: `v1.0.0-beta.4`

---

👤 Author: [Your Name]
📅 Date: 2025-10-19
