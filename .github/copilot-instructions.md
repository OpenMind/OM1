# Copilot Instructions

## Tooling / Dependencies
- We use **uv** for Python dependency management and execution.
- Dependencies are defined in `pyproject.toml` (do not introduce `requirements.txt` unless explicitly requested).
- Use these commands in docs/examples:
  - Install/sync: `uv sync`
  - Add deps: `uv add <package>`
  - Run commands: `uv run <command>` (e.g., `uv run pytest`)
- If adding a new dependency, prefer a well-maintained package and keep the dependency list minimal.
- Assume a lockfile is used (e.g., `uv.lock`) — do not suggest deleting or bypassing it.

## Coding Style
- Use Python 3.10
- Prefer async/await
- Use type hints everywhere
- Follow PEP8 and Black formatting

## Architecture
- Use modular design
- Separate API, service, and data layers
- No business logic inside routes

## Testing
- Use pytest
- Add unit tests for all new functions
- Aim for 90% code coverage
- Prefer to use patch and mock for external dependencies
- For patches, patch it using `with (patch("full.path.to.ClassOrFunction1") as mock1, patch("full.path.to.ClassOrFunction2") as mock2):` to ensure the patch is applied in the correct scope.

## Comments
- Write numpy style docstrings
- Avoid obvious comments
