# OM1 Developer Notes

## Repository Layout
- `src/run.py` — Typer CLI entrypoint that loads configs, decides between single/multi-mode runtimes, and launches the async `CortexRuntime`.
- `src/runtime/` — scheduling core plus config loaders. `single_mode/` and `multi_mode/` parse JSON5 configs, instantiate sensors/actions/backgrounds, and manage hot reload.
- `src/actions`, `src/inputs`, `src/backgrounds`, `src/llm`, `src/simulators` — pluggable modules. Each folder exposes loader helpers that dynamically import interfaces/connectors defined under matching subpackages.
- `config/` — JSON5 agent specs (Spot, TurtleBot, Unitree, etc.), schemas, and mode definitions. Secrets (API keys, robot IPs) should never be committed here.
- `tests/` — pytest suites organized by domain (`runtime/`, `actions/`, `inputs/`, etc.) with async support configured in `pyproject.toml`.
- `docs/` and `mintlify/` — Markdown docs that power the public developer docs site.
- `system_hw_test/`, `gazebo/`, `cyclonedds/` — optional hardware/simulation helpers and DDS assets.
- Supporting files: `pyproject.toml` (uv/pytest config), `uv.lock`, `Dockerfile`, `docker-compose.yml`, `.pre-commit-config.yaml`, `CONTRIBUTING.md`, and `README.md`.

## Environment Setup
Existing instructions live in `README.md` (Getting Started) and `CONTRIBUTING.md`. Locally, run:

```bash
# 1) Create and activate the uv-managed virtualenv
uv venv
source .venv/bin/activate  # or .\.venv\Scripts\activate on Windows

# 2) Install dependencies defined in pyproject/uv.lock
uv pip install --upgrade pip
uv pip install -e .  # grabs OM1 plus extras pinned in uv.lock

# 3) Initialize submodules once after cloning
git submodule update --init

# 4) (Optional) Install native deps mentioned in README
brew install portaudio ffmpeg  # macOS
# or apt packages listed in README for Linux
```

Run sanity checks before hacking:

```bash
# Ensure formatting/test infra is reachable
uv run python -m compileall src
uv run pytest tests/runtime/single_mode/test_config.py
```

Swap `python -m compileall` for quick import validation if you do not want to boot the whole agent yet.

### Native dependency troubleshooting
- **`pyaudio` missing `portaudio.h`** — install PortAudio headers first (`brew install portaudio` on macOS, `sudo apt install portaudio19-dev` on Ubuntu). If headers live outside default paths, export `CFLAGS`/`LDFLAGS` (for Homebrew on Apple Silicon: `export CFLAGS="-I/opt/homebrew/include"` and `export LDFLAGS="-L/opt/homebrew/lib"`). Rerun `uv pip install -e .`.
- **`py-sr25519-bindings` complains about cargo/maturin** — install Rust tooling so `cargo` is on your `PATH` (`brew install rust` or `curl https://sh.rustup.rs -sSf | sh && source $HOME/.cargo/env`). Then rerun the uv install.
- **Suppressing noisy Homebrew env hints** — if brew repeatedly prints “Hide these hints with HOMEBREW_NO_ENV_HINTS=1”, set `export HOMEBREW_NO_ENV_HINTS=1` in your shell profile.

## Configuration & Secrets
- Primary configs live in `config/*.json5`. The runtime automatically falls back to env vars when `robot_ip` or `api_key` are blank or set to the `openmind_free` placeholder.
- Store secrets in `.env` (new `.env.example` template shows the required keys). `.env` is gitignored—copy the template, edit locally, and never commit actual keys.
- Use `config/spot.example.json5` as the base for personal agent configs. Copy it to `config/spot.json5` (or another filename) without committing secrets. Leave `"api_key": ""` to force the loader to pull `OM_API_KEY` from `.env`.
- Files that should stay local only: `.env`, customized `config/*.json5` with real keys, and any generated runtime state (`config/memory/.runtime.json5`).

## How to Run Locally
1. Load environment variables: `cp .env.example .env` and fill placeholders, then `source .venv/bin/activate`.
2. Copy `config/spot.example.json5` to `config/spot.json5` (or create another config) and adjust sensors/actions.
3. Launch an agent with uv + Typer (the default command now accepts the config name directly):
   - `uv run src/run.py spot` — runs `config/spot.json5`.
   - `uv run src/run.py turtlebot4` — TurtleBot4-specific config (requires Zenoh/LiDAR deps).
   - `uv run src/run.py quadruped_sim` — Gazebo quadruped sim, helpful without hardware.
4. Visit http://localhost:8000 if `WebSim` is enabled to observe commands in the browser.
5. Stop the agent with `Ctrl+C` in the terminal once you gather the logs/screenshots you need.
6. Use `--help` for extra options: `uv run src/run.py --help`.

### Example Agents
| Agent config | Command | Description |
| --- | --- | --- |
| `spot` | `uv run src/run.py spot` | Webcam-based dog avatar that maps VLM captions to ROS2 passthrough movement, speech, and facial expressions displayed in WebSim. |
| `turtlebot4` | `uv run src/run.py turtlebot4` | TurtleBot 4 agent combining RPLidar, battery input, Google ASR, and ElevenLabs TTS for autonomous navigation with Zenoh actions. |
| `unitree_go2_basic` | `uv run src/run.py unitree_go2_basic` | Connects to a Unitree Go2 over Ethernet, streams VLM/ASR inputs, and emits movement + TTS actions via specialized connectors. |
| `quadruped_sim` | `uv run src/run.py quadruped_sim` | Gazebo simulation agent consuming ASR + VLMVilaGazebo input and publishing commands through the `move_sim` action connector. |
| `twitter` | `uv run src/run.py twitter` | Social-media focused agent configuration (API key required) for ingesting timelines and producing outward actions. |

If you add your own config (e.g., `config/my_robot.json5`), invoke it with `uv run src/run.py my_robot`.

## Runtime Overview
1. **Entry point**: `src/run.py` uses Typer to parse CLI args, resolves the correct config file (`setup_config_file`), loads environment variables via `python-dotenv`, and decides between `CortexRuntime` (single mode) or `ModeCortexRuntime`.
2. **Config flow**: `runtime/single_mode/config.load_config` reads JSON5, normalizes metadata (API key, URID, robot IP) by merging `.env` values, and instantiates Python objects through module-specific loader helpers:
   - `inputs.load_input` -> `SensorConfig` -> `Sensor` subclass from `src/inputs/<name>`.
   - `actions.load_action` -> `ActionConfig/Connector` in `src/actions/<name>`.
   - `simulators.load_simulator`, `backgrounds.load_background`, `llm.load_llm`.
   The resulting `RuntimeConfig` aggregates these components plus scheduling metadata.
3. **Runtime loop**: `runtime/single_mode/cortex.CortexRuntime` coordinates sensor polling, LLM invocation, fusing (`src/fuser`), and action dispatch. Multi-mode runtime extends this to swap configs at runtime.
4. **Extensibility hooks**:
   - **New sensors/tools**: create a module under `src/inputs/<NewSensor>` implementing `Sensor` + `SensorConfig`, then reference it inside a config’s `agent_inputs`.
   - **Actions/connectors**: define interface + connector inside `src/actions/<action_name>/` and point your config’s `agent_actions` entry to it.
   - **LLMs / cognition**: implement `LLM` subclasses in `src/llm/` and wire them via `cortex_llm` in the config.
   - **Simulators & backgrounds**: extend `src/simulators` or `src/backgrounds` for telemetry and side tasks.

## Potential Contribution Areas
- **Quick bugfix/refactor ideas**
  - `runtime/single_mode/config.py` — add env-var interpolation for `${VAR}` placeholders to reduce manual blank-field handling.
  - `src/run.py` — extend `setup_config_file` error messages/tests so Typer `start` usage matches documentation (`uv run src/run.py spot` vs `start spot`).
  - `config/*.json5` schemas — enforce schema validation (currently optional) to catch typos before runtime.
- **Feature-sized efforts**
  - Implement a DepthAI or RealSense video input plugin under `src/inputs/` to feed richer perception streams.
  - Add an Anthropic/Deepseek LLM wrapper leveraging the existing `llm` interface for on-device vs. cloud fallback.
  - Create a Zenoh-based simulator connector for TurtleBot/Go2 that mirrors field hardware without full Gazebo.

## Testing & Verification
- Discoverable suites live under `tests/`; pyproject sets `addopts = -m "not integration"` so unit tests run without hardware.
- Common commands:
  - `uv run pytest` — runs all non-integration tests.
  - `uv run pytest tests/runtime` — focuses on runtime config/cortex behavior.
  - `uv run pytest -m integration` — opt-in hardware/Gazebo tests (ensure dependencies are ready).
  - `uv run ruff check src tests` and `uv run black --check src tests` if enforcing formatting locally.
- Missing coverage suggestion:
  - New file: `tests/runtime/test_run.py`.
  - Test case: mock filesystem to simulate absence of `.runtime.json5` and assert `setup_config_file(None)` raises `typer.Exit`, plus verify it copies memory configs when present. This guards the CLI bootstrap logic, which currently lacks direct tests.

## Draft PR Description
**Summary**
- Documented OM1 repo layout, setup steps, runtime architecture, contribution ideas, and testing commands in `DEV_NOTES.md`, including troubleshooting for native deps and updated CLI usage (`uv run src/run.py <config>`).
- Added `.env.example` so developers can safely template environment variables without exposing secrets.
- Added `config/spot.example.json5` with comments that explain how to keep API keys out of git while configuring the Spot agent.

**How to Run / Verify**
1. `cp .env.example .env` and fill `OM_API_KEY`.
2. `cp config/spot.example.json5 config/spot.json5` (leave `api_key` blank).
3. `uv venv && source .venv/bin/activate`.
4. `uv pip install -e .`.
5. `uv run src/run.py spot` — confirm the runtime picks up the API key from `.env`.

**Risks / Follow-ups**
- Example configs still require developers to avoid committing real secrets manually; consider ignoring `config/*.json5` overrides in future.
- CLI docs vs. Typer command names should be reconciled (see Potential Contribution Areas).
- Env-var substitution in configs is manual today; improving this would make example configs cleaner.
