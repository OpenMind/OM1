# Integration Tests

This directory contains integration tests for the entire system, from input (images, audio, sensor data) to output (robot commands).

## Test Case Approach

Instead of a monolithic test runner, we use a configuration-based approach where each test case is defined in its own JSON5 configuration file. This allows:

1. Testing different input types independently (VLM, ASR, battery, odometry, GPS)
2. Testing multi-input combinations (VLM + ASR, VLM + battery, etc.)
3. Testing mode transitions (input-triggered, time-based, cooldown)
4. Using different API keys for different tests
5. Easier debugging by separating tests

## Directory Structure

- `data/`: Test data files
  - `test_cases/`: Individual test case configurations (JSON5)
  - `images/`: Test images for VLM testing
  - `lidar/`: LIDAR scan data (JSON)
  - `asr/`: ASR text data (JSON) for speech recognition tests
  - `state/`: Sensor state data (JSON) for battery, odometry
  - `gps/`: GPS location data (JSON)
- `mock_inputs/`: Mock implementations of real input plugins
  - `data_providers/`: Singleton providers for mock data (image, lidar, text, state)

## Supported Input Types

| Input Type | Mock Class | Data Provider | Data Format |
|-----------|-----------|---------------|-------------|
| VLM (OpenAI, Gemini, COCO, Vila) | MockVLM_* | MockImageProvider | JPEG/PNG images |
| RPLidar | MockUnitreeGo2RPLidar | MockLidarProvider | JSON scan arrays |
| Google ASR | MockGoogleASR | MockTextProvider | JSON with "text" field |
| Battery | MockUnitreeGo2Battery | MockStateProvider | JSON with percent/voltage/amperes |
| Odometry | MockUnitreeGo2Odom | MockStateProvider | JSON with x/y/yaw/moving |
| GPS | MockGps | MockStateProvider | JSON with gps_lat/gps_lon/gps_alt/gps_qua |

## Running Tests

### Running All Integration Tests

```bash
uv run pytest -m "integration" tests/integration/test_case_runner.py -v
```

### Running All Integration Tests with logging

```bash
uv run pytest -m "integration" -s --log-cli-level=INFO tests/integration/test_case_runner.py -v
```

### Running a Specific Test Case

```bash
TEST_CASE="asr_greeting_test" uv run pytest -m "integration" tests/integration/test_case_runner.py::test_specific_case -v
```

### Running Mode Transition Tests

```bash
# All mode transition tests
uv run pytest -m "integration" tests/integration/test_case_runner.py -v -k "mode"

# Input-triggered transition only
uv run pytest -m "integration" tests/integration/test_case_runner.py -v -k "test_mode_transition"

# Time-based transition only
uv run pytest -m "integration" tests/integration/test_case_runner.py -v -k "test_time_based"

# Cooldown test only
uv run pytest -m "integration" tests/integration/test_case_runner.py -v -k "test_cooldown"
```

## Creating New Test Cases

1. Create a new JSON5 file in `data/test_cases/` following the format in existing files
2. Add any necessary test data to the appropriate `data/` subdirectory
3. Run your test case to verify it works correctly

### Test Case Format

```json5
{
  // Test case metadata
  "name": "test name",
  "description": "test description",
  "hertz": 1,
  "system_prompt_base": "...",
  "system_governance": "...",
  "system_prompt_examples": "...",
  "agent_inputs": [...],
  "cortex_llm": {...},
  "agent_actions": [...],
  "api_key": "openmind_free",

  // Input data - supports multiple types
  "input": {
    "images": ["../images/indoor_1.jpg"],       // VLM inputs
    "lidar": ["../lidar/sample_scan.json"],      // LIDAR inputs
    "asr": ["../asr/greeting.json"],             // ASR text inputs
    "battery": ["../state/battery_low.json"],    // Battery state
    "odometry": ["../state/odometry_moving.json"], // Odometry state
    "gps": ["../gps/outdoor_location.json"],     // GPS location
  },

  // Expected output
  "expected": {
    "movement": ["stand still", "sit"],       // Expected movement commands
    "keywords": ["person", "furniture"],       // Keywords in LLM prompt
    "emotion": ["happy", "curious"],           // Expected emotions
    "minimum_score": 0.5                       // Minimum score to pass (0.0-1.0)
  }
}
```

### Mode Transition Test Case Format

Mode transition tests use a multi-mode configuration with `modes` and `transition_rules` keys.
These are automatically detected and routed to the mode transition test runner.

```json5
{
  name: "mode_transition_test",
  description: "Test input-triggered mode transition",
  version: "v1.0.2",
  api_key: "openmind_free",
  default_mode: "calm",
  modes: {
    calm: {
      display_name: "Calm",
      description: "Normal calm mode",
      system_prompt_base: "You are a calm and relaxed robot.",
      hertz: 1,
      // timeout_seconds: 0.1,        // For time-based transitions
      agent_inputs: [{type: "GoogleASRInput", config: {}}],
      cortex_llm: {type: "OpenAILLM", config: {agent_name: "Buddy"}},
      agent_actions: [
        {name: "move", llm_label: "move", connector: "ros2"},
        {name: "speak", llm_label: "speak", connector: "ros2"},
        {name: "face", llm_label: "emotion", connector: "ros2"},
      ],
    },
    alert: {
      display_name: "Alert",
      description: "Alert mode for emergencies",
      system_prompt_base: "You are now in alert mode.",
      hertz: 1,
      agent_inputs: [{type: "GoogleASRInput", config: {}}],
      cortex_llm: {type: "OpenAILLM", config: {agent_name: "Buddy"}},
      agent_actions: [
        {name: "move", llm_label: "move", connector: "ros2"},
        {name: "speak", llm_label: "speak", connector: "ros2"},
        {name: "face", llm_label: "emotion", connector: "ros2"},
      ],
    },
  },
  transition_rules: [{
    from_mode: "calm",
    to_mode: "alert",
    transition_type: "input_triggered",    // or "time_based"
    trigger_keywords: ["danger", "emergency", "help"],
    priority: 1,
    cooldown_seconds: 0,                   // Set > 0 to test cooldown
  }],
  input: {asr: ["../asr/emergency.json"]},
  expected: {
    initial_mode: "calm",
    final_mode: "alert",
    // first_transition_mode: "alert",     // For cooldown tests
    // second_transition_mode: "calm",     // For cooldown tests
  },
}
```

**Supported transition types:**
- `input_triggered`: Transitions when ASR input contains trigger keywords
- `time_based`: Transitions after `timeout_seconds` expires on the source mode

### Data File Formats

**ASR data** (`data/asr/*.json`):
```json
{
  "metadata": {"description": "...", "language": "en"},
  "text": "Hello, how are you?"
}
```

**Battery data** (`data/state/battery_*.json`):
```json
{
  "metadata": {"description": "...", "sensor_type": "battery"},
  "data": {"percent": 15, "voltage": 22.1, "amperes": 0.5}
}
```

**Odometry data** (`data/state/odometry_*.json`):
```json
{
  "metadata": {"description": "...", "sensor_type": "odometry"},
  "data": {"x": 1.5, "y": 0.3, "yaw": 45.0, "moving": true, "body_attitude": "STANDING"}
}
```

**GPS data** (`data/gps/*.json`):
```json
{
  "metadata": {"description": "...", "sensor_type": "gps"},
  "data": {"gps_lat": 37.7749, "gps_lon": -122.4194, "gps_alt": 10.0, "gps_qua": 1}
}
```
