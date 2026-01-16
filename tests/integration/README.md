# Integration Tests

This directory contains integration tests for the entire system, from input (images, ETH transactions, and etc.) to output (robot commands).

## Test Case Approach

Instead of a monolithic test runner, we use a configuration-based approach where each test case is defined in its own JSON5 configuration file. This allows:

1. Testing different VLM implementations independently
2. Using different API keys for different tests
3. Easier debugging by separating tests

## Directory Structure

- `data/`: Test data files
  - `test_cases/`: Individual test case configurations
  - `images/`: Test images for VLM testing

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
TEST_CASE="coco_indoor_detection" uv run pytest -m "integration" tests/integration/test_case_runner.py::test_specific_case -v
```

## Creating New Test Cases

1. Create a new JSON5 file in `data/test_cases/` following the format in existing files
2. Add any necessary test images to `data/images/`
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
  "system_prompt_examples": "...'",
  "agent_inputs": [...],
  "cortex_llm": {...},
  "agent_actions": [
    {
      "name": "move",
      "llm_label": "move",
      "connector": "ros2"
    },
    {
      "name": "speak",
      "llm_label": "speak",
      "connector": "ros2"
    },
    {
      "name": "face",
      "llm_label": "emotion",
      "connector": "ros2"
    }
  ],
  "api_key": "openmind_free",

  // Input data
  "input": {
    "images": ["../vlm_test/image1.png", "../vlm_test/image2.png"],
  },

  // Expected output
  "expected": {
    "movement": "turn_left | turn_right | move_forward",
    "keywords": ["person", "furniture", "indoor"],
    "minimum_score": 0.7 // Minimum score required to pass
  }
}
```
