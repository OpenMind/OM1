"""
Unit tests for the JSON parser utility.

Tests cover:
1. Normal JSON parsing (backward compatibility)
2. Markdown-wrapped JSON extraction
3. Embedded JSON in text
4. Edge cases and error handling
5. Function argument parsing
"""

import json
import pytest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from utils.json_parser import (
    extract_json_string,
    safe_json_loads,
    safe_parse_function_args,
)


class TestExtractJsonString:
    """Tests for extract_json_string function."""

    def test_plain_json_object(self):
        """Plain JSON should be returned as-is."""
        text = '{"action": "move", "value": "forward"}'
        result = extract_json_string(text)
        assert result == text

    def test_markdown_json_block(self):
        """Should extract JSON from ```json ... ``` blocks."""
        text = '```json\n{"action": "move"}\n```'
        result = extract_json_string(text)
        assert result == '{"action": "move"}'

    def test_markdown_block_no_language(self):
        """Should extract JSON from ``` ... ``` blocks without language tag."""
        text = '```\n{"key": "value"}\n```'
        result = extract_json_string(text)
        assert result == '{"key": "value"}'

    def test_markdown_block_uppercase_json(self):
        """Should handle uppercase JSON language tag."""
        text = '```JSON\n{"key": "value"}\n```'
        result = extract_json_string(text)
        assert result == '{"key": "value"}'

    def test_json_with_surrounding_text(self):
        """Should extract JSON object from surrounding text."""
        text = 'Here is the response: {"action": "stop"} as requested.'
        result = extract_json_string(text)
        assert result == '{"action": "stop"}'

    def test_json_array(self):
        """Should extract JSON arrays."""
        text = 'The list is: [1, 2, 3]'
        result = extract_json_string(text)
        assert result == '[1, 2, 3]'

    def test_empty_string(self):
        """Should return None for empty string."""
        assert extract_json_string('') is None
        assert extract_json_string('   ') is None

    def test_none_input(self):
        """Should return None for None input."""
        assert extract_json_string(None) is None

    def test_non_string_input(self):
        """Should return None for non-string input."""
        assert extract_json_string(123) is None
        assert extract_json_string(['list']) is None

    def test_nested_json(self):
        """Should handle nested JSON objects."""
        text = '```json\n{"outer": {"inner": "value"}}\n```'
        result = extract_json_string(text)
        assert result == '{"outer": {"inner": "value"}}'

    def test_json_with_newlines(self):
        """Should handle JSON with internal newlines."""
        text = '''```json
{
    "action": "move",
    "value": "forward"
}
```'''
        result = extract_json_string(text)
        assert '"action": "move"' in result
        assert '"value": "forward"' in result


class TestSafeJsonLoads:
    """Tests for safe_json_loads function."""

    def test_valid_json_direct(self):
        """Should parse valid JSON directly."""
        text = '{"action": "move", "value": "forward"}'
        result = safe_json_loads(text)
        assert result == {"action": "move", "value": "forward"}

    def test_valid_json_array(self):
        """Should parse valid JSON arrays."""
        text = '[1, 2, 3]'
        result = safe_json_loads(text)
        assert result == [1, 2, 3]

    def test_markdown_wrapped_json(self):
        """Should parse JSON from Markdown code blocks."""
        text = '```json\n{"action": "stop"}\n```'
        result = safe_json_loads(text)
        assert result == {"action": "stop"}

    def test_markdown_with_explanation(self):
        """Should handle Markdown with surrounding explanation."""
        text = '''Here is the JSON response:
```json
{"status": "ok", "code": 200}
```
This indicates success.'''
        result = safe_json_loads(text)
        assert result == {"status": "ok", "code": 200}

    def test_invalid_json_returns_default(self):
        """Should return default for invalid JSON."""
        text = 'not valid json at all'
        result = safe_json_loads(text, default={"fallback": True})
        assert result == {"fallback": True}

    def test_invalid_json_returns_none(self):
        """Should return None by default for invalid JSON."""
        text = 'not valid json'
        result = safe_json_loads(text)
        assert result is None

    def test_empty_string_returns_default(self):
        """Should return default for empty string."""
        result = safe_json_loads('', default={})
        assert result == {}

    def test_none_input_returns_default(self):
        """Should return default for None input."""
        result = safe_json_loads(None, default=[])
        assert result == []

    def test_whitespace_only_returns_default(self):
        """Should return default for whitespace-only input."""
        result = safe_json_loads('   \n\t  ', default="default")
        assert result == "default"

    def test_json_embedded_in_text(self):
        """Should extract and parse JSON embedded in text."""
        text = 'The robot should do {"action": "dance"} now.'
        result = safe_json_loads(text)
        assert result == {"action": "dance"}

    def test_complex_nested_json(self):
        """Should handle complex nested structures."""
        data = {
            "actions": [
                {"type": "move", "params": {"direction": "forward", "speed": 1.5}},
                {"type": "speak", "params": {"text": "Hello!"}}
            ],
            "metadata": {"timestamp": 12345}
        }
        text = f'```json\n{json.dumps(data)}\n```'
        result = safe_json_loads(text)
        assert result == data

    def test_json_with_unicode(self):
        """Should handle JSON with unicode characters."""
        text = '{"message": "你好世界", "emoji": "🤖"}'
        result = safe_json_loads(text)
        assert result == {"message": "你好世界", "emoji": "🤖"}

    def test_json_with_escaped_characters(self):
        """Should handle JSON with escaped characters."""
        text = '{"text": "line1\\nline2", "path": "C:\\\\Users"}'
        result = safe_json_loads(text)
        assert result["text"] == "line1\nline2"
        assert result["path"] == "C:\\Users"

    def test_log_errors_false_suppresses_logging(self, caplog):
        """Should not log when log_errors=False."""
        import logging
        with caplog.at_level(logging.DEBUG):
            safe_json_loads('invalid', log_errors=False)
        assert len(caplog.records) == 0

    def test_context_in_error_message(self, caplog):
        """Should include context in error messages."""
        import logging
        with caplog.at_level(logging.ERROR):
            safe_json_loads('invalid', context="test_context")
        assert any("test_context" in r.message for r in caplog.records)


class TestSafeParseunctionArgs:
    """Tests for safe_parse_function_args function."""

    def test_valid_function_args(self):
        """Should parse valid function arguments."""
        args = '{"action": "move_forward", "speed": 1.0}'
        result = safe_parse_function_args(args)
        assert result == {"action": "move_forward", "speed": 1.0}

    def test_empty_args(self):
        """Should return empty dict for empty args."""
        result = safe_parse_function_args('{}')
        assert result == {}

    def test_invalid_args_returns_default(self):
        """Should return default for invalid args."""
        result = safe_parse_function_args('not json')
        assert result == {}

    def test_custom_default(self):
        """Should use custom default."""
        result = safe_parse_function_args('invalid', default={"default": True})
        assert result == {"default": True}

    def test_non_dict_result_returns_default(self):
        """Should return default if parsed result is not a dict."""
        result = safe_parse_function_args('[1, 2, 3]')
        assert result == {}

    def test_markdown_wrapped_args(self):
        """Should handle Markdown-wrapped function args."""
        args = '```json\n{"action": "stop"}\n```'
        result = safe_parse_function_args(args)
        assert result == {"action": "stop"}


class TestRealWorldScenarios:
    """Tests based on real-world LLM response patterns."""

    def test_openai_style_response(self):
        """Test typical OpenAI function call arguments."""
        args = '{"action":"move forward","distance":2.5}'
        result = safe_json_loads(args)
        assert result["action"] == "move forward"
        assert result["distance"] == 2.5

    def test_anthropic_style_response(self):
        """Test response with Markdown formatting (common in Claude)."""
        text = '''I'll execute the move action for you.

```json
{
  "action": "move",
  "direction": "forward",
  "units": 3
}
```

This will move the robot forward by 3 units.'''
        result = safe_json_loads(text)
        assert result["action"] == "move"
        assert result["direction"] == "forward"
        assert result["units"] == 3

    def test_qwen_tool_call_format(self):
        """Test Qwen-style tool call response."""
        # Qwen sometimes returns JSON without proper formatting
        text = 'Based on your request, here is the action: {"name": "move", "arguments": {"direction": "left"}}'
        result = safe_json_loads(text)
        assert result["name"] == "move"

    def test_vlm_response_with_description(self):
        """Test VLM response with scene description before JSON."""
        text = '''I can see a person standing in front of the robot.
{"objects": ["person", "chair", "table"], "person_distance": 2.3, "should_stop": true}'''
        result = safe_json_loads(text)
        assert "person" in result["objects"]
        assert result["should_stop"] is True

    def test_malformed_but_recoverable(self):
        """Test slightly malformed but recoverable JSON."""
        # Extra whitespace and newlines
        text = '''
        {
            "action"  :  "wait"  ,
            "duration" : 5
        }
        '''
        result = safe_json_loads(text)
        assert result["action"] == "wait"
        assert result["duration"] == 5


class TestBackwardCompatibility:
    """Tests to ensure backward compatibility with existing json.loads usage."""

    def test_all_json_types(self):
        """Should handle all JSON types correctly."""
        assert safe_json_loads('"string"') == "string"
        assert safe_json_loads('123') == 123
        assert safe_json_loads('12.5') == 12.5
        assert safe_json_loads('true') is True
        assert safe_json_loads('false') is False
        assert safe_json_loads('null') is None
        assert safe_json_loads('[]') == []
        assert safe_json_loads('{}') == {}

    def test_preserves_original_behavior_for_valid_json(self):
        """Verify that valid JSON is parsed identically to json.loads."""
        test_cases = [
            '{"key": "value"}',
            '[1, 2, 3]',
            '{"nested": {"a": 1}}',
            '{"array": [1, "two", null]}',
        ]
        for test in test_cases:
            assert safe_json_loads(test) == json.loads(test)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
