"""Tests for FunctionGenerator in function_call_provider."""

import inspect
from typing import Optional

import pytest

from providers.function_call_provider import FunctionGenerator


class TestFunctionGenerator:
    """Test cases for FunctionGenerator class."""

    def test_extract_function_schema_with_required_parameter(self):
        """Test that required parameters are added to required list."""

        def test_method(required_param: str):
            """Test method with required parameter."""
            pass

        schema = FunctionGenerator.extract_function_schema(test_method)

        assert "function" in schema
        assert "parameters" in schema["function"]
        assert "required" in schema["function"]["parameters"]
        assert "required_param" in schema["function"]["parameters"]["required"]
        assert len(schema["function"]["parameters"]["required"]) == 1

    def test_extract_function_schema_with_optional_parameter(self):
        """Test that optional parameters are NOT added to required list."""

        def test_method(optional_param: str = "default"):
            """Test method with optional parameter."""
            pass

        schema = FunctionGenerator.extract_function_schema(test_method)

        assert "function" in schema
        assert "parameters" in schema["function"]
        assert "required" in schema["function"]["parameters"]
        assert "optional_param" not in schema["function"]["parameters"]["required"]
        assert len(schema["function"]["parameters"]["required"]) == 0

    def test_extract_function_schema_with_empty_string_default(self):
        """Test that empty string default parameters are NOT added to required list."""

        def test_method(optional_param: str = ""):
            """Test method with empty string default."""
            pass

        schema = FunctionGenerator.extract_function_schema(test_method)

        assert "function" in schema
        assert "parameters" in schema["function"]
        assert "required" in schema["function"]["parameters"]
        assert "optional_param" not in schema["function"]["parameters"]["required"]
        assert len(schema["function"]["parameters"]["required"]) == 0
        
        # Check that description indicates it's optional
        properties = schema["function"]["parameters"]["properties"]
        assert "optional_param" in properties
        assert "(optional - can be empty string)" in properties["optional_param"]["description"]

    def test_extract_function_schema_with_mixed_parameters(self):
        """Test function with both required and optional parameters."""

        def test_method(
            required_param: str, optional_param: str = "default", empty_default: str = ""
        ):
            """Test method with mixed parameters."""
            pass

        schema = FunctionGenerator.extract_function_schema(test_method)

        assert "function" in schema
        assert "parameters" in schema["function"]
        assert "required" in schema["function"]["parameters"]
        
        required = schema["function"]["parameters"]["required"]
        assert "required_param" in required
        assert "optional_param" not in required
        assert "empty_default" not in required
        assert len(required) == 1

    def test_extract_function_schema_with_optional_type_hint(self):
        """Test function with Optional type hint."""

        def test_method(optional_param: Optional[str] = None):
            """Test method with Optional type."""
            pass

        schema = FunctionGenerator.extract_function_schema(test_method)

        assert "function" in schema
        assert "parameters" in schema["function"]
        assert "required" in schema["function"]["parameters"]
        assert "optional_param" not in schema["function"]["parameters"]["required"]
