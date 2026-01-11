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
from typing import Any, Dict, List

from providers.function_call_provider import FunctionGenerator, LLMFunction


class TestFunctionGeneratorBugs:

    def test_complex_type_conversion(self):
        """
        Test how python_type_to_json_schema handles complex types like List[int] or Dict[str, Any].
        Current implementation seems to ignore inner types.
        """
        schema_list_int = FunctionGenerator.python_type_to_json_schema(List[int])
        assert schema_list_int == {"type": "array", "items": {"type": "integer"}}

        FunctionGenerator.python_type_to_json_schema(Dict[str, int])
        assert FunctionGenerator.python_type_to_json_schema(int) == {"type": "integer"}

    def test_required_parameters_behavior(self):
        """
        Test if parameters with default values differ from required ones in the schema.
        OpenAI strict mode requires all parameters to be listed in 'required'.
        """

        class TestClass:
            @LLMFunction("test function")
            def test_method(self, req_param: int, opt_param: str = "default"):
                pass

        schema = FunctionGenerator.extract_function_schema(TestClass.test_method)
        params = schema["function"]["parameters"]
        required_list = params["required"]

        assert "req_param" in required_list
        assert "opt_param" in required_list

        props = params["properties"]
        assert "optional" in props["opt_param"].get("description", "")
        assert "default: default" in props["opt_param"].get("description", "")

    def test_nested_generic_types(self):
        """Test nested generic types like List[List[int]]."""
        schema = FunctionGenerator.python_type_to_json_schema(List[List[int]])
        expected = {
            "type": "array",
            "items": {"type": "array", "items": {"type": "integer"}},
        }
        assert schema == expected

    def test_typed_dict_handling(self):
        """Test Dict[str, int] conversion."""
        schema = FunctionGenerator.python_type_to_json_schema(Dict[str, int])
        assert schema == {"type": "object"}

    def test_all_primitive_types_in_list(self):
        """Test List of all primitive types."""
        types = [str, int, float, bool]
        json_types = ["string", "integer", "number", "boolean"]

        for py_type, json_type in zip(types, json_types):
            schema = FunctionGenerator.python_type_to_json_schema(List[py_type])
            assert schema == {"type": "array", "items": {"type": json_type}}

    def test_mixed_method_signature(self):
        """Test a method with mixed required and optional parameters of various types."""

        class TestClass:
            @LLMFunction("complex function")
            def complex_method(
                self,
                ids: List[int],
                config: Dict[str, Any],
                name: str = "robot",
                velocity: float = 1.5,
            ):
                pass

        schema = FunctionGenerator.extract_function_schema(TestClass.complex_method)
        params = schema["function"]["parameters"]
        props = params["properties"]

        assert props["ids"]["type"] == "array"
        assert props["ids"]["items"]["type"] == "integer"

        assert props["config"]["type"] == "object"

        assert "default: robot" in props["name"]["description"]
        assert "default: 1.5" in props["velocity"]["description"]

        assert set(params["required"]) == {"ids", "config", "name", "velocity"}
