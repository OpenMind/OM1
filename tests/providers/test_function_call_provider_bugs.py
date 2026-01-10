"""
Unit tests for Function Call Provider to identify potential bugs and edge cases.
"""
from typing import Any, Dict, List, Union

from providers.function_call_provider import FunctionGenerator, LLMFunction


class TestFunctionGeneratorBugs:
    
    def test_complex_type_conversion(self):
        """
        Test how python_type_to_json_schema handles complex types like List[int] or Dict[str, Any].
        Current implementation seems to ignore inner types.
        """
        # Test List[int]
        schema_list_int = FunctionGenerator.python_type_to_json_schema(List[int])
        # Expected: {"type": "array", "items": {"type": "integer"}}
        assert schema_list_int == {"type": "array", "items": {"type": "integer"}}
        
        # Test Dict[str, int]
        FunctionGenerator.python_type_to_json_schema(Dict[str, int])
        
        # Checking if basic types work
        assert FunctionGenerator.python_type_to_json_schema(int) == {"type": "integer"}
        
        # This assertions will likely fail if the implementation is incomplete
        # We are probing for behavior here.
        print(f"List[int] schema: {schema_list_int}")
        
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
        
        # Check if both are in required list (due to strict: true)
        assert "req_param" in required_list
        assert "opt_param" in required_list
        
        # Check description modification for optional param
        props = params["properties"]
        assert "optional" in props["opt_param"].get("description", "")
        assert "default: default" in props["opt_param"].get("description", "")

    def test_nested_generic_types(self):
        """Test nested generic types like List[List[int]]."""
        # List[List[int]]
        schema = FunctionGenerator.python_type_to_json_schema(List[List[int]])
        expected = {
            "type": "array", 
            "items": {
                "type": "array", 
                "items": {"type": "integer"}
            }
        }
        assert schema == expected

    def test_typed_dict_handling(self):
        """Test Dict[str, int] conversion."""
        # Current implementation maps all Dicts to generic object, which is acceptable but could be improved.
        # Checking consistent behavior.
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
                velocity: float = 1.5
            ):
                pass
                
        schema = FunctionGenerator.extract_function_schema(TestClass.complex_method)
        params = schema["function"]["parameters"]
        props = params["properties"]
        
        # Check List[int]
        assert props["ids"]["type"] == "array"
        assert props["ids"]["items"]["type"] == "integer"
        
        # Check Dict
        assert props["config"]["type"] == "object"
        
        # Check Optionals descriptions
        assert "default: robot" in props["name"]["description"]
        assert "default: 1.5" in props["velocity"]["description"]
        
        # Check all are required (Strict mode)
        assert set(params["required"]) == {"ids", "config", "name", "velocity"}
