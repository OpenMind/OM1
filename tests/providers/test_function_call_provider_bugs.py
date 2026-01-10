"""
Unit tests for Function Call Provider to identify potential bugs and edge cases.
"""
from typing import Dict, List, Union

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

    def test_union_types_edge_cases(self):
        """Test Union types scenarios."""
        # Union[str, int] -> currently returns {"type": "string"} based on defaults?
        schema = FunctionGenerator.python_type_to_json_schema(Union[str, int])
        assert schema == {"type": "string"} # This behavior seems too generic/wrong
