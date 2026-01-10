import importlib.util
from typing import Optional

# Load function_call_provider module directly without triggering providers/__init__.py
spec = importlib.util.spec_from_file_location(
    "function_call_provider", "src/providers/function_call_provider.py"
)
fcp_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fcp_module)
LLMFunction = fcp_module.LLMFunction
FunctionGenerator = fcp_module.FunctionGenerator


class TestLLMFunctionDecorator:
    """Test suite for the LLMFunction decorator."""

    def test_decorator_sets_attributes(self):
        """Test that decorator sets the correct attributes on the function."""

        @LLMFunction(description="Test function description")
        def test_func():
            pass

        assert hasattr(test_func, "_llm_function")
        assert test_func._llm_function is True
        assert test_func._llm_description == "Test function description"
        assert test_func._llm_name == "test_func"

    def test_decorator_with_custom_name(self):
        """Test that decorator respects custom name."""

        @LLMFunction(description="Custom named function", name="custom_name")
        def original_name():
            pass

        assert original_name._llm_name == "custom_name"

    def test_decorator_preserves_function_behavior(self):
        """Test that decorated function still works normally."""

        @LLMFunction(description="Add two numbers")
        def add(a: int, b: int) -> int:
            return a + b

        result = add(2, 3)
        assert result == 5


class TestFunctionGenerator:
    """Test suite for the FunctionGenerator class."""

    def test_python_type_to_json_schema_basic_types(self):
        """Test conversion of basic Python types to JSON schema."""
        assert FunctionGenerator.python_type_to_json_schema(str) == {"type": "string"}
        assert FunctionGenerator.python_type_to_json_schema(int) == {"type": "integer"}
        assert FunctionGenerator.python_type_to_json_schema(float) == {"type": "number"}
        assert FunctionGenerator.python_type_to_json_schema(bool) == {"type": "boolean"}
        assert FunctionGenerator.python_type_to_json_schema(list) == {"type": "array"}
        assert FunctionGenerator.python_type_to_json_schema(dict) == {"type": "object"}

    def test_python_type_to_json_schema_optional(self):
        """Test conversion of Optional types."""
        schema = FunctionGenerator.python_type_to_json_schema(Optional[str])
        assert schema == {"type": "string"}

        schema = FunctionGenerator.python_type_to_json_schema(Optional[int])
        assert schema == {"type": "integer"}

    def test_python_type_to_json_schema_unknown_type(self):
        """Test that unknown types default to string."""

        class CustomClass:
            pass

        schema = FunctionGenerator.python_type_to_json_schema(CustomClass)
        assert schema == {"type": "string"}

    def test_extract_function_schema_simple(self):
        """Test extracting schema from a simple function."""

        @LLMFunction(description="Greet a person")
        def greet(name: str) -> str:
            return f"Hello, {name}!"

        schema = FunctionGenerator.extract_function_schema(greet)

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "greet"
        assert schema["function"]["description"] == "Greet a person"
        assert "name" in schema["function"]["parameters"]["properties"]
        assert schema["function"]["parameters"]["properties"]["name"]["type"] == "string"
        assert "name" in schema["function"]["parameters"]["required"]

    def test_extract_function_schema_multiple_params(self):
        """Test extracting schema from function with multiple parameters."""

        @LLMFunction(description="Calculate sum")
        def calculate(a: int, b: float, enabled: bool) -> float:
            return a + b if enabled else 0

        schema = FunctionGenerator.extract_function_schema(calculate)

        props = schema["function"]["parameters"]["properties"]
        assert props["a"]["type"] == "integer"
        assert props["b"]["type"] == "number"
        assert props["enabled"]["type"] == "boolean"

    def test_extract_function_schema_with_defaults(self):
        """Test extracting schema from function with default values."""

        @LLMFunction(description="Send message")
        def send_message(message: str, priority: int = 1) -> bool:
            return True

        schema = FunctionGenerator.extract_function_schema(send_message)

        assert "message" in schema["function"]["parameters"]["required"]
        assert "priority" in schema["function"]["parameters"]["required"]

    def test_extract_function_schema_custom_name(self):
        """Test that custom LLM name is used in schema."""

        @LLMFunction(description="Do something", name="custom_action")
        def internal_name():
            pass

        schema = FunctionGenerator.extract_function_schema(internal_name)
        assert schema["function"]["name"] == "custom_action"

    def test_generate_functions_from_class(self):
        """Test generating function schemas from a class."""

        class TestActions:
            @LLMFunction(description="Move forward")
            def move(self, distance: float) -> bool:
                return True

            @LLMFunction(description="Turn robot", name="rotate")
            def turn(self, angle: int) -> bool:
                return True

            def not_llm_function(self):
                """This should not be included."""
                pass

        instance = TestActions()
        functions = FunctionGenerator.generate_functions_from_class(instance)

        assert "move" in functions
        assert "rotate" in functions
        assert "not_llm_function" not in functions
        assert functions["move"]["function"]["description"] == "Move forward"
        assert functions["rotate"]["function"]["description"] == "Turn robot"

    def test_schema_has_strict_mode(self):
        """Test that generated schema has strict mode enabled."""

        @LLMFunction(description="Test strict")
        def test_func(param: str) -> None:
            pass

        schema = FunctionGenerator.extract_function_schema(test_func)
        assert schema["function"]["strict"] is True

    def test_schema_disallows_additional_properties(self):
        """Test that schema disallows additional properties."""

        @LLMFunction(description="Test additional props")
        def test_func(param: str) -> None:
            pass

        schema = FunctionGenerator.extract_function_schema(test_func)
        assert schema["function"]["parameters"]["additionalProperties"] is False
