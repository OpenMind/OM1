import pytest

from src.providers.function_call_provider import (
    LLMFunction,
    FunctionGenerator,
)


class DummyFunctions:
    @LLMFunction(description="Adds two numbers")
    def add(self, a: int, b: int) -> int:
        return a + b

    @LLMFunction(description="Optional parameter test")
    def greet(self, name: str, title: str = "Mr"):
        return f"{title} {name}"


class DuplicateNameFunctions:
    @LLMFunction(description="First", name="duplicate")
    def first(self):
        pass

    @LLMFunction(description="Second", name="duplicate")
    def second(self):
        pass


def test_generate_function_schema_basic():
    instance = DummyFunctions()
    functions = FunctionGenerator.generate_functions_from_class(instance)

    assert "add" in functions
    schema = functions["add"]["function"]

    assert schema["name"] == "add"
    assert "parameters" in schema
    assert "a" in schema["parameters"]["properties"]
    assert "b" in schema["parameters"]["properties"]
    assert set(schema["parameters"]["required"]) == {"a", "b"}


def test_optional_parameter_not_required():
    instance = DummyFunctions()
    functions = FunctionGenerator.generate_functions_from_class(instance)

    schema = functions["greet"]["function"]
    required = schema["parameters"]["required"]

    assert "name" in required
    assert "title" not in required


def test_list_type_schema_generation():
    class ListFunc:
        @LLMFunction(description="Process list")
        def process(self, values: list[int]):
            pass

    instance = ListFunc()
    functions = FunctionGenerator.generate_functions_from_class(instance)

    schema = functions["process"]["function"]
    props = schema["parameters"]["properties"]

    assert props["values"]["type"] == "array"
    assert "items" in props["values"]


def test_duplicate_function_name_raises_error():
    instance = DuplicateNameFunctions()

    with pytest.raises(ValueError):
        FunctionGenerator.generate_functions_from_class(instance)


def test_metadata_fields_present():
    instance = DummyFunctions()
    functions = FunctionGenerator.generate_functions_from_class(instance)

    schema = functions["add"]

    assert schema["function"]["x-llm-generated"] is True
    assert "x-source-class" in schema["function"]