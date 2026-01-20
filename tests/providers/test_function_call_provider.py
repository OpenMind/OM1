import typing as T

from providers.function_call_provider import FunctionGenerator, LLMFunction


def test_llm_function_decorator_sets_metadata() -> None:
    @LLMFunction(description="Test description", name="custom_name")
    def sample_method(param: str) -> None:
        pass

    assert getattr(sample_method, "_llm_function") is True  # type: ignore[attr-defined]
    assert (
        getattr(sample_method, "_llm_description")  # type: ignore[attr-defined]
        == "Test description"
    )
    assert (
        getattr(sample_method, "_llm_name")  # type: ignore[attr-defined]
        == "custom_name"
    )


def test_python_type_to_json_schema_basic_types() -> None:
    mapping = {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
        list: {"type": "array"},
        dict: {"type": "object"},
    }

    for py_type, expected_schema in mapping.items():
        assert FunctionGenerator.python_type_to_json_schema(py_type) == expected_schema

    class CustomType:
        pass

    # Unknown/custom types fall back to string representation
    assert FunctionGenerator.python_type_to_json_schema(CustomType) == {
        "type": "string"
    }


def test_python_type_to_json_schema_optional_and_union() -> None:
    optional_int = T.Optional[int]
    schema_optional = FunctionGenerator.python_type_to_json_schema(optional_int)
    assert schema_optional == {"type": "integer"}

    union_type = T.Union[int, str]
    schema_union = FunctionGenerator.python_type_to_json_schema(union_type)
    # For non-optional unions, current behavior is to fall back to string
    assert schema_union == {"type": "string"}


def test_extract_function_schema_includes_parameters_and_required() -> None:
    class Sample:
        @LLMFunction(description="Does something")
        def do_something(self, a: int, b: str = "default") -> None:
            """Example method.

            Parameters
            ----------
            a : int
                First value
            b : str
                Second value
            """

    instance = Sample()
    method = instance.do_something

    schema = FunctionGenerator.extract_function_schema(method)

    assert schema["type"] == "function"
    fn = schema["function"]
    assert fn["name"] == "do_something"
    assert fn["description"] == "Does something"

    params = fn["parameters"]
    assert params["type"] == "object"
    assert set(params["properties"].keys()) == {"a", "b"}

    # Current implementation marks both parameters as required
    assert set(params["required"]) == {"a", "b"}

    a_schema = params["properties"]["a"]
    b_schema = params["properties"]["b"]
    assert a_schema["type"] == "integer"
    assert b_schema["type"] == "string"


def test_generate_functions_from_class_collects_decorated_methods() -> None:
    class Sample:
        @LLMFunction(description="First")
        def first(self, x: int) -> None:
            pass

        def not_exposed(self) -> None:
            pass

    functions = FunctionGenerator.generate_functions_from_class(Sample())

    assert "first" in functions
    assert functions["first"]["function"]["name"] == "first"
    assert "not_exposed" not in functions

