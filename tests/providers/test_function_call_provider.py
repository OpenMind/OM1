import typing as T

import pytest

from providers.function_call_provider import FunctionGenerator, LLMFunction


class _DemoTool:
    @LLMFunction("demo tool")
    def run(
        self,
        required_str: str,
        optional_int: T.Optional[int] = None,
        default_number: int = 5,
    ) -> None:
        return None

    @LLMFunction("variadic tool")
    def variadic(self, first: str, *args: str, **kwargs: str) -> None:
        return None


def test_required_parameters_respected():
    schema = FunctionGenerator.extract_function_schema(_DemoTool.run)

    params = schema["function"]["parameters"]
    assert params["required"] == ["required_str"]
    assert "optional_int" in params["properties"]
    assert "default_number" in params["properties"]
    assert params["properties"]["optional_int"]["type"] == "integer"
    assert params["properties"]["default_number"]["type"] == "integer"


def test_generate_functions_skips_variadic():
    functions = FunctionGenerator.generate_functions_from_class(_DemoTool())

    assert "variadic" not in functions
    assert "run" in functions
    params = functions["run"]["function"]["parameters"]
    assert params["required"] == ["required_str"]

