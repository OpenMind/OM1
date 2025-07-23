import inspect
import typing as T
from typing import get_args, get_origin, get_type_hints


class LLMFunction:
    """
    Decorator to mark methods as LLM-callable functions.
    """

    def __init__(self, description: str, name: str = None):
        """
        Initialize the LLM function decorator.

        Parameters
        ----------
        description : str
            Description of what the function does for the LLM.
        name : str, optional
            Custom name for the function. If None, uses the method name.
        """
        self.description = description
        self.name = name

    def __call__(self, func):
        func._llm_function = True
        func._llm_description = self.description
        func._llm_name = self.name or func.__name__
        return func


class FunctionGenerator:
    """
    Utility class to automatically generate OpenAI function schemas from methods.
    """

    @staticmethod
    def python_type_to_json_schema(python_type: T.Type) -> T.Dict:
        """
        Convert Python type hints to JSON schema format.

        Parameters
        ----------
        python_type : T.Type
            The Python type to convert.

        Returns
        -------
        T.Dict
            JSON schema representation of the Python type.
        """
        if get_origin(python_type) is T.Union:
            args = get_args(python_type)
            if len(args) == 2 and type(None) in args:
                non_none_type = args[0] if args[1] is type(None) else args[1]
                schema = FunctionGenerator.python_type_to_json_schema(non_none_type)
                return schema
            else:
                return {"type": "string"}

        type_mapping = {
            str: {"type": "string"},
            int: {"type": "integer"},
            float: {"type": "number"},
            bool: {"type": "boolean"},
            list: {"type": "array"},
            dict: {"type": "object"},
        }

        return type_mapping.get(python_type, {"type": "string"})

    @staticmethod
    def extract_function_schema(method: callable) -> T.Dict:
        """
        Extract OpenAI function schema from a method.

        Parameters
        ----------
        method : callable
            The method to extract the schema from.

        Returns
        -------
        T.Dict
            JSON schema representation of the function.
        """
        sig = inspect.signature(method)
        type_hints = get_type_hints(method)

        properties = {}
        required = []

        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue

            param_type = type_hints.get(param_name, str)

            param_schema = FunctionGenerator.python_type_to_json_schema(param_type)

            docstring = inspect.getdoc(method)
            if docstring and param_name in docstring:
                param_schema["description"] = f"Parameter {param_name}"

            properties[param_name] = param_schema

            if param.default == inspect.Parameter.empty:
                if get_origin(param_type) is T.Union and type(None) in get_args(
                    param_type
                ):
                    pass
                else:
                    required.append(param_name)

        return {
            "type": "function",
            "function": {
                "name": method._llm_name,
                "description": method._llm_description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    @staticmethod
    def generate_functions_from_class(cls: T.Type) -> T.Dict:
        """
        Generate all function schemas from a class with decorated methods.

        Parameters
        ----------
        cls : type
            The class to extract function schemas from.

        Returns
        -------
        T.Dict
            Dictionary of function names to their JSON schema representations.
        """
        functions = {}

        for _, method in inspect.getmembers(cls, predicate=inspect.ismethod):
            if hasattr(method, "_llm_function") and method._llm_function:
                function_schema = FunctionGenerator.extract_function_schema(method)
                functions[method._llm_name] = function_schema

        return functions
