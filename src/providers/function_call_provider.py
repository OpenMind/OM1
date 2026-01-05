import inspect
import typing as T
from typing import get_args, get_origin, get_type_hints


class LLMFunction:
    """
    Decorator to mark methods as LLM-callable functions.
    """

    def __init__(self, description: str, name: T.Optional[str] = None):
        self.description = description
        self.name = name

    def __call__(self, func):
        func._llm_function = True
        func._llm_description = self.description
        func._llm_name = self.name or func.__name__
        return func


class FunctionGenerator:
    """
    Utility class to automatically generate LLM function schemas from methods.
    """

    @staticmethod
    def python_type_to_json_schema(python_type: T.Type) -> T.Dict:
        origin = get_origin(python_type)
        args = get_args(python_type)

        if origin is T.Union:
            non_none = [a for a in args if a is not type(None)]
            if len(non_none) == 1:
                return FunctionGenerator.python_type_to_json_schema(non_none[0])
            return {"type": "string"}

        if origin in (list, T.List):
            item_type = args[0] if args else str
            return {
                "type": "array",
                "items": FunctionGenerator.python_type_to_json_schema(item_type),
            }

        type_mapping = {
            str: {"type": "string"},
            int: {"type": "integer"},
            float: {"type": "number"},
            bool: {"type": "boolean"},
            dict: {"type": "object"},
        }

        return type_mapping.get(python_type, {"type": "string"})

    @staticmethod
    def extract_function_schema(method: T.Callable) -> T.Dict:
        sig = inspect.signature(method)
        type_hints = get_type_hints(method)
        docstring = inspect.getdoc(method) or ""

        properties = {}
        required = []

        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue

            param_type = type_hints.get(param_name, str)
            param_schema = FunctionGenerator.python_type_to_json_schema(param_type)

            param_schema["description"] = f"Parameter `{param_name}`"
            properties[param_name] = param_schema

            if param.default == inspect.Parameter.empty:
                required.append(param_name)

        return {
            "type": "function",
            "function": {
                "name": getattr(method, "_llm_name", method.__name__),
                "description": getattr(method, "_llm_description", ""),
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                },
                "strict": True,
                "x-llm-generated": True,
            },
        }

    @staticmethod
    def generate_functions_from_class(instance: T.Any) -> T.Dict[str, T.Dict]:
        """
        Generate function schemas from an instance containing LLM-decorated methods.
        """
        functions: T.Dict[str, T.Dict] = {}

        for _, method in inspect.getmembers(instance, predicate=inspect.ismethod):
            if getattr(method.__func__, "_llm_function", False):
                name = getattr(method, "_llm_name", method.__name__)

                if name in functions:
                    raise ValueError(f"Duplicate LLM function name detected: {name}")

                schema = FunctionGenerator.extract_function_schema(method)
                schema["function"]["x-source-class"] = instance.__class__.__name__

                functions[name] = schema

        return functions