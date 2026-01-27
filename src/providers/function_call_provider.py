import inspect
import typing as T
from typing import get_args, get_origin, get_type_hints


class LLMFunction:
    """
    Decorator to mark methods as LLM-callable functions.
    """

    def __init__(self, description: str, name: T.Optional[str] = None):
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

    def __call__(self, func: T.Callable) -> T.Callable:
        """
        Decorate the method to mark it as an LLM function.

        Parameters
        ----------
        func : callable
            The method to decorate.

        Returns
        -------
        callable
            The decorated method with LLM function metadata.
        """
        func._llm_function = True  # type: ignore
        func._llm_description = self.description  # type: ignore
        func._llm_name = self.name or func.__name__  # type: ignore
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
        origin = get_origin(python_type)
        args = get_args(python_type)

        # Handle Optional types
        if origin is T.Union:
            if len(args) == 2 and type(None) in args:
                non_none_type = args[0] if args[1] is type(None) else args[1]
                schema = FunctionGenerator.python_type_to_json_schema(non_none_type)
                return schema
            else:
                return {"type": "string"}

        # Handle List/list generics
        if origin is list or origin is T.List:
            item_type = args[0] if args else str
            return {
                "type": "array",
                "items": FunctionGenerator.python_type_to_json_schema(item_type),
            }

        # Handle Dict/dict generics
        if origin is dict or origin is T.Dict:
            return {"type": "object"}

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
    def extract_function_schema(method: T.Callable) -> T.Dict:
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

                # Basic docstring parsing (Google/NumPy style simple extraction)
            docstring = inspect.getdoc(method)
            if docstring:
                import re
                # Try to find "param_name : type\n    Description" or "param_name\n    Description"
                # This is a simple regex and might not cover all cases but is better than nothing
                pattern = re.compile(rf"{param_name}\s*:?.*?\n\s+(.+?)(\n\s*\n|\n\s*\w+|$)", re.DOTALL)
                match = pattern.search(docstring)
                if match:
                    description = match.group(1).replace("\n", " ").strip()
                    param_schema["description"] = description
                elif param_name in docstring:
                     # Fallback if specific parsing fails but name is mentioned
                     pass 

            if param.default == inspect.Parameter.empty:
                required.append(param_name)
            else:
                # Optional parameter
                desc = param_schema.get("description", f"Parameter {param_name}")
                if "(optional)" not in desc and "(optional" not in desc:
                    param_schema["description"] = (
                        f"{desc} (optional, default: {param.default})"
                    )

            properties[param_name] = param_schema

        return {
            "type": "function",
            "function": {
                "name": getattr(method, "_llm_name", method.__name__),
                "description": getattr(method, "_llm_description", "") or inspect.getdoc(method) or "",
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }

    @staticmethod
    def generate_functions_from_class(cls_instance: T.Type) -> T.Dict:
        """
        Generate all function schemas from a class with decorated methods.

        Parameters
        ----------
        cls_instance : type
            The class to extract function schemas from.

        Returns
        -------
        T.Dict
            Dictionary of function names to their JSON schema representations.
        """
        functions = {}

        # Inspect for both methods (instances) and functions (classes)
        predicate = lambda x: inspect.isfunction(x) or inspect.ismethod(x)
        for _, method in inspect.getmembers(cls_instance, predicate=predicate):
            # Unwrap if it's a bound method to get the function attributes
            func_obj = method.__func__ if hasattr(method, "__func__") else method
            
            if getattr(func_obj, "_llm_function", False):
                function_schema = FunctionGenerator.extract_function_schema(method)
                functions[getattr(func_obj, "_llm_name", func_obj.__name__)] = (
                    function_schema
                )

        return functions
