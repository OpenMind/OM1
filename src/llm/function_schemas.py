"""
Utility functions for generating OpenAI function schemas from AgentActions.

This module is separate from both the actions and llm modules to avoid circular imports.
"""

import json
import logging
from enum import Enum
from typing import get_type_hints

from llm.output_model import Action


def generate_function_schema_from_action(action) -> dict:
    """
    Generate OpenAI function schema from an AgentAction.

    Parameters
    ----------
    action : AgentAction
        The action to generate a function schema for.

    Returns
    -------
    dict
        OpenAI function schema dictionary.
    """
    interface = action.interface
    input_interface = get_type_hints(interface)["input"]

    doc = interface.__doc__ or ""
    doc = doc.replace("\n", " ").strip()

    properties = {}
    required = []

    for field_name, field_type in get_type_hints(input_interface).items():
        if isinstance(field_type, type) and issubclass(field_type, Enum):
            enum_values = [v.value for v in field_type]
            properties[field_name] = {
                "type": "string",
                "enum": enum_values,
                "description": f"The {field_name} to perform. Must be one of: {', '.join(enum_values)}",
            }
        elif isinstance(field_type, str):
            properties[field_name] = {
                "type": "string",
                "description": f"The {field_name} parameter",
            }
        elif isinstance(field_type, int):
            properties[field_name] = {
                "type": "integer",
                "description": f"The {field_name} parameter",
            }
        elif isinstance(field_type, float):
            properties[field_name] = {
                "type": "number",
                "description": f"The {field_name} parameter",
            }
        elif isinstance(field_type, bool):
            properties[field_name] = {
                "type": "boolean",
                "description": f"The {field_name} parameter",
            }
        else:
            properties[field_name] = {
                "type": "string",
                "description": f"The {field_name} parameter",
            }

        required.append(field_name)

    return {
        "type": "function",
        "function": {
            "name": action.llm_label,
            "description": doc or f"Execute {action.llm_label} action",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
            "strict": True,
        },
    }


def generate_function_schemas_from_actions(actions: list) -> list[dict]:
    """
    Generate OpenAI function schemas from a list of AgentActions.

    Parameters
    ----------
    actions : list
        List of actions to generate function schemas for.

    Returns
    -------
    list[dict]
        List of OpenAI function schema dictionaries.
    """
    schemas = []
    for action in actions:
        if not action.exclude_from_prompt:
            try:
                schema = generate_function_schema_from_action(action)
                schemas.append(schema)
                logging.debug(
                    f"Generated function schema for {action.llm_label}: {schema}"
                )
            except Exception as e:
                logging.error(
                    f"Error generating function schema for {action.llm_label}: {e}"
                )

    return schemas


def convert_function_calls_to_actions(function_calls: list[dict]) -> list[Action]:
    """
    Convert OpenAI function call responses to Action objects.

    Parameters
    ----------
    function_calls : list[dict]
        List of function call dictionaries from OpenAI response.

    Returns
    -------
    list[Action]
        List of Action objects for the action orchestrator.
    """
    actions = []

    for call in function_calls:
        try:
            function_name = call.get("function", {}).get("name")
            function_args = call.get("function", {}).get("arguments", "{}")

            # Parse arguments if they're a string
            if isinstance(function_args, str):
                try:
                    args = json.loads(function_args)
                except json.JSONDecodeError:
                    logging.error(
                        f"Failed to parse function arguments: {function_args}"
                    )
                    continue
            else:
                args = function_args

            if "action" in args and len(args) == 1:
                action_value = args["action"]
            elif len(args) > 1:
                action_value = json.dumps(args)
            elif len(args) == 1:
                for param in ["text", "message", "value", "command"]:
                    if param in args:
                        action_value = args[param]
                        break
                else:
                    action_value = str(list(args.values())[0])
            else:
                action_value = ""

            action = Action(type=function_name, value=action_value)
            actions.append(action)

            logging.info(
                f"Converted function call {function_name}({args}) to action: {action}"
            )

        except Exception as e:
            logging.error(f"Error converting function call to action: {e}")
            continue

    return actions
