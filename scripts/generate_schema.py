#!/usr/bin/env python3
"""Generate OM1 configuration schema from codebase."""

import ast
import logging
import os
import sys
from typing import Any, Dict, List

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class ConfigSchemaGenerator:
    """Scans OM1 codebase and generates configuration schema."""

    def __init__(self, root_dir: str):
        """Initialize the schema generator.

        Parameters
        ----------
        root_dir : str
            Absolute path to the OM1 root directory.
        """
        self.root_dir = root_dir
        self.src_dir = os.path.join(root_dir, "src")
        self.inputs_dir = os.path.join(self.src_dir, "inputs/plugins")
        self.llm_dir = os.path.join(self.src_dir, "llm/plugins")
        self.llm_config_path = os.path.join(self.src_dir, "llm/__init__.py")
        self.backgrounds_dir = os.path.join(self.src_dir, "backgrounds/plugins")
        self.actions_dir = os.path.join(self.src_dir, "actions")
        self.hooks_dir = os.path.join(self.src_dir, "hooks")

    def generate(self) -> str:
        """Generate complete configuration schema and save to JSON5 file.

        Scans all component types (inputs, LLMs, backgrounds, actions, hooks)
        and generates a comprehensive schema file.

        Returns
        -------
        str
            Absolute path to the generated schema file.
        """
        import json5

        inputs = self.scan_inputs()
        llms = self.scan_llms()
        backgrounds = self.scan_backgrounds()
        actions = self.scan_actions()
        hooks = self.scan_hooks()

        logging.info(
            f"Extracted from {len(inputs)} inputs, {len(llms)} LLMs, {len(backgrounds)} backgrounds, {len(actions)} actions, {len(hooks)} hook modules"
        )

        schema = {
            "agent_inputs": inputs,
            "cortex_llm": llms,
            "backgrounds": backgrounds,
            "agent_actions": actions,
            "lifecycle_hooks": hooks,
        }

        schema_path = os.path.join(self.root_dir, "OM1_config_schema.json5")
        with open(schema_path, "w") as f:
            json5.dump(schema, f, indent=2)

        return schema_path

    # Input
    def scan_inputs(self) -> List[Dict[str, Any]]:
        """Scan input plugins for FuserInput and Sensor classes.

        Returns
        -------
        List[Dict[str, Any]]
            List of input component schemas.
        """
        return self._scan_plugins(self.inputs_dir, ["FuserInput", "Sensor"], "input")

    # LLM
    def scan_llms(self) -> List[Dict[str, Any]]:
        """Scan LLM plugins.

        Returns
        -------
        List[Dict[str, Any]]
            List of LLM component schemas.
        """
        results = []
        if not os.path.exists(self.llm_dir):
            return results

        # Get base fields from LLMConfig
        base_fields = self._parse_pydantic_class("LLMConfig", self.llm_config_path)

        for filepath in self._py_files(self.llm_dir):
            try:
                tree = ast.parse(open(filepath, "r", encoding="utf-8").read())
                for node in tree.body:
                    if isinstance(node, ast.ClassDef) and self._extends(node, ["LLM"]):
                        fields = {f["name"]: f for f in base_fields}
                        for f in self._parse_getattr(node):
                            fields[f["name"]] = f

                        results.append(
                            {
                                "type": node.name,
                                "category": "llm",
                                "fields": list(fields.values()),
                                "description": ast.get_docstring(node) or "",
                            }
                        )
            except Exception as e:
                logging.error(f"Error parsing {filepath}: {e}")
        return results

    # Background
    def scan_backgrounds(self) -> List[Dict[str, Any]]:
        """Scan background plugins directory for Background classes.

        Returns
        -------
        List[Dict[str, Any]]
            List of background component schemas.
        """
        return self._scan_plugins(self.backgrounds_dir, ["Background"], "background")

    # Action
    def scan_actions(self) -> List[Dict[str, Any]]:
        """Scan action connectors in the actions directory.

        Returns
        -------
        List[Dict[str, Any]]
            List of action connector schemas.
        """
        results = []
        if not os.path.exists(self.actions_dir):
            return results

        for action_name in os.listdir(self.actions_dir):
            action_dir = os.path.join(self.actions_dir, action_name)
            connector_dir = os.path.join(action_dir, "connector")

            if not os.path.isdir(action_dir) or action_name == "__pycache__":
                continue
            if not os.path.exists(connector_dir):
                continue

            for filepath in self._py_files(connector_dir):
                try:
                    tree = ast.parse(open(filepath, "r", encoding="utf-8").read())
                    for node in tree.body:
                        if isinstance(node, ast.ClassDef) and self._extends_connector(
                            node
                        ):
                            connector = os.path.basename(filepath)[:-3]
                            type_name = (
                                action_name
                                if connector == "default"
                                else f"{action_name}_{connector}"
                            )

                            results.append(
                                {
                                    "type": type_name,
                                    "category": "action",
                                    "fields": self._parse_getattr(node),
                                    "description": ast.get_docstring(node) or "",
                                    "action_name": action_name,
                                    "connector_name": connector,
                                }
                            )
                except Exception as e:
                    logging.error(f"Error parsing {filepath}: {e}")
        return results

    # Hooks
    def scan_hooks(self) -> List[Dict[str, Any]]:
        """Scan lifecycle hooks from the hooks directory.

        Identifies all async functions in hook modules as potential lifecycle hooks.

        Returns
        -------
        List[Dict[str, Any]]
            List of hook modules with their function names and arguments.
        """
        results = []
        if not os.path.exists(self.hooks_dir):
            return results

        for filepath in self._py_files(self.hooks_dir):
            try:
                module_name = os.path.basename(filepath)[:-3]
                tree = ast.parse(open(filepath, "r", encoding="utf-8").read())

                functions = []
                for node in tree.body:
                    if isinstance(node, ast.AsyncFunctionDef):
                        functions.append(
                            {
                                "name": node.name,
                                "args": [arg.arg for arg in node.args.args],
                            }
                        )

                if functions:
                    results.append({"module": module_name, "functions": functions})
            except Exception as e:
                logging.error(f"Error parsing {filepath}: {e}")
        return results

    def _scan_plugins(
        self, directory: str, base_classes: List[str], category: str
    ) -> List[Dict[str, Any]]:
        """Generic scanner for plugin directories.

        Parameters
        ----------
        directory : str
            Path to the plugins directory to scan.
        base_classes : List[str]
            List of base class names to match in inheritance.
        category : str
            Category label for the schema.

        Returns
        -------
        List[Dict[str, Any]]
            List of component schemas with extracted fields.
        """
        results = []
        if not os.path.exists(directory):
            return results

        for filepath in self._py_files(directory):
            try:
                tree = ast.parse(open(filepath, "r", encoding="utf-8").read())
                for node in tree.body:
                    if isinstance(node, ast.ClassDef) and self._extends(
                        node, base_classes
                    ):
                        results.append(
                            {
                                "type": node.name,
                                "category": category,
                                "fields": self._parse_getattr(node),
                                "description": ast.get_docstring(node) or "",
                            }
                        )
            except Exception as e:
                logging.error(f"Error parsing {filepath}: {e}")
        return results

    def _py_files(self, directory: str) -> List[str]:
        """List Python files in a directory, excluding init files.

        Parameters
        ----------
        directory : str
            Directory path to scan for Python files.

        Returns
        -------
        List[str]
            List of absolute paths to Python files.
        """
        return [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.endswith(".py") and f != "__init__.py"
        ]

    def _extends(self, node: ast.ClassDef, base_classes: List[str]) -> bool:
        """Check if a class extends any of the specified base classes.

        Parameters
        ----------
        node : ast.ClassDef
            AST node representing a class definition.
        base_classes : List[str]
            List of base class names to check against.

        Returns
        -------
        bool
            True if the class extends any of the base classes, False otherwise.
        """
        for base in node.bases:
            if isinstance(base, ast.Name) and base.id in base_classes:
                return True
            if isinstance(base, ast.Subscript) and isinstance(base.value, ast.Name):
                if base.value.id in base_classes:
                    return True
        return False

    def _extends_connector(self, node: ast.ClassDef) -> bool:
        """Check if a class is a Connector subclass.

        Parameters
        ----------
        node : ast.ClassDef
            AST node representing a class definition.

        Returns
        -------
        bool
            True if the class name contains "Connector" in its base classes.
        """
        for base in node.bases:
            if isinstance(base, ast.Name) and "Connector" in base.id:
                return True
            if isinstance(base, ast.Subscript) and isinstance(base.value, ast.Name):
                if "Connector" in base.value.id:
                    return True
        return False

    def _parse_getattr(self, class_node: ast.ClassDef) -> List[Dict[str, Any]]:
        """Extract configuration parameters from getattr() calls in __init__.

        Parameters
        ----------
        class_node : ast.ClassDef
            The class definition node to extract parameters from.

        Returns
        -------
        List[Dict[str, Any]]
            List of field definitions.
        """
        fields = []
        init = next(
            (
                n
                for n in class_node.body
                if isinstance(n, ast.FunctionDef) and n.name == "__init__"
            ),
            None,
        )
        if not init:
            return fields

        for node in ast.walk(init):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "getattr"
            ):
                continue
            if len(node.args) < 2:
                continue

            arg0 = node.args[0]
            if not (
                (isinstance(arg0, ast.Attribute) and arg0.attr == "config")
                or (isinstance(arg0, ast.Name) and arg0.id == "config")
            ):
                continue

            if not (
                isinstance(node.args[1], ast.Constant)
                and isinstance(node.args[1].value, str)
            ):
                continue
            name = node.args[1].value

            default = (
                node.args[2].value
                if len(node.args) > 2 and isinstance(node.args[2], ast.Constant)
                else None
            )

            field = {
                "name": name,
                "type": (
                    "boolean"
                    if isinstance(default, bool)
                    else "number" if isinstance(default, (int, float)) else "string"
                ),
                "label": name.replace("_", " ").title(),
                "required": default is None,
            }
            if default is not None:
                field["defaultValue"] = default
            fields.append(field)

        return fields

    def _parse_pydantic_class(
        self, class_name: str, file_path: str
    ) -> List[Dict[str, Any]]:
        """Extract fields from a Pydantic BaseModel class definition.

        Parameters
        ----------
        class_name : str
            Name of the Pydantic model class to extract fields from.
        file_path : str
            Absolute path to the file containing the class definition.

        Returns
        -------
        List[Dict[str, Any]]
            List of field definitions extracted from the Pydantic model.
        """
        fields = []
        if not os.path.exists(file_path):
            return fields

        try:
            tree = ast.parse(open(file_path, "r", encoding="utf-8").read())
            for node in tree.body:
                if isinstance(node, ast.ClassDef) and node.name == class_name:
                    for item in node.body:
                        if not (
                            isinstance(item, ast.AnnAssign)
                            and isinstance(item.target, ast.Name)
                        ):
                            continue

                        name = item.target.id
                        if name.startswith("_") or name == "model_config":
                            continue

                        default = self._get_pydantic_default(item.value)
                        if default == "__SKIP__":
                            continue

                        fields.append(
                            {
                                "name": name,
                                "type": self._annotation_to_type(item.annotation),
                                "label": name.replace("_", " ").title(),
                                "required": default is None,
                                **(
                                    {"defaultValue": default}
                                    if default is not None
                                    else {}
                                ),
                            }
                        )
                    break
        except Exception as e:
            logging.error(f"Error parsing Pydantic class: {e}")
        return fields

    def _annotation_to_type(self, annotation: ast.expr) -> str:
        """Convert Python type annotation to JSON schema type.

        Handles:
            - T.Optional[str] -> "string"
            - Optional[int] -> "number"
            - bool -> "boolean"
            - Dict/List -> "object"

        Parameters
        ----------
        annotation : ast.expr
            AST node representing a type annotation.

        Returns
        -------
        str
            JSON schema type string.
        """
        if annotation is None:
            return "string"

        if isinstance(annotation, ast.Subscript):
            if isinstance(annotation.slice, ast.Name):
                t = annotation.slice.id
                if t in ("str", "string"):
                    return "string"
                if t in ("int", "float"):
                    return "number"
                if t == "bool":
                    return "boolean"
            return "object"

        if isinstance(annotation, ast.Name):
            t = annotation.id
            if t in ("str", "string"):
                return "string"
            if t in ("int", "float"):
                return "number"
            if t == "bool":
                return "boolean"

        return "string"

    def _get_pydantic_default(self, value_node: ast.expr):
        """Extract default value from Pydantic field AST node.

        Parameters
        ----------
        value_node : ast.expr
            AST node representing the default value expression.

        Returns
        -------
        Any or str
            The default value, None if no default.
        """
        if value_node is None:
            return None
        if isinstance(value_node, ast.Constant):
            return value_node.value
        if isinstance(value_node, ast.Call):
            return "__SKIP__"
        return None


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)

    try:
        schema_path = ConfigSchemaGenerator(root_dir).generate()
        print(f"✓ Schema generated successfully: {schema_path}")
        return 0
    except Exception as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
