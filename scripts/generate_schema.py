"""Generate OM1 configuration schema from codebase."""

import ast
import json
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
        self.schema_path = os.path.join(
            root_dir, "config/schema/multi_mode_schema.json"
        )

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
        transition_rules = self.scan_transition_rules()

        logging.info(
            f"Extracted from {len(inputs)} inputs, {len(llms)} LLMs, {len(backgrounds)} backgrounds, {len(actions)} actions, {len(hooks.get('available_functions', []))} hook modules, {len(transition_rules.get('transition_types', []))} transition types"
        )

        schema = {
            "agent_inputs": inputs,
            "cortex_llm": llms,
            "backgrounds": backgrounds,
            "agent_actions": actions,
            "lifecycle_hooks": hooks,
            "transition_rules": transition_rules,
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
        results = []
        if not os.path.exists(self.inputs_dir):
            return results

        for filepath in self._py_files(self.inputs_dir):
            try:
                tree = ast.parse(open(filepath, "r", encoding="utf-8").read())

                config_node = None
                sensor_node = None

                for node in tree.body:
                    if isinstance(node, ast.ClassDef):
                        if self._extends(node, ["InputConfig", "SensorConfig"]):
                            config_node = node
                        if self._extends(node, ["FuserInput", "Sensor"]):
                            sensor_node = node

                if sensor_node:
                    fields = (
                        self._parse_pydantic_fields_from_node(config_node)
                        if config_node
                        else []
                    )

                    results.append(
                        {
                            "type": sensor_node.name,
                            "category": "input",
                            "fields": fields,
                            "description": ast.get_docstring(sensor_node) or "",
                        }
                    )
            except Exception as e:
                logging.error(f"Error parsing {filepath}: {e}")
        return results

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

                config_node = None
                llm_node = None

                for node in tree.body:
                    if isinstance(node, ast.ClassDef):
                        if self._extends(node, ["LLMConfig"]):
                            config_node = node
                        if self._extends(node, ["LLM"]):
                            llm_node = node

                if llm_node:
                    fields = {f["name"]: f for f in base_fields}

                    if config_node:
                        for f in self._parse_pydantic_fields_from_node(config_node):
                            fields[f["name"]] = f

                    results.append(
                        {
                            "type": llm_node.name,
                            "category": "llm",
                            "fields": list(fields.values()),
                            "description": ast.get_docstring(llm_node) or "",
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
        results = []
        if not os.path.exists(self.backgrounds_dir):
            return results

        for filepath in self._py_files(self.backgrounds_dir):
            try:
                tree = ast.parse(open(filepath, "r", encoding="utf-8").read())

                config_node = None
                background_node = None

                for node in tree.body:
                    if isinstance(node, ast.ClassDef):
                        if self._extends(node, ["BackgroundConfig"]):
                            config_node = node
                        if self._extends(node, ["Background"]):
                            background_node = node

                if background_node:
                    fields = (
                        self._parse_pydantic_fields_from_node(config_node)
                        if config_node
                        else []
                    )

                    results.append(
                        {
                            "type": background_node.name,
                            "category": "background",
                            "fields": fields,
                            "description": ast.get_docstring(background_node) or "",
                        }
                    )
            except Exception as e:
                logging.error(f"Error parsing {filepath}: {e}")
        return results

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

                    # Find config class and connector class in the file
                    config_node = None
                    connector_node = None

                    for node in tree.body:
                        if isinstance(node, ast.ClassDef):
                            if self._extends(node, ["ActionConfig"]):
                                config_node = node
                            if self._extends_connector(node):
                                connector_node = node

                    if connector_node:
                        if config_node:
                            fields = self._parse_pydantic_fields_from_node(config_node)
                        else:
                            fields = []

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
                                "fields": fields,
                                "description": ast.get_docstring(connector_node) or "",
                                "action_name": action_name,
                                "connector_name": connector,
                            }
                        )
                except Exception as e:
                    logging.error(f"Error parsing {filepath}: {e}")
        return results

    # Hooks
    def scan_hooks(self) -> Dict[str, Any]:
        """Scan lifecycle hooks.

        Returns
        -------
        Dict[str, Any]
            Schema structure and available hook functions.
        """
        result = {
            "schema": self._get_hooks_schema(),
            "available_functions": self._scan_hook_functions(),
        }
        return result

    def _get_hooks_schema(self) -> Dict[str, Any]:
        """Extract lifecycle_hooks schema.

        Returns
        -------
        Dict[str, Any]
            The hooks schema structure with enums and field types.
        """
        if not os.path.exists(self.schema_path):
            logging.warning(f"Schema file not found: {self.schema_path}")
            return {}

        try:
            with open(self.schema_path, "r") as f:
                schema = json.load(f)

            hooks_schema = schema.get("properties", {}).get(
                "global_lifecycle_hooks", {}
            )
            if hooks_schema:
                return hooks_schema.get("items", {}).get("properties", {})
            return {}
        except Exception as e:
            logging.error(f"Error parsing schema file: {e}")
            return {}

    def _scan_hook_functions(self) -> List[Dict[str, Any]]:
        """Scan available hook functions from hooks directory.

        Returns
        -------
        List[Dict[str, Any]]
            List of hook modules with their function names.
        """
        results = []
        if not os.path.exists(self.hooks_dir):
            return results

        for filepath in self._py_files(self.hooks_dir):
            try:
                module_name = os.path.basename(filepath)[:-3]
                tree = ast.parse(open(filepath, "r", encoding="utf-8").read())

                functions = [
                    node.name
                    for node in tree.body
                    if isinstance(node, ast.AsyncFunctionDef)
                ]

                if functions:
                    results.append({"module_name": module_name, "functions": functions})
            except Exception as e:
                logging.error(f"Error parsing {filepath}: {e}")
        return results

    # Transition Rules
    def scan_transition_rules(self) -> Dict[str, Any]:
        """Extract transition_rules schema from multi_mode_schema.json.

        Returns
        -------
        Dict[str, Any]
            Transition rules schema with types and properties.
        """
        if not os.path.exists(self.schema_path):
            logging.warning(f"Schema file not found: {self.schema_path}")
            return {}

        try:
            with open(self.schema_path, "r") as f:
                schema = json.load(f)

            transition_schema = schema.get("properties", {}).get("transition_rules", {})
            if not transition_schema:
                return {}

            items = transition_schema.get("items", {})
            properties = items.get("properties", {})
            required = items.get("required", [])

            transition_types = []
            if "transition_type" in properties:
                transition_types = properties["transition_type"].get("enum", [])

            return {
                "required": required,
                "transition_types": transition_types,
                "properties": properties,
            }
        except Exception as e:
            logging.error(f"Error parsing transition_rules schema: {e}")
            return {}

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
        if not os.path.exists(file_path):
            return []

        try:
            tree = ast.parse(open(file_path, "r", encoding="utf-8").read())
            for node in tree.body:
                if isinstance(node, ast.ClassDef) and node.name == class_name:
                    return self._parse_pydantic_fields_from_node(node)
        except Exception as e:
            logging.error(f"Error parsing Pydantic class: {e}")
        return []

    def _parse_pydantic_fields_from_node(
        self, node: ast.ClassDef
    ) -> List[Dict[str, Any]]:
        """Extract fields from a Pydantic ClassDef node.

        Parameters
        ----------
        node : ast.ClassDef
            The class definition node.

        Returns
        -------
        List[Dict[str, Any]]
            List of extracted fields.
        """
        fields = []
        for item in node.body:
            if not (
                isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)
            ):
                continue

            name = item.target.id
            if name.startswith("_") or name == "model_config":
                continue

            description = ""
            default = None

            if item.value:
                default = self._get_pydantic_default(item.value)

                if (
                    isinstance(item.value, ast.Call)
                    and isinstance(item.value.func, ast.Name)
                    and item.value.func.id == "Field"
                ):
                    for keyword in item.value.keywords:
                        if keyword.arg == "description" and isinstance(
                            keyword.value, ast.Constant
                        ):
                            description = keyword.value.value
            else:
                default = None

            if default == "__SKIP__":
                continue

            has_default = default is not None
            if default == "__HAS_DEFAULT__":
                has_default = True
                default = None

            field = {
                "name": name,
                "type": self._annotation_to_type(item.annotation),
                "label": name.replace("_", " ").title(),
                "required": has_default,
                "description": description,
            }

            if default is not None:
                field["defaultValue"] = default

            fields.append(field)

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
            if isinstance(value_node.func, ast.Name) and value_node.func.id == "Field":
                for keyword in value_node.keywords:
                    if keyword.arg == "default":
                        return self._get_pydantic_default(keyword.value)
                    if keyword.arg == "default_factory":
                        return "__HAS_DEFAULT__"
                if value_node.args:
                    return self._get_pydantic_default(value_node.args[0])
                return None
            return "__SKIP__"
        return None


def main():
    """
    Main entry point for schema generation script.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)

    try:
        schema_path = ConfigSchemaGenerator(root_dir).generate()
        logging.info(f"✓ Schema generated successfully: {schema_path}")
        return 0
    except Exception as e:
        logging.error(f"✗ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
