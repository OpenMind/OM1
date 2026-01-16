import json
import logging
from pathlib import Path

from jsonschema import ValidationError, validate


def _load_schema(schema_file: str) -> dict:
    """
    Load and cache schema files.

    Parameters
    ----------
    schema_file : str
        Name of the schema file to load.

    Returns
    -------
    dict
        The loaded schema dictionary.

    Raises
    ------
    FileNotFoundError
        If the schema file does not exist.
    """
    schema_path = Path(__file__).parent / "../../config/schema" / schema_file

    if not schema_path.exists():
        raise FileNotFoundError(
            f"Schema file not found: {schema_path}. Cannot validate configuration."
        )

    with open(schema_path, "r") as f:
        return json.load(f)


def validate_config_schema(raw_config: dict) -> None:
    """
    Validate the configuration against the appropriate schema.

    Parameters
    ----------
    raw_config : dict
        The raw configuration dictionary to validate.
    """
    schema_file = (
        "multi_mode_schema.json" if "modes" in raw_config else "single_mode_schema.json"
    )

    try:
        schema = _load_schema(schema_file)
        validate(instance=raw_config, schema=schema)

    except FileNotFoundError as e:
        logging.error(str(e))
        raise
    except ValidationError as e:
        field_path = ".".join(str(p) for p in e.path) if e.path else "root"
        logging.error(f"Schema validation failed at field '{field_path}': {e.message}")
        raise
