import json
import os

import json5
import pytest
from jsonschema import ValidationError, validate

json_dir = os.path.join(os.path.dirname(__file__), "../../config")


@pytest.fixture(scope="session")
def single_mode_schema():
    schema_path = os.path.join(json_dir, "schema/single_mode_schema.json")
    with open(schema_path) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def multi_mode_schema():
    schema_path = os.path.join(json_dir, "schema/multi_mode_schema.json")
    with open(schema_path) as f:
        return json.load(f)


def get_all_json_files():
    return [os.path.join(json_dir, f) for f in os.listdir(json_dir) if f.endswith(".json5")]


def is_mode_config(data):
    """Determine if this is a mode-based configuration file."""
    return "modes" in data and "default_mode" in data


@pytest.mark.parametrize("json_file", get_all_json_files())
def test_json_file_valid(json_file, single_mode_schema, multi_mode_schema):
    with open(json_file) as f:
        data = json5.load(f)

    schema = multi_mode_schema if is_mode_config(data) else single_mode_schema

    try:
        validate(instance=data, schema=schema)
    except ValidationError as e:
        pytest.fail(f"{json_file} failed validation: {e.message}")
