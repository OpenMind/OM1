import pytest
from pydantic import ValidationError

from llm.output_model import Action, CortexOutputModel  # noqa: E402


class TestAction:

    def test_action_creation_valid(self):
        """Test creating an Action with valid data."""
        action = Action(type="move", value="forward")

        assert action.type == "move"
        assert action.value == "forward"

    def test_action_required_fields(self):
        """Test that Action requires both type and value fields."""
        with pytest.raises(ValidationError):
            Action.model_validate({"value": "forward"})

        with pytest.raises(ValidationError):
            Action.model_validate({"type": "move"})

        with pytest.raises(ValidationError):
            Action.model_validate({})

    def test_action_field_descriptions(self):
        """Test that Action fields have correct descriptions."""
        action = Action(type="speak", value="hello world")

        assert action.type == "speak"
        assert action.value == "hello world"

    def test_action_type_variations(self):
        """Test various valid action types."""
        valid_types = ["move", "speak", "navigate", "follow", "stop"]

        for action_type in valid_types:
            action = Action(type=action_type, value="test")
            assert action.type == action_type
            assert action.value == "test"

    def test_action_value_variations(self):
        """Test various valid action values."""
        valid_values = ["forward", "backward", "left", "right", "hello", "10 meters"]

        for value in valid_values:
            action = Action(type="move", value=value)
            assert action.type == "move"
            assert action.value == value


class TestCortexOutputModel:

    def test_cortex_output_model_creation_valid(self):
        """Test creating a CortexOutputModel with valid data."""
        actions = [
            Action(type="move", value="forward"),
            Action(type="speak", value="hello"),
        ]
        model = CortexOutputModel(actions=actions)

        assert len(model.actions) == 2
        assert model.actions[0].type == "move"
        assert model.actions[0].value == "forward"
        assert model.actions[1].type == "speak"
        assert model.actions[1].value == "hello"

    def test_cortex_output_model_empty_actions(self):
        """Test creating a CortexOutputModel with empty actions list."""
        model = CortexOutputModel(actions=[])

        assert model.actions == []

    def test_cortex_output_model_single_action(self):
        """Test creating a CortexOutputModel with single action."""
        action = Action(type="speak", value="hello")
        model = CortexOutputModel(actions=[action])

        assert len(model.actions) == 1
        assert model.actions[0].type == "speak"
        assert model.actions[0].value == "hello"

    def test_cortex_output_model_required_actions_field(self):
        """Test that CortexOutputModel requires actions field."""
        with pytest.raises(ValidationError):
            CortexOutputModel.model_validate({})

    def test_cortex_output_model_invalid_actions_type(self):
        """Test that CortexOutputModel validates action types in list."""
        with pytest.raises(ValidationError):
            CortexOutputModel.model_validate({"actions": ["invalid_action"]})

    def test_cortex_output_model_nested_validation(self):
        """Test that nested Action validation works within CortexOutputModel."""
        with pytest.raises(ValidationError):
            CortexOutputModel.model_validate(
                {
                    "actions": [
                        {"type": "move", "value": "forward"},  # Valid
                        {"type": "speak"},  # Invalid - missing value
                    ]
                }
            )

    def test_cortex_output_model_multiple_valid_actions(self):
        """Test creating a CortexOutputModel with multiple valid actions."""
        actions = [
            Action(type="move", value="forward"),
            Action(type="speak", value="hello"),
            Action(type="navigate", value="to kitchen"),
            Action(type="follow", value="person"),
            Action(type="stop", value="now"),
        ]
        model = CortexOutputModel(actions=actions)

        assert len(model.actions) == 5
        assert all(isinstance(action, Action) for action in model.actions)
        assert [a.type for a in model.actions] == [
            "move",
            "speak",
            "navigate",
            "follow",
            "stop",
        ]

    def test_cortex_output_model_dict_serialization(self):
        """Test that models can be serialized to dict."""
        action = Action(type="speak", value="hello")
        model = CortexOutputModel(actions=[action])

        model_dict = model.model_dump()
        assert "actions" in model_dict
        assert len(model_dict["actions"]) == 1
        assert model_dict["actions"][0]["type"] == "speak"
        assert model_dict["actions"][0]["value"] == "hello"

    def test_cortex_output_model_json_serialization(self):
        """Test that models can be serialized to JSON."""
        action = Action(type="move", value="forward")
        model = CortexOutputModel(actions=[action])

        json_str = model.model_dump_json()
        assert '"type":"move"' in json_str
        assert '"value":"forward"' in json_str
        assert '"actions":[' in json_str
