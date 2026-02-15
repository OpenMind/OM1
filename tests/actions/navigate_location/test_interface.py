from actions.navigate_location.interface import NavigateLocation, NavigateLocationInput


def test_interface_instantiation():
    """Test that NavigateLocation can be instantiated with input/output."""
    action = NavigateLocation(
        input=NavigateLocationInput(action="kitchen"),
        output=NavigateLocationInput(action="kitchen"),
    )
    assert action.input.action == "kitchen"
    assert action.output.action == "kitchen"
