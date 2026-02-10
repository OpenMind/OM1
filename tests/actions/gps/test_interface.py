from actions.gps.interface import GPS, GPSAction, GPSInput


class TestGPSAction:
    """Tests for the GPSAction enum."""

    def test_gps_action_values(self):
        """Test that GPSAction enum has correct values."""
        assert GPSAction.SHARE_LOCATION.value == "share location"
        assert GPSAction.IDLE.value == "idle"

    def test_gps_action_is_string_enum(self):
        """Test that GPSAction values are strings."""
        for action in GPSAction:
            assert isinstance(action.value, str)

    def test_gps_action_count(self):
        """Test that GPSAction has expected number of actions."""
        assert len(GPSAction) == 2


class TestGPSInput:
    """Tests for the GPSInput dataclass."""

    def test_gps_input_creation(self):
        """Test creating GPSInput with valid action."""
        gps_input = GPSInput(action=GPSAction.SHARE_LOCATION)
        assert gps_input.action == GPSAction.SHARE_LOCATION

    def test_gps_input_all_actions(self):
        """Test creating GPSInput with all possible actions."""
        for action in GPSAction:
            gps_input = GPSInput(action=action)
            assert gps_input.action == action


class TestGPS:
    """Tests for the GPS interface."""

    def test_gps_creation(self):
        """Test creating GPS with input and output."""
        gps_input = GPSInput(action=GPSAction.SHARE_LOCATION)
        gps = GPS(input=gps_input, output=gps_input)
        assert gps.input == gps_input
        assert gps.output == gps_input

    def test_gps_different_input_output(self):
        """Test creating GPS with different input and output."""
        input_gps = GPSInput(action=GPSAction.SHARE_LOCATION)
        output_gps = GPSInput(action=GPSAction.IDLE)
        gps = GPS(input=input_gps, output=output_gps)
        assert gps.input.action == GPSAction.SHARE_LOCATION
        assert gps.output.action == GPSAction.IDLE
