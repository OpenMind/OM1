from actions.emergency_alert.interface import EmergencyAlert, EmergencyAlertInput


class TestEmergencyAlertInput:
    """Tests for the EmergencyAlertInput dataclass."""

    def test_emergency_alert_input_creation(self):
        """Test creating EmergencyAlertInput with valid message."""
        alert_input = EmergencyAlertInput(action="Person detected in restricted area")
        assert alert_input.action == "Person detected in restricted area"

    def test_emergency_alert_input_empty_string(self):
        """Test creating EmergencyAlertInput with empty string."""
        alert_input = EmergencyAlertInput(action="")
        assert alert_input.action == ""


class TestEmergencyAlert:
    """Tests for the EmergencyAlert interface."""

    def test_emergency_alert_creation(self):
        """Test creating EmergencyAlert with input and output."""
        alert_input = EmergencyAlertInput(action="Injury detected")
        alert = EmergencyAlert(input=alert_input, output=alert_input)
        assert alert.input == alert_input
        assert alert.output == alert_input

    def test_emergency_alert_different_input_output(self):
        """Test creating EmergencyAlert with different input and output."""
        input_alert = EmergencyAlertInput(action="Emergency detected")
        output_alert = EmergencyAlertInput(action="Alert sent")
        alert = EmergencyAlert(input=input_alert, output=output_alert)
        assert alert.input.action == "Emergency detected"
        assert alert.output.action == "Alert sent"
