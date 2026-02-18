"""Tests for the EmergencyAlert action interface."""

from actions.emergency_alert.interface import EmergencyAlert, EmergencyAlertInput


class TestEmergencyAlertInput:
    """Tests for the EmergencyAlertInput dataclass."""

    def test_emergency_alert_input_creation(self):
        """Test creating EmergencyAlertInput with action."""
        alert_input = EmergencyAlertInput(action="Fire detected in building!")
        assert alert_input.action == "Fire detected in building!"

    def test_emergency_alert_input_security(self):
        """Test creating EmergencyAlertInput for security alert."""
        alert_input = EmergencyAlertInput(
            action="Unknown person detected in restricted area"
        )
        assert alert_input.action == "Unknown person detected in restricted area"


class TestEmergencyAlert:
    """Tests for the EmergencyAlert interface."""

    def test_emergency_alert_creation(self):
        """Test creating EmergencyAlert with input and output."""
        alert_input = EmergencyAlertInput(action="Emergency!")
        alert = EmergencyAlert(input=alert_input, output=alert_input)
        assert alert.input == alert_input
        assert alert.output == alert_input

    def test_emergency_alert_different_input_output(self):
        """Test creating EmergencyAlert with different input and output."""
        input_alert = EmergencyAlertInput(action="Input alert")
        output_alert = EmergencyAlertInput(action="Output alert")
        alert = EmergencyAlert(input=input_alert, output=output_alert)
        assert alert.input.action == "Input alert"
        assert alert.output.action == "Output alert"
