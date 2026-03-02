from unittest.mock import MagicMock, patch

import pytest

from backgrounds.base import BackgroundConfig
from backgrounds.plugins.imu_fall_detector import IMUFallDetector
from providers.imu_provider import IMUProvider


@pytest.fixture(autouse=True)
def reset_singleton():
    IMUProvider.reset()
    yield
    IMUProvider.reset()


@pytest.fixture
def config():
    return BackgroundConfig()


@pytest.fixture
def detector(config):
    with patch("backgrounds.plugins.imu_fall_detector.ContextProvider"):
        d = IMUFallDetector(config)
        d.context_provider = MagicMock()
        return d


def test_init(detector):
    assert detector._fall_reported is False
    assert detector._impact_reported is False


def test_run_normal_state(detector):
    IMUProvider().update(0.0, 0.0, 9.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    detector.run()
    detector.context_provider.update_context.assert_not_called()


def test_run_fall_detected(detector):
    IMUProvider().update(0.0, 0.0, 9.8, 0.0, 0.0, 0.0, 50.0, 0.0, 0.0)
    detector.run()
    detector.context_provider.update_context.assert_called_once()
    call_args = detector.context_provider.update_context.call_args[0][0]
    assert call_args["imu_fall_detected"] is True
    assert detector._fall_reported is True


def test_run_fall_not_reported_twice(detector):
    IMUProvider().update(0.0, 0.0, 9.8, 0.0, 0.0, 0.0, 50.0, 0.0, 0.0)
    detector.run()
    detector.run()
    assert detector.context_provider.update_context.call_count == 1


def test_run_fall_resolved(detector):
    IMUProvider().update(0.0, 0.0, 9.8, 0.0, 0.0, 0.0, 50.0, 0.0, 0.0)
    detector.run()
    detector.context_provider.update_context.reset_mock()
    IMUProvider().update(0.0, 0.0, 9.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    IMUProvider().reset_alerts()
    detector.run()
    call_args = detector.context_provider.update_context.call_args[0][0]
    assert call_args["imu_fall_detected"] is False
    assert detector._fall_reported is False


def test_run_impact_detected(detector):
    IMUProvider().update(15.0, 15.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    detector.run()
    call_args_list = detector.context_provider.update_context.call_args_list
    impact_calls = [c for c in call_args_list if "imu_impact_detected" in c[0][0]]
    assert len(impact_calls) == 1
    assert impact_calls[0][0][0]["imu_impact_detected"] is True
    assert detector._impact_reported is True


def test_run_impact_not_reported_twice(detector):
    IMUProvider().update(15.0, 15.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    detector.run()
    detector.run()
    impact_calls = [
        c
        for c in detector.context_provider.update_context.call_args_list
        if "imu_impact_detected" in c[0][0]
    ]
    assert len(impact_calls) == 1


def test_run_impact_resolved(detector):
    IMUProvider().update(15.0, 15.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    detector.run()
    detector.context_provider.update_context.reset_mock()
    IMUProvider().reset_alerts()
    detector.run()
    call_args = detector.context_provider.update_context.call_args[0][0]
    assert call_args["imu_impact_detected"] is False
    assert detector._impact_reported is False


def test_stop(detector):
    detector.stop()


def test_run_fall_and_impact_together(detector):
    IMUProvider().update(15.0, 15.0, 0.0, 0.0, 0.0, 0.0, 50.0, 0.0, 0.0)
    detector.run()
    assert detector._fall_reported is True
    assert detector._impact_reported is True
    assert detector.context_provider.update_context.call_count == 2
