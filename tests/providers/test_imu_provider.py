import pytest

from providers.imu_provider import IMUProvider


@pytest.fixture(autouse=True)
def reset_singleton():
    IMUProvider.reset()
    yield
    IMUProvider.reset()


def test_initial_state():
    provider = IMUProvider()
    state = provider.state
    assert state["accel_x"] == 0.0
    assert state["accel_y"] == 0.0
    assert state["accel_z"] == 0.0
    assert state["gyro_x"] == 0.0
    assert state["gyro_y"] == 0.0
    assert state["gyro_z"] == 0.0
    assert state["roll"] == 0.0
    assert state["pitch"] == 0.0
    assert state["yaw"] == 0.0
    assert state["is_fallen"] is False
    assert state["impact_detected"] is False
    assert state["last_update"] == 0.0


def test_update_normal():
    provider = IMUProvider()
    provider.update(0.1, 0.2, 9.8, 0.0, 0.0, 0.0, 1.0, 2.0, 90.0)
    state = provider.state
    assert state["accel_x"] == 0.1
    assert state["accel_y"] == 0.2
    assert state["accel_z"] == 9.8
    assert state["roll"] == 1.0
    assert state["pitch"] == 2.0
    assert state["yaw"] == 90.0
    assert state["is_fallen"] is False
    assert state["impact_detected"] is False
    assert state["last_update"] > 0.0


def test_fall_detection_roll():
    provider = IMUProvider()
    provider.update(0.0, 0.0, 9.8, 0.0, 0.0, 0.0, 50.0, 0.0, 0.0)
    assert provider.state["is_fallen"] is True


def test_fall_detection_pitch():
    provider = IMUProvider()
    provider.update(0.0, 0.0, 9.8, 0.0, 0.0, 0.0, 0.0, -50.0, 0.0)
    assert provider.state["is_fallen"] is True


def test_no_fall_within_threshold():
    provider = IMUProvider()
    provider.update(0.0, 0.0, 9.8, 0.0, 0.0, 0.0, 44.9, 0.0, 0.0)
    assert provider.state["is_fallen"] is False


def test_impact_detection():
    provider = IMUProvider()
    provider.update(15.0, 15.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert provider.state["impact_detected"] is True


def test_no_impact_within_threshold():
    provider = IMUProvider()
    provider.update(0.1, 0.2, 9.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert provider.state["impact_detected"] is False


def test_reset_alerts():
    provider = IMUProvider()
    provider.update(15.0, 15.0, 0.0, 0.0, 0.0, 0.0, 50.0, 0.0, 0.0)
    assert provider.state["is_fallen"] is True
    assert provider.state["impact_detected"] is True
    provider.reset_alerts()
    assert provider.state["is_fallen"] is False
    assert provider.state["impact_detected"] is False


def test_singleton():
    p1 = IMUProvider()
    p2 = IMUProvider()
    p1.update(0.0, 0.0, 9.8, 0.0, 0.0, 0.0, 5.0, 3.0, 10.0)
    assert p2.state["roll"] == 5.0


def test_custom_thresholds():
    provider = IMUProvider()
    provider.fall_threshold = 30.0
    provider.impact_threshold = 10.0
    provider.update(0.0, 0.0, 9.8, 0.0, 0.0, 0.0, 35.0, 0.0, 0.0)
    assert provider.state["is_fallen"] is True


def test_stop():
    provider = IMUProvider()
    provider.stop()


def test_negative_roll_fall():
    provider = IMUProvider()
    provider.update(0.0, 0.0, 9.8, 0.0, 0.0, 0.0, -50.0, 0.0, 0.0)
    assert provider.state["is_fallen"] is True


def test_gyro_data_stored():
    provider = IMUProvider()
    provider.update(0.0, 0.0, 9.8, 1.1, 2.2, 3.3, 0.0, 0.0, 0.0)
    state = provider.state
    assert state["gyro_x"] == 1.1
    assert state["gyro_y"] == 2.2
    assert state["gyro_z"] == 3.3
