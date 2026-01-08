from backgrounds.plugins.data_analytics import (
    DataAnalytics,
    DataAnalyticsConfig,
)


class TestDataAnalyticsConfig:
    """Tests for DataAnalyticsConfig."""

    def test_config_defaults(self):
        """Test configuration defaults."""
        config = DataAnalyticsConfig()
        assert config.enable_ml is True
        assert config.window_size == 100
        assert config.contamination == 0.1
        assert config.enable_api is True
        assert config.api_port == 8080

    def test_config_custom_values(self):
        """Test configuration with custom values."""
        config = DataAnalyticsConfig(
            enable_ml=False,
            window_size=50,
            contamination=0.2,
            enable_api=False,
            api_port=9000,
        )
        assert config.enable_ml is False
        assert config.window_size == 50
        assert config.contamination == 0.2
        assert config.enable_api is False
        assert config.api_port == 9000


class TestDataAnalytics:
    """Tests for DataAnalytics background process."""

    def test_initialization(self):
        """Test initializing DataAnalytics."""
        config = DataAnalyticsConfig(enable_api=False)
        analytics = DataAnalytics(config)
        assert analytics.analytics_provider is not None
        assert analytics.config.enable_ml is True

    def test_process_sensor_data(self):
        """Test processing sensor data."""
        config = DataAnalyticsConfig(enable_api=False, enable_ml=False)
        analytics = DataAnalytics(config)
        reading, is_anomaly = analytics.process_sensor_data(
            temperature=25.0, humidity=60.0, air_quality=100.0
        )
        assert reading.temperature == 25.0
        assert reading.humidity == 60.0
        assert reading.air_quality == 100.0
        assert isinstance(is_anomaly, bool)

    def test_process_sensor_data_partial(self):
        """Test processing sensor data with partial values."""
        config = DataAnalyticsConfig(enable_api=False, enable_ml=False)
        analytics = DataAnalytics(config)
        reading, is_anomaly = analytics.process_sensor_data(temperature=25.0)
        assert reading.temperature == 25.0
        assert reading.humidity is None
        assert reading.air_quality is None

    def test_get_metrics(self):
        """Test getting metrics from analytics provider."""
        config = DataAnalyticsConfig(enable_api=False, enable_ml=False)
        analytics = DataAnalytics(config)
        # Add some data
        for i in range(5):
            analytics.process_sensor_data(temperature=20.0 + i, humidity=50.0 + i)
        metrics = analytics.analytics_provider.get_metrics()
        assert metrics.total_readings == 5
        assert metrics.avg_temperature is not None
