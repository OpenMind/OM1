from zenoh_msgs.idl.geographic_msgs import GeoPoint, GeoPointStamped
from zenoh_msgs.idl.std_msgs import Header, Time


class TestGeoPoint:
    """Tests for the GeoPoint dataclass."""

    def test_create_geopoint(self):
        """Test GeoPoint creation with latitude, longitude, altitude."""
        gp = GeoPoint(latitude=1.0, longitude=2.0, altitude=3.0)
        assert gp.latitude == 1.0
        assert gp.longitude == 2.0
        assert gp.altitude == 3.0

    def test_geopoint_negative_values(self):
        """Test GeoPoint with negative coordinates."""
        gp = GeoPoint(latitude=-90.0, longitude=-180.0, altitude=-100.0)
        assert gp.latitude == -90.0
        assert gp.longitude == -180.0
        assert gp.altitude == -100.0

    def test_geopoint_zero(self):
        """Test GeoPoint with zero values."""
        gp = GeoPoint(latitude=0.0, longitude=0.0, altitude=0.0)
        assert gp.latitude == 0.0
        assert gp.longitude == 0.0
        assert gp.altitude == 0.0


class TestGeoPointStamped:
    """Tests for the GeoPointStamped dataclass."""

    def test_create_geopointstamped(self):
        """Test GeoPointStamped creation with header and position."""
        stamp = Time(sec=1, nanosec=0)
        header = Header(stamp=stamp, frame_id="map")
        position = GeoPoint(latitude=1.0, longitude=2.0, altitude=3.0)
        gps = GeoPointStamped(header=header, position=position)
        assert gps.header.frame_id == "map"
        assert gps.position.latitude == 1.0
        assert gps.position.longitude == 2.0
        assert gps.position.altitude == 3.0
