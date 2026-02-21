from unittest.mock import patch

from zenoh_msgs.idl.std_msgs import (
    ColorRGBA,
    Duration,
    Header,
    String,
    Time,
    prepare_header,
)


class TestTime:
    """Tests for the Time dataclass."""

    def test_create_time(self):
        """Test Time creation with sec and nanosec."""
        t = Time(sec=100, nanosec=500000000)
        assert t.sec == 100
        assert t.nanosec == 500000000


class TestDuration:
    """Tests for the Duration dataclass."""

    def test_create_duration(self):
        """Test Duration creation with sec and nanosec."""
        d = Duration(sec=10, nanosec=250000000)
        assert d.sec == 10
        assert d.nanosec == 250000000


class TestHeader:
    """Tests for the Header dataclass."""

    def test_create_header(self):
        """Test Header creation with stamp and frame_id."""
        stamp = Time(sec=1, nanosec=0)
        header = Header(stamp=stamp, frame_id="base_link")
        assert header.stamp.sec == 1
        assert header.frame_id == "base_link"

    def test_header_empty_frame_id(self):
        """Test Header with empty frame_id."""
        stamp = Time(sec=0, nanosec=0)
        header = Header(stamp=stamp, frame_id="")
        assert header.frame_id == ""


class TestColorRGBA:
    """Tests for the ColorRGBA dataclass."""

    def test_create_color(self):
        """Test ColorRGBA creation."""
        color = ColorRGBA(r=1.0, g=0.5, b=0.0, a=1.0)
        assert color.r == 1.0
        assert color.g == 0.5
        assert color.b == 0.0
        assert color.a == 1.0


class TestString:
    """Tests for the String dataclass."""

    def test_create_string(self):
        """Test String creation."""
        s = String(data="hello")
        assert s.data == "hello"

    def test_empty_string(self):
        """Test String with empty data."""
        s = String(data="")
        assert s.data == ""


class TestPrepareHeader:
    """Tests for the prepare_header function."""

    @patch("zenoh_msgs.idl.std_msgs.time.time", return_value=1700000000.5)
    def test_returns_header(self, mock_time):
        """Test that prepare_header returns a Header instance."""
        header = prepare_header()
        assert isinstance(header, Header)

    @patch("zenoh_msgs.idl.std_msgs.time.time", return_value=1700000000.5)
    def test_default_frame_id_is_empty(self, mock_time):
        """Test that the default frame_id is an empty string."""
        header = prepare_header()
        assert header.frame_id == ""

    @patch("zenoh_msgs.idl.std_msgs.time.time", return_value=1700000000.5)
    def test_custom_frame_id(self, mock_time):
        """Test prepare_header with a custom frame_id."""
        header = prepare_header(frame_id="odom")
        assert header.frame_id == "odom"

    @patch("zenoh_msgs.idl.std_msgs.time.time", return_value=1700000000.5)
    def test_timestamp_seconds(self, mock_time):
        """Test that the seconds portion of the timestamp is correct."""
        header = prepare_header()
        assert header.stamp.sec == 1700000000

    @patch("zenoh_msgs.idl.std_msgs.time.time", return_value=1700000000.5)
    def test_timestamp_nanoseconds(self, mock_time):
        """Test that the nanoseconds portion of the timestamp is correct."""
        header = prepare_header()
        assert header.stamp.nanosec == 500000000

    @patch("zenoh_msgs.idl.std_msgs.time.time", return_value=1000.0)
    def test_zero_nanoseconds(self, mock_time):
        """Test timestamp when time has no fractional part."""
        header = prepare_header()
        assert header.stamp.sec == 1000
        assert header.stamp.nanosec == 0

    @patch("zenoh_msgs.idl.std_msgs.time.time", return_value=1700000000.123456789)
    def test_fractional_nanoseconds(self, mock_time):
        """Test nanosecond conversion from fractional seconds."""
        header = prepare_header()
        assert header.stamp.sec == 1700000000
        expected_ns = int(0.123456789 * 1000000000)
        assert abs(header.stamp.nanosec - expected_ns) < 100
