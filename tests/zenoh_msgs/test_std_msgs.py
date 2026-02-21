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

    def test_returns_header(self):
        header = prepare_header()
        assert isinstance(header, Header)

    def test_default_frame_id_is_empty(self):
        header = prepare_header()
        assert header.frame_id == ""

    def test_custom_frame_id(self):
        header = prepare_header(frame_id="odom")
        assert header.frame_id == "odom"

    def test_timestamp_seconds_is_int(self):
        header = prepare_header()
        assert isinstance(header.stamp.sec, int)
        assert header.stamp.sec > 0

    def test_timestamp_nanoseconds_is_int(self):
        header = prepare_header()
        assert isinstance(header.stamp.nanosec, int)
        assert 0 <= header.stamp.nanosec < 1_000_000_000

    def test_timestamp_is_recent(self):
        import time

        before = int(time.time()) - 1
        header = prepare_header()
        after = int(time.time()) + 1
        assert before <= header.stamp.sec <= after

    def test_fractional_nanoseconds(self):
        header = prepare_header()
        assert 0 <= header.stamp.nanosec < 1_000_000_000
