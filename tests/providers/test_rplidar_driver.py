import pytest
from providers.rplidar_driver import (
    ExpressPacket,
    RPLidarException,
    _process_scan,
)


class TestRPLidarDriver:
    def test_process_scan_valid(self):
        # Construct a valid raw packet (5 bytes)
        # Byte 0: Quality (6 bits) | !S (1 bit) | S (1 bit)
        # Quality=10 (001010), S=1 (New Scan), !S=0
        # Byte 0 = 00101001 = 0x29

        # Byte 1: Angle_q6_checkbit (1 bit)
        # Checkbit must be 1 (bit 0).
        # Angle = 64.0 degrees.
        # Angle * 64 = 4096 = 0x1000.
        # Byte 1 (bits 1-7) = lower 7 bits of (4096 >> 1) = 0.
        # Byte 1 = 0x01 (checkbit 1)

        # Byte 2: Upper bits of angle
        # Byte 2 = (4096 >> 1) >> 7 = 4096 >> 8 = 16 = 0x10?
        # Let's re-calc:
        # angle = ((_b2i(raw[1]) >> 1) + (_b2i(raw[2]) << 7)) / 64.0
        # 4096 = (raw[1] >> 1) + (raw[2] << 7)
        # 4096 = 0 + (32 << 7) -> 32 * 128 = 4096.
        # So raw[2] = 32 = 0x20.

        # Distance = 1000.0 mm.
        # distance = (_b2i(raw[3]) + (_b2i(raw[4]) << 8)) / 4.0
        # 4000 = raw[3] + (raw[4] << 8)
        # 4000 = 0x0FA0. raw[3]=0xA0, raw[4]=0x0F.

        raw = b"\x29\x01\x20\xA0\x0F"

        new_scan, quality, angle, distance = _process_scan(raw)

        assert new_scan is True
        assert quality == 10
        assert angle == 64.0
        assert distance == 1000.0

    def test_process_scan_invalid_checkbit(self):
        # Byte 1 checkbit (bit 0) is 0.
        raw = b"\x29\x00\x20\xA0\x0F"
        with pytest.raises(RPLidarException, match="Check bit not equal to 1"):
            _process_scan(raw)

    def test_process_scan_flag_mismatch(self):
        # Byte 0: S=1, !S=1 (Invalid)
        # S=1 (bit 0), !S=1 (bit 1) -> ...11 -> 0x03
        raw = b"\x03\x01\x20\xA0\x0F"
        with pytest.raises(RPLidarException, match="New scan flags mismatch"):
            _process_scan(raw)

    def test_express_packet_from_string_valid(self):
        # Express Packet is 84 bytes.
        # Sync1 = 0xA, Sync2 = 0x5.
        # Byte 0: High nibble 0xA. Low nibble part of checksum.
        # Byte 1: High nibble 0x5. Low nibble part of checksum.

        # Payload bytes 2-83 (82 bytes).
        # Checksum = XOR sum of bytes 2-83.

        # Create payload with all zeros for simplicity.
        payload = bytearray(82)

        # Calculate checksum
        chk = 0
        for b in payload:
            chk ^= b
        # chk = 0

        # Checksum verification:
        # checksum == (packet[0] & 0x0F) + ((packet[1] & 0x0F) << 4)
        # 0 == (packet[0] & 0xF) + ((packet[1] & 0xF) << 4)
        # So packet[0] low nibble = 0, packet[1] low nibble = 0.

        header = bytearray([0xA0, 0x50])
        data = header + payload

        packet = ExpressPacket.from_string(data)

        # start_angle = (packet[2] + ((packet[3] & 0x7F) << 8)) / 64
        # packet[2]=0, packet[3]=0 -> start_angle=0.0
        assert packet.start_angle == 0.0
        assert packet.new_scan == 0

    def test_express_packet_invalid_sync(self):
        data = bytearray(84)
        data[0] = 0x00  # Invalid sync
        data[1] = 0x50

        with pytest.raises(ValueError, match="try to parse corrupted data"):
            ExpressPacket.from_string(data)

    def test_express_packet_invalid_checksum(self):
        # Valid syncs
        data = bytearray(84)
        data[0] = 0xA0
        data[1] = 0x50
        # Payload all zeros -> checksum 0. Matches header.

        # Corrupt payload
        data[2] = 0x01
        # Now checksum of payload is 1. Header says 0.

        with pytest.raises(ValueError, match="Invalid checksum"):
            ExpressPacket.from_string(data)