"""

A parser for TBS crsf data.

Based on Bryan Mayland's CRSF "Python Parser"
https://github.com/crsf-wg/crsf/wiki/Python-Parser

Uses the public CRSF documentation:
https://github.com/tbs-fpv/tbs-crsf-spec

Known issues:

Occasionally, one of the RC_Channels has a value of >2000, which is noise.
This is a problem, since it tells the receiver that one of the switches was
activated, even if they were not. I'm rejecting all RC data with values >2000.
A typical range for a valid RC signal is 174 to 1800 - this is true for both
sticks and switches.

Run like this:

uv run parse_crsf_radio.py

"""

import argparse
from enum import IntEnum

import serial


class PacketsTypes(IntEnum):
    GPS = 0x02
    VARIO = 0x07
    BATTERY_SENSOR = 0x08
    BARO_ALT = 0x09
    HEARTBEAT = 0x0B
    VIDEO_TRANSMITTER = 0x0F
    LINK_STATISTICS = 0x14
    RC_CHANNELS_PACKED = 0x16
    ATTITUDE = 0x1E
    FLIGHT_MODE = 0x21
    DEVICE_INFO = 0x29
    CONFIG_READ = 0x2C
    CONFIG_WRITE = 0x2D
    RADIO_ID = 0x3A
    PARAMETER_PING = 0x28
    SYNC_BYTE = 0xC8


def crc8_dvb_s2(crc, a) -> int:
    crc = crc ^ a
    for ii in range(8):
        if crc & 0x80:
            crc = (crc << 1) ^ 0xD5
        else:
            crc = crc << 1
    return crc & 0xFF


def crc8_data(data) -> int:
    crc = 0
    for a in data:
        crc = crc8_dvb_s2(crc, a)
    return crc


def crsf_validate_frame(frame) -> bool:
    # print(f"\nfull frame: {frame.hex()}")
    # print(f"payload len: {len(frame[2:-1])}")
    # print(f"payload: {frame[2:-1].hex()}")
    # checksum = crc8_data(frame[2:-1])
    # print(f"CRC calculated: {hex(checksum)}")
    # print(f"CRC at data end: {hex(frame[-1])}")
    return crc8_data(frame[2:-1]) == frame[-1]


def signed_byte(b):
    return b - 256 if b >= 128 else b


def n(val):
    # reject garbage values and noise (anything < 174 or > 1806)

    res = val - 174
    res = res / 1632

    if res > 1.0:
        res = 1.0
    elif res < 0.0:
        res = 0.0

    return round(res, 2)


def handleCrsfPacket(ptype, data):
    if ptype == PacketsTypes.RADIO_ID and data[5] == 0x10:
        # print(f"OTX sync")
        pass
    elif ptype == PacketsTypes.LINK_STATISTICS:
        # LINK_STATISTICS = 0x14
        rssi1 = signed_byte(data[3])
        rssi2 = signed_byte(data[4])
        lq = data[5]
        snr = signed_byte(data[6])
        antenna = data[7]
        mode = data[8]
        power = data[9]
        # telemetry strength
        downlink_rssi = signed_byte(data[10])
        downlink_lq = data[11]
        downlink_snr = signed_byte(data[12])
        print(
            f"RSSI={rssi1}/{rssi2}dBm LQ={lq:03} mode={mode} "
            f"ant={antenna} snr={snr} power={power} drssi={downlink_rssi} dlq={downlink_lq} dsnr={downlink_snr}"
        )
    elif ptype == PacketsTypes.ATTITUDE:
        pitch = int.from_bytes(data[3:5], byteorder="big", signed=True) / 10000.0
        roll = int.from_bytes(data[5:7], byteorder="big", signed=True) / 10000.0
        yaw = int.from_bytes(data[7:9], byteorder="big", signed=True) / 10000.0
        print(f"Attitude: Pitch={pitch:0.2f} Roll={roll:0.2f} Yaw={yaw:0.2f} (rad)")
    elif ptype == PacketsTypes.FLIGHT_MODE:
        packet = "".join(map(chr, data[3:-2]))
        print(f"Flight Mode: {packet}")
    elif ptype == PacketsTypes.BATTERY_SENSOR:
        vbat = int.from_bytes(data[3:5], byteorder="big", signed=True) / 10.0
        curr = int.from_bytes(data[5:7], byteorder="big", signed=True) / 10.0
        mah = data[7] << 16 | data[8] << 7 | data[9]
        pct = data[10]
        print(f"Battery: {vbat:0.2f}V {curr:0.1f}A {mah}mAh {pct}%")
    elif ptype == PacketsTypes.BARO_ALT:
        print("BaroAlt: ")
    elif ptype == PacketsTypes.DEVICE_INFO:
        packet = " ".join(map(hex, data))
        print(f"Device Info: {packet}")
    elif data[2] == PacketsTypes.GPS:
        lat = int.from_bytes(data[3:7], byteorder="big", signed=True) / 1e7
        lon = int.from_bytes(data[7:11], byteorder="big", signed=True) / 1e7
        gspd = int.from_bytes(data[11:13], byteorder="big", signed=True) / 36.0
        hdg = int.from_bytes(data[13:15], byteorder="big", signed=True) / 100.0
        alt = int.from_bytes(data[15:17], byteorder="big", signed=True) - 1000
        sats = data[17]
        print(
            f"GPS: Pos={lat} {lon} GSpd={gspd:0.1f}m/s Hdg={hdg:0.1f} Alt={alt}m Sats={sats}"
        )
    elif ptype == PacketsTypes.VARIO:
        vspd = int.from_bytes(data[3:5], byteorder="big", signed=True) / 10.0
        print(f"VSpd: {vspd:0.1f}m/s")
    elif ptype == PacketsTypes.RC_CHANNELS_PACKED:
        # RC_CHANNELS_PACKED = 0x16
        packet = data[2:-1]
        packet = packet[1:-1]  # remove type and crc
        packet_bin_8 = ["{0:08b}".format(i)[::-1] for i in packet]  # [::-1] reverse
        packet_bin_full = "".join(packet_bin_8)
        packet_bin_11 = [packet_bin_full[11 * i : 11 * (i + 1)] for i in range(16)]
        rc_packet = [int(b[::-1], 2) for b in packet_bin_11]

        # sometimes there is noise in the packets - anything above a value of 2000 is garbage
        if max(rc_packet) > 2000:
            return

        # print(f"Control packet: {rc_packet}")
        lud = n(rc_packet[2])
        llr = n(rc_packet[3])
        rud = n(rc_packet[0])
        rlr = n(rc_packet[1])
        swA = n(rc_packet[4])
        swA = "in" if swA > 0.6 else "off"
        swB = n(rc_packet[5])
        swB = "front" if swB <= 0.33 else "back" if swB >= 0.66 else "middle"
        swC = n(rc_packet[6])
        swC = "front" if swC <= 0.33 else "back" if swC >= 0.66 else "middle"
        swD = n(rc_packet[7])
        swD = "in" if swD > 0.6 else "off"
        swE = n(rc_packet[8])
        swE = "on" if swE > 0.6 else "off"
        swF = n(rc_packet[9])
        swF = "on" if swF > 0.6 else "off"
        print(
            f"LUD:{lud},LLR:{llr},RUD:{rud},RLR:{rlr},SWA:{swA},SWB:{swB},SWC:{swC},SWD:{swD},SWE:{swE},SWF:{swF}"
        )
    elif ptype == PacketsTypes.PARAMETER_PING:
        packet = " ".join(map(hex, data))
        print(f"PING 0x{ptype:02x}: {packet}")
    else:
        packet = " ".join(map(hex, data))
        print(f"Unknown 0x{ptype:02x}: {packet}")


parser = argparse.ArgumentParser()
parser.add_argument(
    "-P", "--port", default="/dev/cu.usbserial-B003ABY3", required=False
)
parser.add_argument("-b", "--baud", default=420000, required=False)
parser.add_argument(
    "-t",
    "--tx",
    required=False,
    default=False,
    action="store_true",
    help="Enable sending CHANNELS_PACKED every 20ms (all channels 1500us)",
)
args = parser.parse_args()

with serial.Serial(
    args.port,
    args.baud,
    timeout=1,
) as ser:
    input = bytearray()
    while True:
        if ser.in_waiting > 0:
            input.extend(ser.read(ser.in_waiting))
        while len(input) > 2:
            expected_len = input[1] + 2
            if expected_len > 64 or expected_len < 4:
                input = bytearray()
            elif len(input) >= expected_len:
                single = input[:expected_len]  # copy out this whole packet
                input = input[expected_len:]  # and remove it from the buffer
                if single[0] == PacketsTypes.SYNC_BYTE:
                    if not crsf_validate_frame(single):
                        packet = " ".join(map(hex, single))
                        print(f"crc error: {packet}")
                    else:
                        handleCrsfPacket(single[2], single)
            else:
                break
