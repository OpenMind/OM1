---
title: GPS and Compass Support
description: "Arduino GPS and Compass Support"
---

## Hardware

A solution for outdoor localization and basic navigation is an Arduino board, such as the [Adafruit Feather nRF52840 Sense](https://www.adafruit.com/product/4516) with a GPS shield, such as the [Adafruit Ultimate GPS FeatherWing](https://www.adafruit.com/product/3133). The `Sense` provides a micro controller, Bluetooth, a 6DoF IMU, and a magnetometer (a LIS3MDL). The `FeatherWing` provides a GPS.

## Assembly

Solder the `FeatherWing` to the `Sense`. Add support for the [nRF52](https://learn.adafruit.com/adafruit-circuit-playground-bluefruit/arduino-support-setup) to the Arduino IDE. Then, add the board - it's the `Adafruit nRF52840 Sense`. The Arduino IDE will probably need to also install the nRF core library (`adafruit:nrf52@1.6.1`). You should now be able to connect to the board.

## Software

Add the following libraries to the Arduino IDE:

* `Adafruit GPS library`
* `Adafruit LIS3MDL`
* `Adafruit_LSM6DS`

Run the sketch provided in `/system_hw_test/gps_mag`. It yields all the data for tilt compensated magnetic heading, a full AHRS solution (yaw, tilt, roll) and a GPS location, as well as altitude and velocity.

*WARNING* The direction and AHRS data will be entirely incorrect unless you calibrate your magnetometer, gyro, and accelerometer.

## Calibration

In the `/system_hw_test/gps_mag.ino`, see:

```c
/* For calibration, set CALIBRATION to true, run the script, close
the Arduino IDE, close the serial monitor, and open MotionCal.

MotionCal can be downloaded here:

https://github.com/PaulStoffregen/MotionCal

If you do not see your Arduino in MotionCal's serial port
drop down menu, follow these instructions to compile MotionCal
for your OS and platform

https://github.com/PaulStoffregen/MotionCal/issues/11#issuecomment-2412937251
*/

bool CALIBRATION = false;
```

Once you have established your 15 (3+9+3) calibration coefficients (see https://github.com/PaulStoffregen/MotionCal), determine the device string of your sensor (provided in the initial serial debug output), and add your calibration coefficients following the example given in `gap_mag.ino`:

```c
if (strcmp(DeviceID, "38e4bfd6") == 0) {
	float mh[] = {24.420, -16.87, -3.880}; // in uTesla
	float ms[] = { 0.959,  0.050, -0.038,  \
	               0.050,  1.078,  0.002,  \
	              -0.038,  0.002,  0.970};
	float gc[] = { 0.000,  0.020,  0.010}; // in Radians/s
	memcpy(mag_hardiron,  mh, sizeof(mag_hardiron));
	memcpy(mag_softiron,  ms, sizeof(mag_softiron));
	memcpy(gyro_zerorate, gc, sizeof(gyro_zerorate));
} else {
	Serial.println("CAUTION: Magnetometer not calibrated - code will yield garbage - please calibrate your Magnetometer and IMU");
}
```

*WARNING* If you use different sensor hardware (and not the nRF52840 Sense with the FeatherWing), you will need to adapt all data flowing from your gyros, accelerators, and magnetometer to (1) have the correct units, (2) follow the correct sign conventions, and (3) accommodate your pcb/sensor geometry as well as how the sensors are mounted relative to your robot. This tends to be extremely tedious and time consuming, so we recommend using the Sense/FeatherWing for which this driver has been developed.

## Usage

Once your Sense/FeatherWing is streaming data on the serial line, it can be fed into OM1. See `src/inputs/plugins/serial_reader` for an example of how to do that.

### Finding the Arduino on Linux

When connecting to the Arduino via USB, you should see the Arduino serial port appear as `/dev/ttyACM0` (sometimes the number can be different, for example `/dev/ttyACM1`). Run:

```bash
sudo dmesg | grep ttyACM*
```
and you should see it. If you're not sure which tty device is the Arduino board, run `sudo dmesg` and looks for entries with "Arduino" in it. This way you will easily spot the serial device name of your Arduino. You can read the data with:

```bash
screen /dev/ttyACM0 115200
```

Note: on a typical TurtleBot4, the RPLIDAR uses `/dev/ttyUSB0`. The TurtleBot4 **assumes** that the RPLIDAR is accessible at `/dev/ttyUSB0`. If you change which USB port the LIDAR is plugged into, the LIDAR will fail.

### Finding the Arduino GPS Feather on Mac

Determine the serial port the sensor is using:

```bash
ls /dev/tty.*
# or
ls /dev/cu.*
```
It should be something similar to `/dev/cu.usbmodem8401`. Even on a mac, you can use `screen` to see the data: `screen /dev/cu.usbmodem21201 115200`

### Testing

```bash
uv run src/run.py test_gps
```

### RTCM / RTK Precision GPS

An RTK GPS system can provide centimeter accuracy localization. In a typical setup, an RTK-compatible GPS receiver (such as the [u-blox ZED-F9P/simpleRTK2B](https://www.ardusimple.com/product/simplertk2b/) accepts `RTCM` messages over XBee or the internet, and uses those data to correct (and greatly improve) its location estimate. The `simpleRTK2B` has two usb ports, one labeled `power + GPS` and the other `power + XBee`. The first port provides a full set of NEMA and UBX messages, including GPGGA, GNRMC, GNGLL, and UBX(NAV-PVT). The `power + XBee` port provides direct access to received RTK correction messages, such as `1005` and `1074`. In standard usage, a rover would use the NEMA messages to track its location whilst monitoring the status, mode, and quality fields.

```bash
<NMEA(GNGGA, time=23:56:12, lat=12.4003072367, NS=N, lon=-2.1187726617, EW=W, quality=5, numSV=12, HDOP=0.74, alt=26.189, altUnit=M, sep=-30.024, sepUnit=M, diffAge=3.0, diffStation=0)>
```
This GNGGA message shows a high quality RTK fix (quality=5):

* 1 autonomous (standard accuracy) solution
* 2 code-differential (DGNSS, SBAS) solution
* 4 fixed RTK
* 5 float RTK

This GNRC message shows a valid status(`A`) using RTK mode with fixed integers (`posMode=R`)

```bash
<NMEA(GNRMC, time=23:35:31.400000, status=A, lat=12.40030754, NS=N, lon=-2.1187695133, EW=W, spd=0.036, cog=, date=2025-06-01, mv=, mvEW=, posMode=R)>
```

Status indicator:
A: Data valid
V: Data not valid

Mode indicator:
A: Autonomous mode
D: Differential mode
P: Precise
E: Estimated (dead reckoning) mode
M: Manual input mode
S: Simulator mode
R: Fixed RTK. RTK mode with fixed integers
F: Float RTK. RTK mode with floating integers.
N: Data not valid

### RTK Data Plotting

The best software to use is `pygpsclient`:

```bash
python3 -m venv pygpsclient
source pygpsclient/bin/activate
python3 -m pip install --upgrade pygpsclient
pygpsclient
```
