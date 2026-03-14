---
title: PIR Motion Sensor (HC-SR501)
description: "HC-SR501 Passive Infrared Motion Sensor"
icon: sensor
---

## Overview

The HC-SR501 is a Passive Infrared (PIR) sensor that detects motion by measuring changes in infrared radiation emitted by people and animals. It outputs a simple digital HIGH/LOW signal, making it an efficient and low-power trigger for robot guard and security modes.

Because the sensor consumes only ~1 mA in standby, it is ideal as a **wake-up trigger**: keep the PIR active continuously, then activate heavier subsystems (camera, VLM, LLM) only when motion is detected.

## Hardware

| Parameter | Value |
|-----------|-------|
| Supply voltage | 4.5 V – 20 V (typically 5 V) |
| Output voltage | 3.3 V HIGH / 0 V LOW |
| Detection range | Up to 7 m |
| Detection angle | ~120° cone |
| Trigger hold time | 0.3 s – ~200 s (adjustable via onboard potentiometer) |
| Current draw | ~1 mA standby |

The board has two potentiometers:
- **Sensitivity** — adjusts detection range (turn clockwise to increase)
- **Time delay** — adjusts how long OUT stays HIGH after detection

And a jumper for trigger mode:
- **H (repeatable)** — OUT stays HIGH as long as motion continues (recommended)
- **L (single)** — OUT pulses once per trigger event

## Connector Options

The `PIRMotionInput` plugin supports four hardware backends:

| Connector | Hardware | Use case |
|-----------|----------|----------|
| `serial` | Arduino / any USB microcontroller | Cross-platform, multi-sensor |
| `gpio` | Raspberry Pi / Jetson | Direct wiring, minimal hardware |
| `zenoh` | Any Zenoh-capable device | Distributed / multi-robot setups |
| `mock` | No hardware | Development and testing (default) |

## Wiring

### Arduino (serial connector)
```
HC-SR501 VCC  →  Arduino 5V
HC-SR501 GND  →  Arduino GND
HC-SR501 OUT  →  Arduino D2
```

Upload this sketch to the Arduino:
```cpp
const int PIR_PIN = 2;

void setup() {
    Serial.begin(9600);
    pinMode(PIR_PIN, INPUT);
}

void loop() {
    Serial.println(digitalRead(PIR_PIN) ? "MOTION:1" : "MOTION:0");
    delay(500);
}
```

### Raspberry Pi (gpio connector)
```
HC-SR501 VCC  →  RPi Pin 2  (5V)
HC-SR501 GND  →  RPi Pin 6  (GND)
HC-SR501 OUT  →  RPi Pin 11 (GPIO17, BCM)
```

> **Note:** HC-SR501 OUT is typically 3.3 V-safe, but verify your specific sensor's datasheet before connecting directly to a 3.3 V GPIO pin.

### Finding the Arduino on Linux
```bash
sudo dmesg | grep ttyUSB*
# or
sudo dmesg | grep ttyACM*
```

Read the data to verify the sensor is streaming:
```bash
screen /dev/ttyUSB0 9600
```

### Finding the Arduino on macOS
```bash
ls /dev/cu.*
```

It should appear as something like `/dev/cu.usbmodem1101`.

## Configuration

Add `PIRMotionInput` to your config's `agent_inputs`:
```json5
{
  agent_inputs: [
    {
      type: "PIRMotionInput",
      config: {
        connector: "serial",       // serial | gpio | zenoh | mock
        port: "/dev/ttyUSB0",      // serial only
        baudrate: 9600,            // serial only
        gpio_pin: 17,              // gpio only (BCM numbering)
        zenoh_topic: "om/sensors/pir",  // zenoh only
        cooldown: 5.0,             // seconds between LLM alerts
      },
    },
  ],
}
```

The `cooldown` parameter is important: HC-SR501 can hold its output HIGH for up to ~200 seconds depending on the time delay potentiometer setting. Without cooldown the LLM would receive hundreds of identical motion alerts. The default of 5.0 seconds is a safe starting point — increase it if your robot speaks too frequently.

## Running

A ready-to-use config is provided:
```bash
python -m run --config pir_guard
```

This runs the robot in security/guard mode using the mock connector by default. Change `connector` to `"serial"` or `"gpio"` for real hardware.

## Zenoh Distributed Setup

If the sensor is on a remote device (e.g. a Raspberry Pi on the robot body, while OM1 runs on a separate computer):
```python
import zenoh
import time
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BCM)
GPIO.setup(17, GPIO.IN)

session = zenoh.open()
pub = session.declare_publisher("om/sensors/pir")

while True:
    pub.put("1" if GPIO.input(17) else "0")
    time.sleep(0.5)
```

Then set `connector: "zenoh"` in your OM1 config.
