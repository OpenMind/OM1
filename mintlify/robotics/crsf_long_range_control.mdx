---
title: Long Range RF Robot Control
description: "CRSF Protocol Support"
---

## Introduction

Current safety-focused manual control of robot motion typically uses Bluetooth game controllers, but these are not particularly reliable. The biggest problem is that most game controllers go to sleep after several minutes of inactivity, resulting in control gaps. True panic consists of giving OpenAI's o4-mini full control of a humanoid and then watching it head off to the nearest bus station while you realize that your "disable AI" game controller has gone to sleep to save battery. A more robust solution are professional drone control radios, such as the [TBS Tango2 II transmitter/receiver combination](https://www.team-blacksheep.com/products/prod:tbs_tango_2).

## Pairing

Set up the radio/receiver per the Tango II instructions. This is can be unexpectedly difficult, based on numerous PCB and firmware revisions and outdated/misleading documentation. Expect to spend several hours trying to find current instructions relevant to your transmitter/receiver, updating the transmitter/receiver firmware, and finally getting them to pair correctly.

*UPDATE* The best solution is the new [TBS Agent M browser based configurator](https://www.team-blacksheep.com/agentm/). This makes updating the entire system less painful.

## Hardware Configuration

The traditional friction/ratchet throttle control is probably not what you want. You can disable the throttle ratchet and friction, and enable the return to center stick tension, by removing the back of the transmitter and following these instructions: [Adjusting throttle ratchet strength](https://www.team-blacksheep.com/media/files/tbs-tango2-manual.pdf).

## Software Configuration

The radios can have the throttle control on the left or right side. The default throttle on the TBS Tango 2 is on the **left** stick in the default Mode 2 setting. The OpenMind CRSF driver assumes mode 2, otherwise the channel mapping will be messed up. You can change the mode setting in the transmitter menus.

* Long-press the Menu button to enter the Radio Setup
* Press the Page button to get to page 3 of 7, which is the RADIO SETUP
* Scroll down to the very end using the Rocker-switch (the rotary control dial at the right)
* Change the Mode to **2**

More information - see [page 16 of the Tango TBS II manual](https://www.team-blacksheep.com/media/files/tbs-tango2-manual.pdf)

Other suggested settings for the radio are:

* Internal RF
	- CRSF
	- Ch. range CH1-16
	- Receiver 00
* External RF
	- Mode OFF

We suggest using a Baud setting of `420000`.

## Cabling

Connect the receiver to a TTL->USB adapter, such as the [WaveShare Industrial USB to TTL Converter with Original FT232RNL](https://www.amazon.com/Waveshare-Industrial-USB-TTL-Protection/dp/B087RJ7X32).

*WARNING* the fake CP2102 modules from Amazon do not work, but give CRC errors for unknown reasons.

## Usage

Find the serial port via:

```bash
ls /dev/tty.*
```

It should be something like `/dev/tty.usbserial-0001` or `/dev/tty.usbserial-B003ABY3`. Then, start the driver:

```
uv run parse_crsf_radio.py --port /dev/tty.usbserial-B003ABY3
```

## Code Origin

The code is based on Bryan Mayland's CRSF ["Python Parser"](https://github.com/crsf-wg/crsf/wiki/Python-Parser). The code also uses the [public CRSF documentation](https://github.com/tbs-fpv/tbs-crsf-spec).

## Known Issues

Occasionally, one of the RC_Channels has a value of >2000, which is noise. This is a problem, since it could signal the receiver that a switch was pressed, even though it was not. We therefore reject all data >2000.

A typical range for a valid RC signal is 174 to 1806 - this is true for both sticks and switches. We therefore clip all stick control values to the 174 to 1806 range, and then map this range to the `0.0<>1.0` interval.
