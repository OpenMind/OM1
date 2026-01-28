---
title: Smart Toy
description: "Combining Inputs, an LLM, and Outputs to create a smart, engaging toy."
---

## LLM Model

This example takes two inputs, emotional state and voice inputs, sends them to an LLM, and produces speech outputs and physical movements. The overall behavior of the system is configured in `/config/cubly.json5`.

## Emotion Detection with Cubly

Run
```bash
uv run src/run.py cubly
```

You should see your webcam light turn on and Cubly should speak to you from your default laptop speaker. You can see what is happening in the `WebSim` simulator window.

* There will be an initial delay for your system to download various packages and AI/ML models.
* Arduino based movement generation only works if you actually have a suitable actuator connected to an Arduino, which is connected to your computer via a USB serial dongle. On Mac, you can determine the correct serial port name to use via `ls /dev/cu.usb*`. If you do not specify your computer's serial port, the example will provide logging data that simulates what it would send.
* The **emotion estimation input** is provided by webcam data feeding into `cv2.CascadeClassifier(haarcascade_frontalface_default)`. See `/inputs/plugins/webcam_to_face_emotion`.
* The **voice input** originates from the default microphone, which sends data to the cloud instance of Nvidia's RIVA. See `/inputs/plugins/asr`.
* The **voice output** uses a cloud text to speech endpoint, which sends audio data to the default speaker. See `/actions/speak/connector/tts`.
* The **Arduino serial movement actions** are sent to serial com port `COM1`, flowing to a connected Arduino, which can then generate servo commands. See `/actions/move_serial_arduino/connector/serial_arduino`.
