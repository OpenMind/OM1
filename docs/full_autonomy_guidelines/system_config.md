---
title: Configure System
description: "System Configuration"
icon: gear
---

## System Services

If you don't have the BrainPack, you can skip this section.

### Screen Animation Service

To enable the screen animation service, install unclutter first to hide the mouse cursor:

```bash
sudo apt install unclutter
```

Then, add the script to `/usr/local/bin/start-kiosk.sh` and make it executable:

```bash
#!/bin/bash

unclutter -display :0 -idle 0.1 -root &

HOST=localhost
PORT=4173

# Wait for Docker service to listen
while ! nc -z $HOST $PORT; do
  echo "Waiting for $HOST:$PORT..."
  sleep 0.1
done

# Launch with autoplay permissions
exec chromium \
  --kiosk http://$HOST:$PORT \
  --start-fullscreen \
  --disable-infobars \
  --noerrdialogs \
  --autoplay-policy=no-user-gesture-required \
  --disable-features=PreloadMediaEngagementData,MediaEngagementBypassAutoplayPolicies \
  --no-first-run \
  --disable-session-crashed-bubble \
  --disable-translate \
  --window-position=0,0
```

Make it executable:

```bash
sudo chmod +x /usr/local/bin/start-kiosk.sh
```

Add the script to /etc/systemd/system/kiosk.service to launch the kiosk mode automatically on boot.

```bash
# /etc/systemd/system/kiosk.service
[Unit]
Description=Kiosk Browser
After=docker.service
Requires=docker.service

[Service]
Environment=DISPLAY=:0
ExecStart=/usr/local/bin/start-kiosk.sh
Restart=always
User=openmind

[Install]
WantedBy=graphical.target
Enable and start the service:

sudo systemctl daemon-reload
sudo systemctl enable kiosk.service
sudo systemctl start kiosk.service
```

>**Note:** To stop the kiosk service, use `sudo systemctl stop kiosk.service`.

### AEC Service

To enable the Acoustic Echo Cancellation (AEC) service, uninstall PipWire if it's installed and install PulseAudio

```bash
sudo apt remove --purge pipewire-audio-client-libraries pipewire-pulse wireplumber
```

Then install PulseAudio:

```bash
sudo apt install pulseaudio pulseaudio-module-bluetooth pulseaudio-utils pavucontrol
```

Next, stop the PipWire daemon and start the PulseAudio daemon if it's not already running:

```bash
systemctl --user mask pipewire.service
systemctl --user mask pipewire.socket
systemctl --user mask pipewire-pulse.service
systemctl --user mask pipewire-pulse.socket
systemctl --user mask wireplumber.service
systemctl --user stop pipewire-pulse.service
systemctl --user stop pipewire.service wireplumber.service
systemctl --user disable pipewire.service wireplumber.service
systemctl --user enable --now pulseaudio.service
```

Next, add the script to prevent PulseAudio from going into auto-exit mode.

```bash
mkdir -p ~/.config/pulse
cat > ~/.config/pulse/client.conf << 'EOF'
autospawn = yes
daemon-binary = /usr/bin/pulseaudio
EOF

# Create daemon config to disable idle timeout
cat > ~/.config/pulse/daemon.conf << 'EOF'
exit-idle-time = -1
EOF
```

Now, you can restart the system to ensure PulseAudio is running properly.

```bash
sudo reboot
```

[!Note]
After reboot, if the audio devices are not automatically detected, you may need to manually start PulseAudio with the command:

```bash
systemctl --user restart pulseaudio
```

Now, you can add the script to `/usr/local/bin/set-audio-defaults.sh` and make it executable:

```bash
#!/bin/bash
set -e

sleep 5

# First, set the master source volume to 200%
pactl set-source-volume "alsa_input.usb-R__DE_R__DE_VideoMic_GO_II_FEB0C614-00.mono-fallback" 131072
pactl set-source-mute "alsa_input.usb-R__DE_R__DE_VideoMic_GO_II_FEB0C614-00.mono-fallback" 0

# Unload then load AEC module
pactl unload-module module-echo-cancel || true
pactl load-module module-echo-cancel \
  use_master_format=1 \
  aec_method=webrtc \
  source_master="alsa_input.usb-R__DE_R__DE_VideoMic_GO_II_FEB0C614-00.mono-fallback" \
  sink_master="alsa_output.platform-88090b0000.had.hdmi-stereo" \
  source_name="default_mic_aec" \
  sink_name="default_output_aec" \
  source_properties="device.description=Microphone_with_AEC" \
  sink_properties="device.description=Speaker_with_AEC"

# Wait a moment for the module to fully initialize
sleep 2

# Set defaults
pactl set-default-source default_mic_aec
pactl set-default-sink default_output_aec

# Retry volume setting until device appears and volume is set correctly
for i in {1..15}; do
  if pactl list short sources | grep -q default_mic_aec; then
    # Set volume to 200% (131072)
    pactl set-source-volume default_mic_aec 131072
    pactl set-source-mute default_mic_aec 0

    # Verify the volume was set
    CURRENT_VOL=$(pactl list sources | grep -A 7 "Name: default_mic_aec" | grep "Volume:" | awk '{print $3}')

    if [ "$CURRENT_VOL" = "131072" ]; then
      echo "Microphone volume successfully set to 200%"
      break
    else
      echo "Volume is $CURRENT_VOL, retrying... ($i/15)"
    fi
  else
    echo "Waiting for AEC source to appear... ($i/15)"
  fi
  sleep 1
done

# Final verification
pactl list sources | grep -A 7 "Name: default_mic_aec" | grep -E "Name:|Volume:"
```

Use the following command to get the list of audio sources and sinks:

```bash
pactl list short
```

>**Note:**

Replace `alsa_output.platform-88090b0000.had.hdmi-stereo with your speaker source and `alsa_input.usb-R__DE_R__DE_VideoMic_GO_II_FEB0C614-00.mono-fallback with mic source

Make it executable:

```bash
sudo chmod +x /usr/local/bin/set-audio-defaults.sh
```

Create a systemd user service to run the script on login:

```bash
mkdir -p ~/.config/systemd/user
sudo vim ~/.config/systemd/user/audio-defaults.service
```

Add the following content:

```bash
[Unit]
Description=Set Default Audio Devices
After=pulseaudio.service
Wants=pulseaudio.service

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/usr/local/bin/set-audio-defaults.sh

[Install]
WantedBy=default.target
Enable and start the service:

systemctl --user daemon-reload
systemctl --user enable audio-defaults.service
systemctl --user start audio-defaults.service
```

Once you're done with above steps, you can proceed with OTA setup [here](./ota_setup.md)
