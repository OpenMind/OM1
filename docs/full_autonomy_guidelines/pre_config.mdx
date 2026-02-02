---
title: Pre-Config
description: "Pre Configuration Setup"
---

To get the complete system up and running, make sure you follow the steps mentioned below.

Our boot sequence is as follows:
    - Docker service starts
    - Wait 5 seconds to set up the speaker and microphone with echo cancellation
    - Wait 15 seconds before restarting om1

- Setup environment variables
    Get your API key from [OM1 Portal](https://portal.openmind.org/)

    For Bash:
    Open the bashrc file
    ```vim ~/.bashrc or ~/.bash_profile.``` and add ``` export OM_API_KEY=your API key ```

    For Unitree G1, Add the following as well
    ```bash
    export ROBOT_TYPE=g1
    export CYCLONEDDS_INTERFACE=enP2p1s0
    ```

    Note: By default, the docker container has `CYCLONEDDS_INTERFACE` set to `eno1`, it is necessary to update it to the correct interface for G1 setup. Ideally `ROBOT_TYPE` can also be setup for Go2 but it is not mandatory.

    The cyclone dds interface can be found by running `ifconfig` in your terminal and selecting the name of the ethernet connection. Usually for the dog it is `eno1` and for g1 it is `enP2p1s0`.

- Install ROS2-jazzy and CycloneDDS on the edge device. (Optional)
    [ROS2 installation link](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html#id7)

- Establish Ethernet and DDS connectivity

    Open the network settings and find the network interface that is connected to the Go2 EDU/G1. In the IPv4 settings, change the IPv4 mode to manual, set the address to 192.168.123.99, and set the mask to 255.255.255.0. After completion, click apply (or equivalent) and wait for the network to reconnect.

- Create cyclonedds.xml file inside /Documents/Github/cyclonedds/
    ```vi cyclonedds.xml```
    Add the following -
    ```bash
    <CycloneDDS>
        <Domain>
            <General>
            <Interfaces>
                <NetworkInterface name="eno1" priority="default" multicast="default" />
            </Interfaces>
            </General>
            <Discovery>
            <Peers>
                <Peer address="192.168.123.161"/>   <!-- Go2 Robot 1 -->
                <Peer address="192.168.123.100"/>   <!-- Control station 1 -->
                <Peer address="192.168.123.99"/>   <!-- Control station 2 -->
            </Peers>
            </Discovery>
        </Domain>
    </CycloneDDS>
    ```
    Make sure you configure the correct IP address for control station 2. It should be the same you configure in IPv4 settings above.

    For Unitree G1, make sure you update the following
    ```bash
    <NetworkInterface name="enP2p1s0" priority="default" multicast="default" />
    ```

- Set CYCLONEDDS_HOME, CMAKE_PREFIX_PATH, and CYCLONEDDS_URI (Optional)
    ```bash
    source /opt/ros/humble/setup.bash
    export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
    export CYCLONEDDS_HOME=/home/{username}/Documents/Github/cyclonedds/install
    export CYCLONEDDS_URI=~/Documents/Github/cyclonedds/cyclonedds.xml
    ```

- Install unclutter on your system to hide the mouse cursor after a period of inactivity:
    ```bash
    sudo apt install unclutter
    ```

- Install Chromium and lock the snapd version for stability:
    ```bash
    sudo snap install chromium
    snap download snapd --revision=24724
    sudo snap ack snapd_24724.assert
    sudo snap install snapd_24724.snap
    sudo snap refresh --hold snapd
    ```

- Add the script to /usr/local/bin/start-kiosk.sh and make it executable:
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

    exec chromium --kiosk http://$HOST:$PORT --disable-infobars --noerrdialogs
    ```

- Make the script executable:

    ```bash
    chmod +x /usr/local/bin/start-kiosk.sh
    ```

- Add the script to /etc/systemd/system/kiosk.service to launch the kiosk mode automatically on boot.
    ```bash
    # /etc/systemd/system/kiosk.service
    [Unit]
    Description=Kiosk Browser
    After=graphical.target docker.service
    Requires=docker.service

    [Service]
    Environment=DISPLAY=:0
    ExecStart=/usr/local/bin/start-kiosk.sh
    Restart=always
    User=openmind

    [Install]
    WantedBy=graphical.target
    ```

- Enable and start the kiosk service:
    ```bash
    sudo systemctl daemon-reload
    sudo systemctl enable kiosk.service
    sudo systemctl start kiosk.service
    ```
Note: To stop the kiosk service, use sudo systemctl stop kiosk.service.

- Setup default speaker and microphone:
    - Connect your robot with a microphone and a speaker if not already connected.
    - Install pulseaudio, if not already installed
        ```bash
        sudo apt install pulseaudio
        ```
    - Find the available devices using following commands -
        ```bash
        # for speakers
        pactl list sinks
        # for microphone
        pactl list sources
        ```
    **Make a note of the device names.**

    ```bash
    mkdir -p ~/.config/systemd/user
    vim ~/.config/systemd/user/audio-defaults.service
    ```
    Add the following to the file -
    ```bash
    [Unit]
    Description=Set Default Audio Devices
    After=pulseaudio.service
    Wants=pulseaudio.service

    [Service]
    Type=oneshot
    RemainAfterExit=yes
    ExecStart=/bin/bash -c '\
    sleep 5 && \
    pactl unload-module module-echo-cancel || true && \
    pactl load-module module-echo-cancel \
        aec_method=webrtc \
        source_master=alsa_input.usb-046d_Brio_101_2520APKJ1778-02.mono-fallback \
        sink_master=alsa_output.usb-Solid_State_System_Co._Ltd._USB_PnP_Audio_Device_000000000000-00.analog-stereo \
        source_name=default_mic_aec \
        sink_name=default_output_aec \
        source_properties="device.description=Microphone_with_AEC" \
        sink_properties="device.description=Speaker_with_AEC" && \
    pactl set-default-source default_mic_aec && \
    pactl set-default-sink default_output_aec'

    [Install]
    WantedBy=default.target
    ```

- Use
    ```bash
    pactl list short
    ```
    and replace alsa_output.usb-Solid_State_System_Co._Ltd._USB_PnP_Audio_Device_000000000000-00.analog-stereo with your speaker source and alsa_input.usb-046d_Brio_101_2520APKJ1778-02.mono-fallback with mic source

- Enable and start the audio defaults service:
        ```bash
        systemctl --user daemon-reload
        systemctl --user enable audio-defaults.service
        systemctl --user start audio-defaults.service
        ```

- Settings to be updated for edge device

    Go to your system settings and update the following
        - Set **screen blank** to **Never**
        - Disable **Automate suspend**
        - Enable **Autohide Dock**

- Restart the OM1 container to apply these changes when the system starts to do audio configuration:
    ```bash
    docker-compose restart om1
    ```

- Add this service to /etc/systemd/system/om1-container.service to automatically restart the OM1 container on boot (Optional):
    ```bash
    [Unit]
    Description=Restart OM1
    After=docker.service multi-user.target
    Wants=docker.service

    [Service]
    Type=simple
    ExecStart=/bin/bash -c 'sleep 15 && docker restart om1'
    RemainAfterExit=no

    [Install]
    WantedBy=multi-user.target

    sudo systemctl daemon-reload
    sudo systemctl enable om1-container.service
    sudo systemctl start om1-container.service
    ```

After completing all pre-configuration steps, clone each repository individually and build the corresponding Docker images. Once all services are successfully set up and running, the robot will operate in Full Autonomy Mode.
