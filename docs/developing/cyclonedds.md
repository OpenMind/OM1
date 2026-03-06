---
title: CycloneDDS
description: "Installation"
icon: gear
---

## Install CycloneDDS

Install cyclonedds from this [link](https://cyclonedds.io/docs/cyclonedds/latest/installation/installation.html) or follow the instructions below.

```bash
sudo apt-get install git cmake gcc
```

```bash
git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x
cd cyclonedds && mkdir build install && cd build
cmake -DBUILD_EXAMPLES=ON -DCMAKE_INSTALL_PREFIX=$HOME/Documents/GitHub/cyclonedds/install ..
cmake --build . --target install
```

### CycloneDDS config

#### for Unitree Simulation (Gazebo or Isaac Sim)

Use this CycloneDDS configuration for running simulation. It uses `lo` as the network interface. We recommend that you export this in your .bashrc or equivalent configuration file cyclonedds.xml. To add it to cyclonedds.xml:

```bash
cd cyclonedds
vi cyclonedds.xml
```

Add the following, then save and exit.

```bash
<CycloneDDS>
    <Domain>
        <General>
            <Interfaces>
                <NetworkInterface address="127.0.0.1" priority="default" multicast="default" />
            </Interfaces>
        </General>
        <Discovery>
            <MaxAutoParticipantIndex>200</MaxAutoParticipantIndex>
        </Discovery>
    </Domain>
</CycloneDDS>
```

Open your bashrc file

```bash
vi ~/.bashrc
```

Add the following, replacing /path/to/cyclonedds with the actual path to your CycloneDDS installation:

```bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=/path/to/cyclonedds/cyclonedds.xml
```

Apply the changes

```bash
source ~/.bashrc
```

To add the config to your bashrc, run:

```bash
vim ~/.bashrc
```

And add the following, replacing `/path/to/cyclonedds` with the actual path to your CycloneDDS installation:

```bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI='
<CycloneDDS>
    <Domain>
        <General>
            <Interfaces>
                <NetworkInterface address="127.0.0.1" priority="default" multicast="default" />
            </Interfaces>
        </General>
        <Discovery>
            <MaxAutoParticipantIndex>200</MaxAutoParticipantIndex>
        </Discovery>
    </Domain>
</CycloneDDS>'
```

Now run

```bash
source ~/.bashrc
```
This will apply the latest changes in the current shell session.
