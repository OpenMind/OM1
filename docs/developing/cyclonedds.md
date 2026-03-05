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

### CycloneDDS config for Unitree Go2

### CycloneDDS config for Unitree G1

### CycloneDDS config for Unitree Simulation (Gazebo or Isaac Sim)
