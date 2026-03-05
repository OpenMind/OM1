## Install ROS2-humble

#### Set locale

Make sure you have a locale which supports UTF-8.

If you are in a minimal environment (such as a docker container), the locale may be something minimal like POSIX. We test with the following settings. However, it should be fine if you’re using a different UTF-8 supported locale.

```bash
locale  # check for UTF-8

sudo apt update && sudo apt install locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

locale  # verify settings
```

#### Setup Sources

You will need to add the ROS 2 apt repository to your system.

First ensure that the Ubuntu Universe repository is enabled.

```bash
sudo apt install software-properties-common
sudo add-apt-repository universe
```

The ros-apt-source packages provide keys and apt source configuration for the various ROS repositories.

Installing the `ros2-apt-source` package will configure ROS 2 repositories for your system. Updates to repository configuration will occur automatically when new versions of this package are released to the ROS repositories.

Now, run

```bash
sudo apt update
sudo apt upgrade
sudo apt install ros-humble-desktop
```

This will install ROS and all the relevant packages.
